import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from math import sqrt

#Only main Device
MAX_THREADS_PER_BLOCK = \
    drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

BLOCK_SIZE = int(sqrt(MAX_THREADS_PER_BLOCK))

import numpy as np
from ...utils import check_arrays, unique_labels

def root_mean_square_error(y_real, y_pred):
    """
    It computes the root mean squared difference (RMSE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like

    y_pred : array-like

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the mean square error

    """
    y_real, y_pred = check_arrays(y_real, y_pred)

    dim = y_real.shape[0]

    solution = np.zeros(dim)
    solution = solution.astype(np.float32)
    
    mod = SourceModule("""
        #include <math.h>
        
        __global__ void rmse(int *x, int *y, float *solution, int dim) {
            int idx = threadIdx.x;
            solution[idx] = pow((x[idx*2] - y[idx*2]) * 1.0, 2);
        }
    """)

    func = mod.get_function('rmse')
    func(drv.In(y_real), drv.In(y_pred), drv.Out(solution), np.int32(dim), block=(dim, 1, 1))

    return np.sqrt(np.sum(solution) / dim)

def mean_absolute_error(y_real, y_pred):
    """
    It computes the average absolute difference (MAE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like

    y_pred : array-like

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the mean absolute error


    """
    y_real, y_pred = check_arrays(y_real, y_pred)
    
    dim = y_real.shape[0]

    solution = np.zeros(dim)
    solution = solution.astype(np.float32)
    
    mod = SourceModule("""
        #include <math.h>
        
        __global__ void rmse(int *x, int *y, float *solution, int dim) {
            int idx = threadIdx.x;
            solution[idx] = abs(x[idx*2] - y[idx*2]);
        }
    """)

    func = mod.get_function('rmse')
    func(drv.In(y_real), drv.In(y_pred), drv.Out(solution), np.int32(dim), block=(dim, 1, 1))

    return np.sum(solution) / y_real.size

def normalized_mean_absolute_error(y_real, y_pred, max_rating, min_rating):
    """
    It computes the normalized average absolute difference (NMAE)
    between predicted and actual ratings for users.

    Parameters
    ----------
    y_real : array-like
        The real ratings.

    y_pred : array-like
        The predicted ratings.

    max_rating:
        The maximum rating of the model.

    min_rating:
        The minimum rating of the model.

    Returns
    -------

    Positive floating point value: the best value is 0.0.

    return the normalized mean absolute error


    """
    y_real, y_pred = check_arrays(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return mae / (max_rating - min_rating)

def evaluation_error(y_real, y_pred, max_rating, min_rating):
    """
    It computes the NMAE, MAE and RMSE between predicted
    and actual ratings for users.

    Parameters
    ----------
    y_real : array-like
        The real ratings.

    y_pred : array-like
        The predicted ratings.

    max_rating:
        The maximum rating of the model.

    min_rating:
        The minimum rating of the model.

    Returns
    -------
    mae: Positive floating point value: the best value is 0.0.
    nmae: Positive floating point value: the best value is 0.0.
    rmse: Positive floating point value: the best value is 0.0.

    """
    mae = mean_absolute_error(y_real, y_pred)
    nmae = normalized_mean_absolute_error(y_real, y_pred,
             max_rating, min_rating)
    rmse = root_mean_square_error(y_real, y_pred)

    return mae, nmae, rmse