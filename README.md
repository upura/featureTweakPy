featureTweakPy
===
How can prediction result be changed?

## Referrence
[This ipynb](https://github.com/upura/featureTweakPy/blob/master/Interpretable-Predictions-of-Tree-based-Ensembles.ipynb) is inspired by the following link:  
http://setten-qb.hatenablog.com/entry/2017/10/22/232016

I fixed some codes and added some explanations:

- Fix `load_iris()` to `datasets.load_iris()` at In [2]
- Fix `rfc.fit(x, y)` to `rfc.fit(x_arr, y_arr)` at In [3]
- Fix `aim_label = 3` to `aim_label = 2` at In [7] and [22]
- Add the usage of feature_tweaking()
- Add featureTweakPy.py to extract functions

## Description
Python implementation of Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking (KDD 2017)  
https://arxiv.org/abs/1706.06691

## Requirements
- Python 3.x
  - numpy
  - pandas
  - scipy.stats

## Usage

### Download
```
git clone git@github.com:upura/featureTweakPy.git
cd featureTweakPy
```

### Package install
if necessary

### Package import

```
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from featureTweakPy import feature_tweaking
```

### Dataset import

```
iris = datasets.load_iris()
x_arr = iris['data']
y_arr = iris['target']
```

### Random Forest Prediction

```
rfc = RandomForestClassifier()
rfc.fit(x_arr, y_arr)
```

### Using function()
#### Hyper Parameters Setting

```
class_labels = [0, 1, 2]  # Same as input[7]
aim_label = 2
epsilon = 0.1
class_labels = [0, 1, 2]  # Same as input[7]
aim_label = 2
epsilon = 0.1
```

#### Cost Function Setting

```
def cost_func(a, b):
    return np.linalg.norm(a-b)
```

#### Sample Data for Demonstration

```
x = x_arr[0]
x
```
```
array([5.1, 3.5, 1.4, 0.2])
```

#### Using feature_tweaking()

```
feature_tweaking(rfc, x, class_labels, aim_label, epsilon, cost_func)
x_new = feature_tweaking(rfc, x, class_labels, aim_label, epsilon, cost_func)
x_new
```
```
array([5.1       , 2.9999999 , 4.75000038, 0.90000001])
```

