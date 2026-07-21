# MPS File Solver

Read and solve LP/MILP problems from standard MPS files using cuOpt.

## Problem Description

MPS (Mathematical Programming System) is a standard file format for representing linear and mixed-integer programming problems. This model demonstrates how to:

1. Load an MPS file using `Problem.readMPS()` (static method)
2. Solve the problem using cuOpt's GPU-accelerated solver
3. Extract and display the solution

This is useful when you have optimization problems in standard MPS format from other solvers, modeling tools, or benchmark libraries like MIPLIB.

## MPS File Format

MPS is a column-oriented format with sections:

```
NAME          problem_name
ROWS
 N  OBJ                    (objective row)
 L  CON1                   (≤ constraint)
 G  CON2                   (≥ constraint)
 E  CON3                   (= constraint)
COLUMNS
    X1        OBJ        1.0
    X1        CON1       2.0
    X2        OBJ        2.0
    X2        CON1       3.0
RHS
    RHS       CON1       10.0
BOUNDS
 LO BND       X1         0.0
 UP BND       X1         5.0
ENDATA
```

## Usage

```bash
# Solve the sample problem
python model.py

# Solve a custom MPS file
python model.py --file path/to/problem.mps

# With time limit
python model.py --file problem.mps --time-limit 120
```

## Model Characteristics

- **Type**: LP or MILP (detected from MPS file)
- **Input**: Standard MPS file format
- **Output**: Solution values, objective, status

## Sample Problem

The included `data/air05.mps` is a MIPLIB benchmark (airline crew scheduling):

- **Variables**: 7,195 (binary)
- **Constraints**: 426
- **Known optimal**: 26,374
- **Typical solve time**: ~2 seconds

## Key API Usage

```python
from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.solver_settings import SolverSettings

# Load MPS file (static method - returns Problem object)
problem = Problem.readMPS("path/to/problem.mps")

# Configure and solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Check solution
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Objective: {problem.ObjValue}")
```

## Source

Based on cuOpt's built-in MPS support via `Problem.readMPS()`.
