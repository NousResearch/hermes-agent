# QP: Python API Examples

## Portfolio Optimization

```python
"""
Minimize portfolio variance (risk):
    minimize    x^T * Q * x
    subject to  sum(x) = 1         (fully invested)
                r^T * x >= target  (minimum return)
                x >= 0             (no short selling)

Note: QP is beta and MUST use MINIMIZE (not MAXIMIZE)
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("Portfolio")

# Portfolio weights (decision variables)
x1 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_c")

# Expected returns
r1, r2, r3 = 0.12, 0.08, 0.05  # 12%, 8%, 5%
target_return = 0.08

# Covariance matrix Q:
# [[0.04, 0.01, 0.005],
#  [0.01, 0.02, 0.008],
#  [0.005, 0.008, 0.01]]
#
# Quadratic objective: x^T * Q * x
# Expanded: 0.04*x1² + 0.02*x2² + 0.01*x3² + 2*0.01*x1*x2 + 2*0.005*x1*x3 + 2*0.008*x2*x3

problem.setObjective(
    0.04*x1*x1 + 0.02*x2*x2 + 0.01*x3*x3 +
    0.02*x1*x2 + 0.01*x1*x3 + 0.016*x2*x3,
    sense=MINIMIZE  # MUST be MINIMIZE for QP!
)

# Linear constraints
problem.addConstraint(x1 + x2 + x3 == 1, name="budget")
problem.addConstraint(r1*x1 + r2*x2 + r3*x3 >= target_return, name="min_return")

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Results
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Portfolio variance: {problem.ObjValue:.6f}")
    print(f"Portfolio std dev: {problem.ObjValue**0.5:.4f}")
    print(f"\nAllocation:")
    print(f"  Stock A: {x1.getValue()*100:.2f}%")
    print(f"  Stock B: {x2.getValue()*100:.2f}%")
    print(f"  Stock C: {x3.getValue()*100:.2f}%")

    actual_return = r1*x1.getValue() + r2*x2.getValue() + r3*x3.getValue()
    print(f"\nExpected return: {actual_return*100:.2f}%")
```

## Least Squares

```python
"""
Minimize ||Ax - b||² = (Ax-b)^T(Ax-b)

Example: Find point closest to (3, 4)
minimize (x-3)² + (y-4)² = x² - 6x + 9 + y² - 8y + 16
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("LeastSquares")

x = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="y")

# Quadratic objective: (x-3)² + (y-4)²
# Expanded: x² + y² - 6x - 8y + 25
problem.setObjective(
    x*x + y*y - 6*x - 8*y + 25,
    sense=MINIMIZE
)

result = problem.solve(SolverSettings())

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"x = {x.getValue():.4f}")  # Should be ~3
    print(f"y = {y.getValue():.4f}")  # Should be ~4
else:
    raise RuntimeError(f"Solver failed with status: {problem.Status.name}")
```

## Quadratic with Linear Constraints

```python
"""
minimize    x² + y² + z²
subject to  x + y + z = 10
            x >= 0, y >= 0, z >= 0
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE

problem = Problem("QuadraticConstrained")

x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")
z = problem.addVariable(lb=0, vtype=CONTINUOUS, name="z")

problem.setObjective(x*x + y*y + z*z, sense=MINIMIZE)
problem.addConstraint(x + y + z == 10)

problem.solve()

if problem.Status.name == "Optimal":
    print(f"x = {x.getValue():.4f}")
    print(f"y = {y.getValue():.4f}")
    print(f"z = {z.getValue():.4f}")
    print(f"Objective = {problem.ObjValue:.4f}")
```

## Maximization Workaround

```python
"""
QP only supports MINIMIZE.
To maximize f(x), minimize -f(x).

Example: maximize -x² + 4x  (parabola with max at x=2)
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE

problem = Problem("MaxWorkaround")

x = problem.addVariable(lb=0, ub=10, vtype=CONTINUOUS, name="x")

# Want to maximize: -x² + 4x
# Instead minimize: -(-x² + 4x) = x² - 4x
problem.setObjective(x*x - 4*x, sense=MINIMIZE)

problem.solve()

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"x = {x.getValue():.4f}")  # Should be 2
    print(f"Minimized value = {problem.ObjValue:.4f}")  # Should be -4
    print(f"Original maximum = {-problem.ObjValue:.4f}")  # Should be 4
else:
    print(f"Solver did not find optimal solution. Status: {problem.Status.name}")
```

## Expanding Covariance Matrix

Given covariance matrix Q and weight vector x:

```python
# Covariance matrix
Q = [
    [0.04, 0.01, 0.005],
    [0.01, 0.02, 0.008],
    [0.005, 0.008, 0.01]
]

# Expansion: x^T * Q * x
# = Q[0,0]*x1² + Q[1,1]*x2² + Q[2,2]*x3²
#   + 2*Q[0,1]*x1*x2 + 2*Q[0,2]*x1*x3 + 2*Q[1,2]*x2*x3
#
# = 0.04*x1*x1 + 0.02*x2*x2 + 0.01*x3*x3
#   + 0.02*x1*x2 + 0.01*x1*x3 + 0.016*x2*x3

objective = (
    Q[0][0]*x1*x1 + Q[1][1]*x2*x2 + Q[2][2]*x3*x3 +
    2*Q[0][1]*x1*x2 + 2*Q[0][2]*x1*x3 + 2*Q[1][2]*x2*x3
)
```

## Critical Reminders

1. **MINIMIZE only** - solver rejects MAXIMIZE for QP
2. **Convexity** - Q should be positive semi-definite
3. **Beta status** - API may change in future versions
4. **Status checking** - use PascalCase: `"Optimal"` not `"OPTIMAL"`

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File | Description |
|---------|------|-------------|
| Simple QP | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/simple_qp_example.py` | Basic QP setup |
| QP with Matrix | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/qp_matrix_example.py` | CSR matrix format for Q |

These examples are tested by CI (`ci/test_doc_examples.sh`) and represent canonical usage.
