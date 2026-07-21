# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Portfolio: minimize variance x'Qx subject to sum(x)=1, r'x >= target, x >= 0.
QP is beta; MUST use MINIMIZE.
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("Portfolio")

x1 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_c")

r1, r2, r3 = 0.12, 0.08, 0.05
target_return = 0.08

problem.setObjective(
    0.04 * x1 * x1
    + 0.02 * x2 * x2
    + 0.01 * x3 * x3
    + 0.02 * x1 * x2
    + 0.01 * x1 * x3
    + 0.016 * x2 * x3,
    sense=MINIMIZE,
)
problem.addConstraint(x1 + x2 + x3 == 1, name="budget")
problem.addConstraint(
    r1 * x1 + r2 * x2 + r3 * x3 >= target_return, name="min_return"
)

settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Portfolio variance: {problem.ObjValue:.6f}")
    print(f"Std dev: {problem.ObjValue**0.5:.4f}")
    print(f"  Stock A: {x1.getValue() * 100:.2f}%")
    print(f"  Stock B: {x2.getValue() * 100:.2f}%")
    print(f"  Stock C: {x3.getValue() * 100:.2f}%")
    print(
        f"Expected return: {(r1 * x1.getValue() + r2 * x2.getValue() + r3 * x3.getValue()) * 100:.2f}%"
    )
else:
    print(f"Status: {problem.Status.name}")
