# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Least squares: minimize (x-3)² + (y-4)². Solution should be x=3, y=4.
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("LeastSquares")

x = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="y")

problem.setObjective(x * x + y * y - 6 * x - 8 * y + 25, sense=MINIMIZE)

problem.solve(SolverSettings())

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"x = {x.getValue():.4f}")
    print(f"y = {y.getValue():.4f}")
else:
    print(f"Status: {problem.Status.name}")
