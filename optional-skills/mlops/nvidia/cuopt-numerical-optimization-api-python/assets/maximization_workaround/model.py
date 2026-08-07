# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Maximize -x² + 4x (max at x=2) by minimizing x² - 4x; then report -objective.
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE

problem = Problem("MaxWorkaround")

x = problem.addVariable(lb=0, ub=10, vtype=CONTINUOUS, name="x")
problem.setObjective(x * x - 4 * x, sense=MINIMIZE)

problem.solve()

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"x = {x.getValue():.4f}")
    print(f"Minimized value = {problem.ObjValue:.4f}")
    print(f"Original maximum = {-problem.ObjValue:.4f}")
else:
    print(f"Status: {problem.Status.name}")
