# MPS Solver Results

## Problem: air05.mps (MIPLIB benchmark)

**Description:** Airline crew scheduling - set partitioning problem

### Problem Characteristics
- **Variables:** 7195 (all binary)
- **Constraints:** 426
- **Nonzeros:** 52121
- **Best Known Optimal:** 26374

---

## Gap Tolerance Comparison

Comparing different MIP relative gap tolerances to show trade-off between solution quality and solve time.

### Run Configuration
- **Time Limit:** 60 seconds
- **cuOpt Version:** 26.2.0
- **Device:** Quadro RTX 8000 (47.24 GiB VRAM)
- **CPU:** AMD Ryzen Threadripper PRO 3975WX (32 cores)

### Results Summary

| Gap Tolerance | Objective | Gap to Optimal | Solve Time | Nodes Explored |
|--------------|-----------|----------------|------------|----------------|
| 0.1% | **26374** | 0.00% | 8.42s | 386 |
| 1.0% | 26491 | 0.44% | 3.23s | 328 |

### Key Observations

1. **Tighter gap finds optimal**: The 0.1% gap tolerance found the exact best-known optimal solution (26374)
2. **Trade-off**: The looser 1.0% gap converged faster (3.2s vs 8.4s) but with 0.44% suboptimality
3. **Both are fast**: cuOpt solved this 7195-variable MILP in under 10 seconds

---

## Detailed Solver Output (0.1% gap)

```
Solving a problem with 426 constraints, 7195 variables (7195 integers), and 52121 nonzeros

Presolve removed: 90 constraints, 1116 variables, 16171 nonzeros
Presolved problem: 336 constraints, 6079 variables, 35950 nonzeros

Root relaxation objective +2.58776093e+04

Strong branching using 7 threads and 222 fractional variables
Explored 386 nodes in 7.73s.

Optimal solution found within relative MIP gap tolerance (1.0e-03)
Solution objective: 26374.000000
relative_mip_gap 0.000992
total_solve_time 8.421934
```

---

## Detailed Solver Output (1.0% gap)

```
Solving a problem with 426 constraints, 7195 variables (7195 integers), and 52121 nonzeros

Presolve removed: 90 constraints, 1116 variables, 16171 nonzeros
Presolved problem: 336 constraints, 6079 variables, 35950 nonzeros

Root relaxation objective +2.58776093e+04

Strong branching using 63 threads and 222 fractional variables
Explored 328 nodes in 1.09s.

Optimal solution found within relative MIP gap tolerance (1.0e-02)
Solution objective: 26491.000000
relative_mip_gap 0.009669
total_solve_time 3.233650
```

---

## Usage

```bash
# Default: download air05.mps and solve with comparison
python model.py --compare --time-limit 60

# Solve custom MPS file
python model.py --file path/to/problem.mps --time-limit 300 --mip-gap 0.001
```
