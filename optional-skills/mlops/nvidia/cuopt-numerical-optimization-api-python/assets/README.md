# Assets — reference models

LP, MILP, and QP reference implementations. Use as reference when building new applications; do not edit in place.

| Model | Type |
|-------|------|
| lp_basic | LP |
| lp_duals | LP |
| lp_warmstart | LP |
| milp_basic | MILP |
| milp_production_planning | MILP |
| mps_solver | LP/MILP |
| portfolio | QP |
| least_squares | QP |
| maximization_workaround | QP |

**Run:** From each subdir, `python model.py`. QP is **beta** and supports **MINIMIZE** only. See [resources/qp_examples.md](../resources/qp_examples.md) for additional QP examples.
