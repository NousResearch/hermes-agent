# LoRA Rank Convergence Study

**Research Question:** Does LoRA rank affect convergence speed on small datasets?

## Bootstrap

### Literature

Key papers:
- Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models
- Valipour et al. (2023) — DyLoRA: Parameter-Efficient Tuning with Dynamic Search

Gap: Most papers focus on final performance, not convergence dynamics.

### Hypotheses

- **H1:** Higher rank (r=16) converges faster but may overfit on small data
- **H2:** Lower rank (r=4) converges slower but generalizes better
- **H3:** There's an optimal rank (r=8) that balances speed and generalization

## Experiments

### H001 — Baseline (r=8)

```bash
# Protocol: Train with rank 8, measure convergence steps to 90% of max accuracy
# Prediction: Baseline behavior, ~50 steps to converge
```

**Results:**
- Convergence steps: 47
- Final accuracy: 0.892
- Wall time: 12 min

### H002 — Low Rank (r=4)

**Results:**
- Convergence steps: 68 (+44% vs baseline)
- Final accuracy: 0.887 (-0.6%)

### H003 — High Rank (r=16)

**Results:**
- Convergence steps: 41 (-13% vs baseline)
- Final accuracy: 0.894 (+0.2%)

## Outer Loop #1

**Pattern:** Higher rank → faster convergence, minimal overfit on this dataset

**Decision:** DEEPEN — Test r=32 and r=64 to find saturation point

### H004 — Very High Rank (r=32)

**Results:**
- Convergence steps: 38 (-6% vs r=16)
- Final accuracy: 0.891 (-0.3%)
- **Diminishing returns observed**

### H005 — Optimal Search (r=6, r=10, r=12)

[Running...]

## Current Findings

1. Convergence speed improves with rank up to r=16, then plateaus
2. Final accuracy relatively stable across ranks (±0.5%)
3. For small datasets, r=8-12 appears optimal (speed vs compute tradeoff)

## Next Steps

- Complete H005-H007
- Test on different dataset sizes (generalization)
- Write up findings
