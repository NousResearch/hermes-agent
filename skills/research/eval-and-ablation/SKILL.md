---
name: eval-and-ablation
description: Plan and interpret model evaluations and ablations for post-training research. Use when comparing checkpoints, designing ablations, selecting benchmarks, or summarizing what changed after training.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [evaluation, ablation, benchmarking, analysis]
---

# Eval And Ablation

## Workflow

1. Decide the primary evaluation question first.
2. Compare against the strongest relevant baseline, not just the previous checkpoint.
3. Split outputs into:
- headline metrics
- failure cases
- regressions
- cost or latency tradeoffs
4. Design ablations that isolate one factor at a time:
- data
- reward or preference signal
- prompt format
- optimizer or hyperparameters
- checkpoint selection
5. Write down what changed and what remains uncertain.

## Minimum report

- benchmark table
- one paragraph of interpretation
- one paragraph on regressions or ambiguity
- concrete next step
