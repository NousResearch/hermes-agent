# OBLITERATUS methods

CLI `--method` choices (package `0.1.3`): `basic`, `advanced`, `aggressive`, `spectral_cascade`, `informed`, `surgical`, `optimized`, `som`, `inverted`, `nuclear`.

`--direction-method`: `diff_means`, `svd`, `leace`, `som`.

`som` is accepted by the CLI. The public README does not document the algorithm — do not invent flags or hyperparameters for it.

Four extra methods (`failspy`, `gabliteration`, `heretic`, `rdo`) appear only in Python imports. Do not use them from Hermes: AGPL + MIT boundary. CLI only.

## Weight projection (permanent)

README seven-preset table (escalating thoroughness):

| Method | Directions | Key features | Best for |
|---|---|---|---|
| `basic` | 1 (diff-in-means) | Fast baseline | Quick test, small models |
| `advanced` | 4 (SVD) | Norm-preserving, bias projection, 2 passes | **Default.** Clean removal, minimal capability loss |
| `aggressive` | 8 (SVD) | Whitened SVD, iterative refinement, 3 passes | Maximum guardrail removal |
| `surgical` | 8 (SVD) | EGA, head surgery, SAE, layer-adaptive, MoE-aware | Precision MoE models |
| `optimized` | 4 (SVD) | Bayesian auto-tuned, CoT-aware, KL co-optimized | Best quality with auto-tuning |
| `inverted` | 8 (SVD) | Semantic refusal inversion (2x reflection) | Refusal inversion experiments |
| `nuclear` | 8 (SVD) | All techniques + expert transplant + steering | Maximum force |

Also on the CLI, not in that seven-row table: `spectral_cascade`, `informed`, `som`.

- `informed` — analysis modules run *during* obliteration and auto-configure directions, layers, regularization, and Ouroboros passes. Experimental; slower than `advanced`.
- `spectral_cascade` — DCT frequency-domain decomposition. Research path.

## Direction extraction

| Flag | Description |
|---|---|
| `--direction-method diff_means` | Difference-in-means (default when unset on `self-improve`; `obliterate` default is method-dependent / None) |
| `--direction-method svd` | Multi-direction SVD |
| `--direction-method leace` | LEACE closed-form linear erasure |
| `--direction-method som` | CLI-accepted; undocumented in README |

## Selection flowchart

```
Quick test?                            → basic
Precision MoE?                         → surgical
Best quality with auto-tuning?         → optimized
Else                                   → advanced
Need maximum guardrail removal?        → aggressive
Need maximum force?                    → nuclear
advanced left >10% refusal on 3B+?     → aggressive, or self-improve --audit
```

Do not start with `informed` or `aggressive`.

## Key CLI overrides

| Flag | Notes |
|---|---|
| `--n-directions` | Override extracted direction count |
| `--regularization` | Fraction to preserve, `0.0`–`1.0` |
| `--refinement-passes` | Iterative re-probe |
| `--quantization` | `4bit` or `8bit` only |
| `--verify-sample-size` | Default 30 |
| `--large-model` | 120B+ conservative defaults |
| `--max-perplexity-increase` | Default `3.0` × stock (abort) |
| `--min-coherence-retention` | Default `0.5` |
| `--max-degenerate-fraction` | Default `0.2` |

## Troubleshooting

| Problem | Fix |
|---|---|
| Refusal rate > 20% | More `--n-directions`, `aggressive`, or `self-improve --audit` |
| Refusal rate 5–20% | `--refinement-passes 3`, `--direction-method svd`, `--residue-file` |
| Coherence/degeneracy trip | Fewer directions, higher `--regularization`, drop to `basic` |
| MoE still refuses | `surgical` (precision MoE), then `nuclear` (maximum force) |
| Reasoning / CoT degraded | `optimized` (CoT-aware) |
| OOM | `--quantization 4bit`, `--gpus` with more cards (not Qwen3.8), `--gpu-memory-utilization` |
| Qwen3.8 load rejected | `.[qwen-hybrid]`, one GPU, no `device_map=auto` |

## Steering (reversible)

Inference-time steering exists in the Python package (`SteeringVectorFactory`, `SteeringHookManager`). Hermes must not import it. If the user wants reversible changes, say so and keep the work in an AGPL-licensed project, or skip weight surgery.
