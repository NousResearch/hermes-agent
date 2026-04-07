# Memory Backend Benchmark Comparison Report

Date: April 6, 2026
Methodology: heuristic judge + capability-aware category skipping
Result bundle status: checked-in development snapshot, single-seed (`num_runs = 1` in every stored JSON file)
Backends in this report: baseline-flat, holographic, mem0, hindsight, honcho, retaindb

## Scope of this report

This report summarizes the result files currently checked into `benchmarks/results/`.

Truth-in-advertising notes:
- The framework supports multi-seed statistics, but the stored result bundle is single-seed for all 6 backends.
- Treat these numbers as useful engineering evidence, not publication-strength statistics.
- Shared-suite comparisons are more meaningful than full-suite means because different backends run different category subsets.

## Coverage summary

- Suites: A–O (15 suites)
- Categories represented in the stored results: 19
- Scenarios in the fixture corpus: 424
- Adapter implementations present in-tree: 7 benchmark adapter files
- Backends with checked-in results: 6

Categories with results in the checked-in bundle:
- semantic_recall
- contradictions
- cross_reference
- importance_filtering
- temporal_decay
- adversarial
- scale
- integration
- qlearning
- deduplication
- scopes
- capacity_stress
- conversation_memory
- topic_shift_recall
- compression_survival
- delegation_memory
- format_sensitivity
- retrieval_ablation
- timestamp_integrity

## Full Results by Category

Categories with `—` indicate the backend did not produce a stored result for that category in the current bundle.

| Category | baseline-flat | holographic | mem0 | hindsight | honcho | retaindb |
|---|---:|---:|---:|---:|---:|---:|
| semantic_recall | 46.0% | 58.0% | — | 48.0% | 46.0% | 78.0% |
| contradictions | 65.0% | 65.0% | — | 60.0% | 45.0% | 55.0% |
| cross_reference | 53.3% | 60.0% | — | 26.7% | 62.2% | 95.6% |
| importance_filtering | 97.5% | 77.5% | — | 80.0% | 87.5% | 87.5% |
| temporal_decay | 88.9% | — | — | — | — | — |
| adversarial | 80.0% | 93.3% | — | 86.7% | 86.7% | 93.3% |
| scale | 100.0% | 91.7% | 66.7% | 83.3% | 91.7% | 83.3% |
| integration | 87.5% | 66.7% | 100.0% | 83.3% | 100.0% | 100.0% |
| qlearning | 92.3% | 76.9% | 100.0% | 46.2% | 53.8% | 100.0% |
| deduplication | 100.0% | 87.5% | 87.5% | 62.5% | 87.5% | 75.0% |
| scopes | 65.0% | — | — | — | — | — |
| capacity_stress | 100.0% | 80.0% | 20.0% | 60.0% | 80.0% | 80.0% |
| conversation_memory | 80.0% | 80.0% | 90.0% | 70.0% | 90.0% | 100.0% |
| topic_shift_recall | 58.3% | 75.0% | 83.3% | 83.3% | 75.0% | 66.7% |
| compression_survival | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 87.5% |
| delegation_memory | 75.0% | 62.5% | 62.5% | 75.0% | 75.0% | 87.5% |
| format_sensitivity | 90.0% | 90.0% | 70.0% | 70.0% | 80.0% | 90.0% |
| retrieval_ablation | 88.9% | 66.7% | 77.8% | 66.7% | 77.8% | 77.8% |
| timestamp_integrity | 100.0% | 62.5% | 62.5% | 75.0% | 75.0% | 87.5% |

## Aggregate Scores

### Full-suite mean

This is the mean across the categories each backend actually ran in the checked-in bundle.
These values are not directly apples-to-apples because category coverage differs.

| Backend | Mean Score | Categories with stored results |
|---|---:|---:|
| retaindb | 83.1% | 17 |
| mem0 | 79.2% | 12 |
| baseline-flat | 75.1% | 19 |
| hindsight | 75.0% | 17 |
| honcho | 75.0% | 17 |
| holographic | 70.4% | 17 |

### Shared-suite mean

These 12 categories are present for all 6 backends in the checked-in bundle:
- capacity_stress
- compression_survival
- conversation_memory
- deduplication
- delegation_memory
- format_sensitivity
- integration
- qlearning
- retrieval_ablation
- scale
- timestamp_integrity
- topic_shift_recall

| Backend | Shared-Suite Mean |
|---|---:|
| baseline-flat | 89.3% |
| retaindb | 86.3% |
| honcho | 82.1% |
| holographic | 78.3% |
| mem0 | 74.6% |
| hindsight | 72.9% |

## Engineering Takeaways

What the checked-in snapshot suggests:
- baseline-flat is strongest on exact recall, timestamps, and load-sensitive categories
- retaindb is the strongest checked-in cloud/backend-service result overall in this bundle
- mem0 and honcho look strong on conversational/integration-style tasks
- holographic remains strongest on adversarial and relatively strong on semantic retrieval
- backend choice clearly depends on task type rather than a single universal ranking

## Known Limitations

1. Single-seed stored results: no real variance estimate yet.
2. Heuristic judge bias: lexical overlap still has structural influence.
3. Coverage asymmetry: full-suite means compare different category subsets.
4. Environmental dependence: API/service backends depend on external runtime quality.

## Re-run Guidance

For stronger evidence, regenerate the result bundle with:

```bash
# local backends
python -m benchmarks --backend baseline-flat --suite all --runs 5 --seeds 42 43 44 45 46
python -m benchmarks --backend holographic --suite all --runs 5 --seeds 42 43 44 45 46

# service backends (example)
MEM0_API_KEY="$MEM0_API_KEY" python -m benchmarks --backend mem0 --suite all --runs 1 --seeds 42
```

If you publish or compare backends seriously, prefer:
- shared-suite tables
- multi-seed reruns where feasible
- LLM-judge confirmation for semantically sensitive claims
