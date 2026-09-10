# Agent Memory Benchmark Suite

A system-agnostic benchmark framework for evaluating AI agent memory systems across 424 scenarios, 19 benchmark categories, and 15 suites. It measures retrieval quality, contradiction handling, scope isolation, adversarial robustness, scale handling, feedback-driven reranking, and multi-agent memory behaviors.

This framework currently includes:
- 7 adapter implementations under `benchmarks/backends/`
- 22 benchmark tests across 9 test files in `tests/benchmarks/`
- reporting, comparison, optimization, and validation tooling

This branch is the framework-first PR. It intentionally does not commit a result bundle.

## What This Benchmarks

Memory systems are evaluated on dimensions that matter for real agent use:

1. Can it find what it stored? — semantic recall vs keyword overlap
2. Does it know what is current? — contradictions, temporal integrity
3. Can it isolate context? — scopes and boundary handling
4. Does it resist manipulation? — adversarial prompt/instruction injection
5. Does it scale? — retrieval under 10–1000 fact loads
6. Does it learn from feedback? — reward-driven reranking
7. Does it preserve delegated and compressed context? — compression survival and delegation memory

## Quick Start

```bash
# Fast smoke test: one backend, one small suite, one seed
python -m benchmarks --backend baseline-flat --suite d --runs 1 --seeds 42

# Multi-seed local run for stronger evidence
python -m benchmarks --backend baseline-flat --suite all --runs 5 --seeds 42 43 44 45 46

# Generate a markdown report from a locally produced result file
python -m benchmarks --backend baseline-flat --suite d --runs 1 --seeds 42 --output-dir /tmp/bench-out
python -m benchmarks --report --result-file /tmp/bench-out/baseline-flat.json

# Compare two locally produced result files
python -m benchmarks --compare /tmp/bench-out/before.json /tmp/bench-out/after.json
```

## Suites

| Suite | Scenarios | Categories | What it Tests |
|-------|-----------|------------|---------------|
| A | 200 | semantic_recall, contradictions, cross_reference, importance_filtering, temporal_decay | Core retrieval under distractors, currentness, multi-hop recall |
| B | 30 | compression, consolidation | Consolidation and compression-oriented memory behavior |
| C | 20 | scopes | Scope isolation and boundary handling |
| D | 15 | adversarial | Prompt injection, malicious overwrite, exfiltration attempts |
| E | 12 | scale | Needle-in-haystack retrieval at increasing corpus sizes |
| F | 11 | integration | Store → recall → update lifecycle flows |
| G | 13 | qlearning | Reward-driven reranking and feedback response |
| H | 53 | deduplication, supersession, typed_decay, scope_lifecycle, notation_parsing | Structured-memory operations |
| I | 15 | conversation_memory, capacity_stress | Conversational memory and load degradation |
| J | 12 | topic_shift_recall | Topic interference and context pivots |
| K | 8 | compression_survival | Survival of critical facts after compression |
| L | 8 | delegation_memory | Parent-agent recall of delegated outcomes |
| M | 10 | format_sensitivity | Retrieval sensitivity to JSON/markdown/prose storage |
| N | 9 | retrieval_ablation | Keyword-vs-semantic retrieval signal contribution |
| O | 8 | timestamp_integrity | Preservation of temporal ordering metadata |

Total: 424 scenarios across 19 categories and 15 suites.

## Implemented Adapters

Adapter implementations currently present in-tree:
- `baseline-flat`
- `byterover`
- `hindsight`
- `holographic`
- `honcho`
- `mem0`
- `openviking`
- `retaindb`

The framework PR includes adapters and tooling only. Result snapshots can be generated locally or proposed in a follow-up PR.

## Methodology

### Judge Modes

Heuristic judge (default):
- 75% keyword threshold
- trigram overlap requirement
- deterministic and cheap
- biased toward verbatim/near-verbatim returns

LLM judge (opt-in):
- semantic equivalence checking
- slower and costs money
- better for publication-quality semantic comparisons

For serious cross-backend claims, prefer rerunning with `--judge llm` or reporting both heuristic and LLM outcomes.

### Capability-Aware Execution

Backends declare support through `BackendCapabilities`. Categories that require unsupported capabilities are skipped instead of being counted as failures. This prevents punishing a backend for lacking features outside its design scope.

### Shared-Suite Aggregation

Aggregate comparisons should use only categories that all compared backends actually ran. Full-suite means across mismatched category subsets can mislead.

### Seeds and Statistical Validity

The framework supports multi-seed runs, confidence intervals, and significance testing.

This framework-only branch does not ship committed result JSONs. When generating results locally, prefer 5-seed reruns where feasible before making strong benchmark claims.

## Interpreting Results

What scores mean:
- Per-category scores: fraction of scenarios passed in that category
- Overall score: mean across the categories a backend actually ran
- Shared-suite mean: mean across only the categories run by every backend in the comparison set

What scores do not mean:
- A single higher score does not prove a universally better memory system
- Scores from heuristic judge and LLM judge are not directly interchangeable
- Full-suite means across different category subsets can mislead; shared-suite means are fairer

## Known Limitations

1. Heuristic judge bias: lexical overlap is still favored over paraphrase.
2. External/service backends vary in reset semantics and environmental dependencies.
3. Result quality depends on backend environment quality (API keys, local services, embedding providers).
4. Statistical strength depends on how many seeds/runs you actually execute.

## Running the Benchmark Test Suite

```bash
pytest tests/benchmarks -q
```

Current benchmark test status in this cleaned branch: 22 tests passing.

## Adding a New Backend

1. Implement `BenchmarkableStore` in `benchmarks/backends/<name>_adapter.py`
2. Export:
   - `BACKEND_NAME`
   - `BACKEND_CLASS`
   - `BACKEND_CAPABILITIES`
3. Smoke test it:

```bash
python -m benchmarks --backend <name> --suite d --runs 1 --seeds 42
```

4. Add focused tests under `tests/benchmarks/`

## References

The suite draws from practical IR evaluation, adversarial robustness, and cognitive-memory-inspired testing ideas. See `benchmarks/METHODOLOGY_REVIEW.md` for the audit trail and current methodological caveats.
