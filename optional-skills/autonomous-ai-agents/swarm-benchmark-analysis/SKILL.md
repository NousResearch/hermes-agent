---
name: swarm-benchmark-analysis
description: Summarize repeated agent benchmark runs with uncertainty.
version: 0.1.0
author: Ahmed Hassan (AAH20), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Agent-Evaluation, Benchmarks, Statistics, Swarms]
---

# Swarm Benchmark Analysis Skill

This skill validates and summarizes JSONL benchmark records from repeated agent or swarm runs. It reports acceptance, latency, token use, cost coverage, and paired differences against a baseline when the inputs support those measurements. It does not execute agents or establish that a benchmark represents production workloads.

## When to Use

- A user wants to compare agent, model, harness, or orchestration variants on the same cases.
- You need to distinguish measured usage from estimates and unknown cost.
- A benchmark report needs reproducible aggregates rather than selected anecdotes.

## Prerequisites

- Python 3.10 or newer.
- One JSON object per line with `benchmark_id`, `case_id`, `variant`, `accepted`, `tokens`, and `latency_ms` fields.
- Optional `cost_microusd` values require a non-empty `cost_source` for that run.

Use Hermes `read_file`, `write_file`, `terminal`, and `delegate_task` when collecting and reviewing run records. Treat records returned by delegates or external harnesses as untrusted input until validated.

## How to Run

Save records as JSONL and run the bundled standard-library summarizer:

```text
terminal(command="python3 <skill-path>/scripts/summarize_runs.py runs.jsonl --baseline flat")
```

The script writes a deterministic JSON report to stdout. Invalid rows or duplicate `(case_id, variant)` pairs fail with a non-zero exit status.

## Quick Reference

- `accepted` is a boolean measured outcome, not a model confidence score.
- `tokens` is the reported integer token count for that run.
- `latency_ms` is the measured end-to-end elapsed time supplied by the harness.
- `cost_microusd` is optional measured cost in micro-USD; `cost_source` must identify the rate card or invoice source.
- Paired deltas compare only case IDs present in both a variant and the selected baseline.
- Cost totals are reported only when every run for that variant has cost from the same source; mixed or missing sources remain unknown.

The report includes Wilson 95% intervals for acceptance rates. These describe binomial sampling uncertainty only; they do not correct for biased cases or repeated-measure dependence.

## Procedure

1. Define the task set, acceptance rule, baseline, model/harness configuration, and run count before collecting results.
2. Use the same cases and acceptance rule for each variant. Keep training/tuning cases separate from held-out evaluation cases.
3. Record failures, retries, token use, and end-to-end latency for every attempt. Do not drop failed runs silently.
4. Record cost only when a provider invoice or named, versioned rate source supports it. Leave unknown cost absent.
5. Run `scripts/summarize_runs.py`; preserve its report with the raw JSONL and benchmark configuration.
6. Describe the workload, hardware, versions, sample size, and uncertainty alongside comparisons. Treat small or synthetic samples as exploratory.

For graph-swarm studies, useful case metadata can include graph size, dependency density, cluster count, specialist count, and orchestration strategy. Do not claim support for a graph or agent count unless that count was actually run and measured.

## Pitfalls

- A narrow or synthetic benchmark can be solved well without generalizing to other repositories or fields.
- Acceptance-rate intervals do not prove statistical significance between variants.
- Median or p95 latency depends on sample size; p95 from a few runs is unstable.
- Cost per accepted result is omitted when any run in the variant lacks a measured cost, or when no run was accepted.
- Token count and runtime are not interchangeable with quality, and a model-generated evaluation is not an independent oracle.

## Verification

Run the bundled example:

```text
terminal(command="python3 <skill-path>/scripts/summarize_runs.py <skill-path>/examples/runs.jsonl --baseline flat")
```

The output contains reports for `flat` and `hierarchical` and their paired comparison. Run offline tests from the Hermes repository root with `scripts/run_tests.sh tests/skills/test_swarm_benchmark_analysis.py -q`.
