# Benchmark Suite Methodology Review

Date: 2026-04-06
Reviewer: Hermes agent automated audit
Status: framework audit complete; this branch is a framework-first benchmark PR, with methodology notes focused on how future result bundles should be interpreted

## Executive Summary

This benchmark framework is real, substantial, and worth shipping. The core architecture is clean: adapters, capability declarations, suite fixtures, runner, judge, metrics, and reporting are all present.

The audit found five important methodological issues in the original benchmark work:
1. heuristic judge bias toward lexical overlap
2. weak/ceiling-effect suites in early versions of some fixtures
3. aggregate comparisons across mismatched category subsets
4. single-seed result artifacts
5. misleading or overstated academic framing in some suite descriptions

This framework branch addresses much of the framework-side honesty problem:
- capability-aware skipping exists
- shared-suite comparison reporting exists
- suite counts and fixture corpus are concrete
- methodology caveats are documented clearly
- benchmark tests provide a real verification surface

## What Improved

### 1. Capability-aware execution
Backends are no longer forced through obviously unsupported categories. Unsupported categories can be skipped instead of silently turning into fake failures.

### 2. Fixture hardening
The suite corpus now includes the expanded benchmark set through suites A–O, totaling 424 scenarios. The fixture layout is coherent and testable.

### 3. Shared-suite reporting
A fairer comparison mode exists for categories all compared backends actually ran.

### 4. Audit trail and benchmark tests
The branch carries benchmark-focused tests and documentation that make the system easier to verify and maintain.

## Remaining Caveats

### 1. Heuristic judge bias remains
The heuristic judge is still useful for speed and reproducibility, but lexical overlap can still favor verbatim systems over semantically equivalent paraphrases.

Implication:
- use heuristic mode for development
- use LLM judging, or at least dual reporting, for stronger semantic claims

### 2. Aggregate interpretation still needs care
Full-suite means can still be misleading because they summarize different category subsets. Shared-suite means are the safer comparison surface.

### 3. External backend quality is environment-dependent
Cloud/service backends depend on API keys, embedding providers, and service configuration. Results may shift when the external environment shifts.

### 4. Statistical strength depends on actual rerun discipline
The framework supports multi-seed runs, confidence intervals, and significance testing, but those only matter if result bundles are generated with enough repeated runs.

## Recommended Release Posture

This benchmark suite is strong enough to merge as benchmark infrastructure.

Good claims:
- Hermes now has a substantial capability-aware memory benchmark framework
- the framework includes 15 suites, 19 categories, and 424 scenarios
- benchmark tests pass in the cleaned branch
- the framework can produce local result bundles and compare them honestly

Claims to avoid without reruns:
- statistically significant superiority claims
- strong statements based on variance or confidence intervals from a tiny result set
- overconfident semantic claims from heuristic-only judging

## Recommended Next Step After Merge

Generate one or more follow-up result bundles with explicit methodology notes:
- 5 seeds for local backends where feasible
- as many repeated runs as practical for service backends
- LLM-judge confirmation on the most semantically sensitive suites
- comparison reports derived from those refreshed result sets

## Bottom Line

The framework itself is ready to be reviewed as real product-quality benchmark infrastructure.

Result bundles should be treated as follow-up artifacts whose strength depends on how rigorously they are rerun.
