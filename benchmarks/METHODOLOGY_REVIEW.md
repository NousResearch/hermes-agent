# Benchmark Suite Methodology Review

Date: 2026-04-06
Reviewer: Hermes agent automated audit
Status: framework audit complete; several methodology fixes landed, but the checked-in result bundle is still single-seed and should be treated as development evidence rather than final statistical evidence

## Executive Summary

This benchmark framework is real, substantial, and worth shipping. The core architecture is clean: adapters, capability declarations, suite fixtures, runner, judge, metrics, and reporting are all present.

The audit found five important methodological issues in the original benchmark work:
1. heuristic judge bias toward lexical overlap
2. weak/ceiling-effect suites in early versions of some fixtures
3. aggregate comparisons across mismatched category subsets
4. single-seed result artifacts
5. misleading or overstated academic framing in some suite descriptions

The cleaned benchmark branch addresses much of the framework-side honesty problem:
- capability-aware skipping exists
- shared-suite comparison reporting exists
- suite counts and fixture corpus are concrete
- methodology caveats are now documented more honestly

The main unresolved limitation in the checked-in artifact set is still item 4: the stored result JSON files are single-seed snapshots.

## What Improved

### 1. Capability-aware execution
Backends are no longer forced through obviously unsupported categories. Unsupported categories can be skipped instead of silently turning into fake failures.

### 2. Fixture hardening
The suite corpus now includes the expanded benchmark set through suites A–O, totaling 424 scenarios. The fixture layout is coherent and testable.

### 3. Shared-suite reporting
A fairer comparison mode exists for categories all compared backends actually ran.

### 4. Audit trail and benchmark tests
The branch now carries benchmark-focused tests and documentation that make the system easier to verify and maintain.

## Remaining Caveats

### 1. Heuristic judge bias remains
The heuristic judge is still useful for speed and reproducibility, but lexical overlap can still favor verbatim systems over semantically equivalent paraphrases.

Implication:
- use heuristic mode for development
- use LLM judging, or at least dual reporting, for stronger semantic claims

### 2. Stored results are single-seed
Every checked-in result JSON currently reports `num_runs = 1`.

Implication:
- no meaningful variance estimate from the checked-in bundle
- no strong significance claims yet
- checked-in results are best read as engineering snapshots, not final science

### 3. Aggregate interpretation still needs care
Full-suite means can still be misleading because they summarize different category subsets. Shared-suite means are the safer comparison surface.

### 4. External backend quality is environment-dependent
Cloud/service backends depend on API keys, embedding providers, and service configuration. Stored numbers may shift when the external environment shifts.

## Recommended Release Posture

This benchmark suite is strong enough to merge as benchmark infrastructure and a checked-in development result bundle, provided the PR is honest about what is and is not claimed.

Good claims:
- Hermes now has a substantial capability-aware memory benchmark framework
- the framework includes 15 suites, 19 categories, and 424 scenarios
- benchmark tests pass in the cleaned branch
- checked-in results provide an initial comparative snapshot across 6 backends

Claims to avoid until rerun:
- statistically significant superiority claims
- strong statements based on variance or confidence intervals from the checked-in bundle
- overconfident semantic claims from heuristic-only judging

## Recommended Next Step After Merge

Run a publication-strength refresh of the result bundle:
- 5 seeds for local backends
- as many repeated runs as practical for service backends
- LLM-judge confirmation on the most semantically sensitive suites
- regenerate comparison report from the refreshed result set

## Bottom Line

The framework itself is ready to be reviewed as real product-quality benchmark infrastructure.

The checked-in result artifacts are honest enough to ship only if they are described as a development snapshot, not as the final word.
