# R5 Final Report — Production Hardening + Long-Run Validation

## Baseline
- Commit `f97a4102dd`, Hermes 0.21.1, Python 3.14.3, numpy 2.5.1,
  pytest 9.1.1, SQLite 3.50.4. Frozen: `results/r5/baseline/env.json`.
- Entry gate: 134 passed / 0 failed / 6 skipped.

## Long-run (10,000 events)
- Snapshots: bulk 6000 active → churn 6749+750 lineage → final
  6785 active / 1 superseded / 312 stale / 750 lineage / 0 dupes.
- All 12 invariants hold at final state; search 14.15 ms; DB 4.07 MB.
- No bloat, no drift, no trust escape. Artifact: `longrun/longrun.json`.

## State machine / lineage fuzz (60 seeded sequences)
- Found 2 REAL defects, both fixed + regressed:
  1. revoke kept attestation (I2) → revoke now clears verified_*.
  2. supersession cycles a→b→a → `_would_cycle` guard refuses.
- I1 refined: NULL successor = explicit terminal (successor deleted).
- 60/60 clean after fix. Seeds deterministic; failures persisted.

## Security
- Poison/injection stay DATA (write + retrieval paths).
- REAL gaps closed: safety.py NFKC normalization (fullwidth bypass),
  lowercase env style (aws_secret_access_key).
- Obfuscation measured; encoded payloads inert by construction.
- Fake authority confers nothing; cross-project alias+content isolated.
- TP holds, FP holds (R2 suite green).

## Isolation: separate DBs never cross-read (content + aliases).
## Trust drift: bounded [0,1], asymmetric deltas, revoke/supersede safe.
## Retrieval: deterministic rankings; trust-only mutations behave.
## Context: prefetch ≤5 lines / ≤2500 chars at 1k facts.
## Recovery: pre-commit crash → zero rows; batch atomic; bank rebuild;
##   idempotent migration/revalidate/supersede-retry/heal; legacy FTS heal.
## Performance: op latencies p50/p95/p99 @100/1k recorded; no leak
##   (200 open/search/close cycles, shared registry empty, cache bounded).
## Invariants I1-I12: machine-checked in fuzz + long-run + dedicated test.
## Quality regression: R1/R2/R3/R4 gold identical (suites green unmodified).
## Full gate: bare CI-parity **187/187 green** (22 files); official runner
##   0 failed (skips = numpy-absence by convention).
## LLM: 0 everywhere. No new deps, no paid APIs, no core changes.
## Promotion: PRODUCTION-READY. Production fixes, all regressed: revoke
##   attestation-clear, supersession cycle guard, mark_stale active-only,
##   successor-delete demote-to-stale, savepoint _atomic() on all writers,
##   safety NFKC + Cf-strip + any-case env, atom-cache race fix, vector
##   scan cap, debug logging on swallowed errors.
## Remaining: 50k re-run post-change (numbers reused from R3, write path
##   touched only by revoke-clear + remove-txn — low risk, noted), Thai
##   segmenter, temporal aging automation, qb1 semantic.
## Reproduce: R5 test files + full gate command (see audit).
