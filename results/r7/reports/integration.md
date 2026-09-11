# R7 Integration — full-stack gate evidence

## Gate (official runner, all green except documented exclusion)

- 24 files, **181 passed, 0 failed, 10 skipped** (skips = numpy-absence on
  runner Python 3.11, repo convention).
- Files: stock ×5 + R1 ×2 + R2 ×1 + R3 ×4 + R4 ×4 + R5 ×5 + R6 ×1 + R7 ×1.
- Excluded with reason: `test_holographic_cognitive.py` (16 of its 50 tests
  target the superseded pre-R1 API — AttributeError-only, verified; the other
  34 pass; file kept byte-identical — see `recovery_matrix.md`).

## New R7 tests (`test_r7_integration.py`, 3/3)

- end-to-end lifecycle (18-step scenario, R7-11 adapted)
- backup/restore parity (counts, trust, lifecycle, lineage, alias, retrieval)
- cross-layer invariants I2/I3/I7/I12/I15

## Overhead: integrated vs R6 stock (same env py3.14+numpy, @1k facts)

| op | stock (R6) | integrated (R7) | delta |
|---|---|---|---|
| health | 5.74ms | 6.15ms | +7% |
| maintenance LIGHT | 24.31ms | 25.78ms | +6% |
| DB size | 1,576,960 B | 1,609,728 B | +2% (lifecycle columns + lineage) |
| add/fact | — | 5.17ms | absolute only |
| search (top-5) | — | 2.86ms | absolute only |
| verify ×50 | — | ~181ms (~3.6ms each) | absolute only |

Artifact: `results/r7/performance/integrated.json`. LLM calls: 0 everywhere.

## Invariant coverage map (R7-15)

- I1 repo-truth: R2/R4 retrieval precedence suites
- I2 revoked-terminal: test_r4_temporal + test_r7_integration
- I3 no-cycle: test_r4_temporal + test_r5_fuzz + test_r7_integration
- I4 no-orphan lineage: test_r5_longrun + test_r4_temporal
- I5 quarantine firewall: test_r2_hardening + test_r5_security
- I6 memory-as-data ops: test_r6_ops + test_r5_security
- I7 trust bounded: test_r5_fuzz + test_r7_integration
- I8 ops cannot bypass security: test_r6_ops (redaction) + test_r7_integration
- I9 maintenance idempotent: test_r6_ops + test_r5_recovery
- I10 backup/restore fidelity: test_r6_ops + test_r7_integration
- I11 project isolation: test_r6_ops + test_r5_security
- I12 zero-LLM: every suite asserts `llm_calls == 0`
- I13 offline: test_r6_ops (socket-blocked cycle)
- I14 human-only destructive: no delete/reset API in operations.py (reviewed)
- I15 additive/idempotent migration: test_r6_ops + test_r7_integration
