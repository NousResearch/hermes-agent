# BUILD BRIEF: rank-4 extraction — compaction fork logic → agent/fork_ext/compaction_ext.py

Worktree branch `feat/fork-ext-r4-compaction` (off fork/main). ONE extraction from
`docs/sync/2026-07-16-fork-mergeability-refactor-SPEC.md` (read FULLY; ritual IN ORDER).
REUSE `scripts/refactor_equiv/` (landed #372); extend mutate.py's registry.

## Target
Fork-only PURE logic in `agent/context_compressor.py` (9 hunks) + `agent/conversation_compression.py`
(3 hunks). Read both + fork history to identify the pure fork surface. Known fork-only candidates:
- P2 skew-calibration math (reset_skew_calibration, the skew compute/persist helpers) — the
  PURE math, not the persistence I/O wiring,
- per-model threshold resolution IF locally defined here (check _resolve_per_model_threshold
  lives in agent_init.py — do NOT move cross-file logic; scheduler rule applies),
- the compaction-announce message BUILDERS (string formatting from stats) — pure formatters.

⚠️ HIGH CAUTION file: compression is fleet-critical and heavily contract-tested
(tests/agent/test_context_compressor.py 164+ tests, TestPreflightDeferral, skew tests).
Extract ONLY functions that are already pure (no self-mutation beyond reading attrs). When a
candidate mutates compressor state, wrap the pure CORE (compute) and leave the mutation at the
call site. If the safely-extractable pure surface is under ~40 lines, STOP and write
docs/sync/review/r4-verdict.md (honest no-go) — this file's value is mostly contract tests,
and a forced extraction here is worse than none.

## Ritual (spec order)
1. Golden-capture untouched → tests/golden/compaction_ext/.
2. Pure move; 1-line call sites.
3. Golden-replay byte-identical.
4. ≥3 mutations; all RED; revert.
5. Run tests/agent/test_context_compressor.py + test_compaction_announce.py +
   test_compression_concurrent_fork.py + tests/gateway/test_session_hygiene.py (the
   hygiene-announce fork feature guards).
6. fork-features.json add/migrate ("hygiene compaction announces in-chat" entry may need a
   paths update if the announce builder moves; tests = collectable nodeids only;
   lint-manifest clean).
7. NET budget = 10 + 2×call_sites.
8. Import-order audit.

## Constraints
venv pytest green, py_compile, commit locally per-step, DO NOT push/PR, STOP (or no-go verdict).
