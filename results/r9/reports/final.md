# R9 FINAL — STATUS: RELEASE CANDIDATE (re-review; commit decision open)

## STATUS: RELEASE CANDIDATE
- First review: FAIL-as-BASELINE (stale manifest, CRLF checksums, missing
  file list/log, uncommitted tree). All mechanically fixable items repaired:
  manifest regenerated post-fix (store.py 45301 B), LF `checksums.sha256`
  (`sha256sum -c` POSIX-OK), exact 25-file list in
  `reproducibility/commands.json`, full log preserved at
  `reports/test_closure.log` (25 files, 186/0/11), `enable_llm` hook named
  in release notes.
- Remaining: worktree uncommitted (no commit requested — decision left to
  user). Freeze defined by `manifest.json` + base `45a6101f`.

## COMMIT: `45a6101f` (tree: R1–R8 work uncommitted by design)
## MANIFEST: `results/r9/release/manifest.json` (15 files, sha256 each)
## ENVIRONMENT: `results/r9/release/environment.json`
  (Hermes 0.21.1, runner py3.11 / shell py3.14.3, numpy 2.5.1, pytest 9.1.1, SQLite 3.50.4)

## CURRENT TESTS
- Test closure (R9-14): **25 files, 186 passed, 0 failed, 11 skipped**
  (official runner; 10 numpy-absence + 1 HRR-gated backfill test).
- Historical suite segregated under `results/r8/legacy/` (never current failures).

## HISTORICAL TESTS
- 16 pre-R1 API failures documented (R7/R8); file preserved byte-identical.

## SECURITY
- Release audit: 52 passed, 0 failed (R2/R4/R5/R1-sem suites); critical leakage 0.
- Artifact: `results/r9/security/audit.json`.

## PERFORMANCE (frozen baseline @1k, py3.14+numpy)
- search p50 2.71 / p95 3.67 / p99 4.02ms; health 6.01/6.86/7.11ms;
  feedback 3.8/4.1/5.0ms; DB 1,581,056 B.
- Release ops: startup 94.12ms, add 5.48ms/fact, prefetch 5.19ms,
  update 4.56ms, maintenance 27.61ms, backup 26.32ms, verify 15.86ms,
  restore 18.18ms. Artifacts: `performance/percentiles.json`, R8 baseline.

## BACKUP: verified (checksum + schema + integrity), scratch-restore parity.
## RESTORE: scratch-first only; no prod-overwrite API exists.
## ROLLBACK: `rollback/demo.json` — failed upgrade in scratch recovered via
  known-good backup; prod untouched. Policy: rollback = restore backup.
## OFFLINE: socket-blocked full cycle (R6) + clean-env run; `network_dependencies() == []`.
## LLM: 0 on every path (startup/search/prefetch/maintenance/backup/restore/migration).

## REPRODUCIBILITY
- Canonical: `bash scripts/run_tests.sh <files>` (CI-parity env built in).
- Clean-env proof: fresh HERMES_HOME + foreign cwd + no venv → lifecycle green
  (`clean_env/result.json`). Note: `env -i` bare-metal run fails only because
  the PyManager shim needs its env (harness artifact, not a product dependency).

## RELEASE DECISION: hold at RELEASE CANDIDATE — freeze the architecture
  (no new features without versioned proposals + benchmark evidence).
  Promotion to PRODUCTION BASELINE requires a committed tree (user decision
  pending).

## PRODUCTION CHANGE IN R9 (one, release-blocking, minimal)
- `store._rebuild_bank` backfills NULL HRR vectors for legacy rows (derived
  state only, content untouched, idempotent). Cause: upgraded legacy DBs could
  never rebuild banks → perpetual NEEDS_REPAIR + phantom repair report.
- Evidence: ad-hoc 6/6 + `test_r6_legacy_vector_backfill` (HRR-gated) +
  13-file regression 114/0. No other production edits in R9.

## LIMITATIONS
- Runner py3.11 lacks numpy (10 skips by convention).
- 50k integrated long-run not re-run (standing note).
- No local backup encryption (documented warning).
- Uncommitted worktree (no commit requested).

## EVIDENCE
- `results/r9/{baseline,clean_env,fresh_db,legacy,backup,rollback,security,performance,reproducibility,release}/`
- `results/r9/reports/final.md` (this file)
