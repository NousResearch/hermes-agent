# BUILD BRIEF: rank-3 extraction — cron fork logic → cron/fork_ext/scheduler_ext.py

Worktree branch `feat/fork-ext-r3-cron` (off fork/main). ONE extraction from
`docs/sync/2026-07-16-fork-mergeability-refactor-SPEC.md` (read FULLY; follow the
"Per-extraction proof ritual" IN ORDER). REUSE `scripts/refactor_equiv/` (landed #372);
extend `mutate.py`'s registry (pattern: `relay_header_mutations`).

## Target
Fork-only logic in `cron/scheduler.py` (12 hunks on the 07-15 sync). Read the file and its
fork history (`git log --oneline fork/main -- cron/scheduler.py | head -20`) to identify the
fork-only PURE surface. Known fork-only candidates:
- `_get_script_timeout()` and the 7200s script-timeout resolution chain,
- the per-job `reasoning_effort` resolution helper (the fork's per-job override that feeds
  parse_reasoning_effort/resolve_reasoning_config),
- the cron ContextVar session-marking helpers IF they are pure and locally defined here
  (set_cron_session lives in gateway/session_context.py — do NOT move that; only extract
  scheduler-side pure helpers).
Extract only PURE functions with 1-line call sites. Do NOT touch the run_job control flow,
the run-claim heartbeat (upstream's, from the 07-15 merge), or anything async. If the pure
surface is under ~30 lines, STOP and write docs/sync/review/r3-verdict.md (honest no-go).

## Ritual (spec order)
1. Golden-capture untouched tree → tests/golden/scheduler_ext/.
2. Pure move; 1-line call sites.
3. Golden-replay byte-identical.
4. ≥3 mutations registered in mutate.py; all RED; revert.
5. Run tests/cron/ (test_scheduler.py is a 300s-cap giant — run the targeted files:
   test_per_job_reasoning_effort.py, test_cron_script.py, test_cronjob_schema.py + any file
   greping the moved symbols).
6. fork-features.json add/migrate (tests = collectable nodeids only; lint-manifest clean).
7. NET budget = 10 + 2×call_sites.
8. Import-order audit.

## Constraints
Same as always: venv pytest green, py_compile, commit locally per-step, DO NOT push/PR, STOP.
