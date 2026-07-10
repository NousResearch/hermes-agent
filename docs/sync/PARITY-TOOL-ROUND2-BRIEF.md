# hermes-parity — ROUND 2 GAP BRIEF (2026-07-10)

Read `docs/sync/2026-07-10-hermes-parity-SPEC.md` (SPEC v2.1) FULLY — it is now present in
this worktree (round 1 built without it; this round closes the gaps). Existing code in
`scripts/hermes_parity/` is a good skeleton — EXTEND it, do not rewrite. Keep stdlib-only.
Do NOT commit; the orchestrator commits.

## Gaps found in review of round-1 code vs SPEC v2.1

1. **`status`**: missing `--fail-behind N` (exit 2 when behind ≥ N — cron alert hook).
   Also `--predict` exists but spec CUT it for v1 — remove the flag (keep the helper
   function if tests use it, but drop the CLI surface).

2. **`bisect`** — biggest gap. Spec requires:
   - Cached baseline worktree auto-managed at `~/.hermes/worktrees/parity-baseline`
     pinned to `fork_main_at_start` from `.parity-state.json` — operator should NOT have
     to pass `--baseline/--merge` by hand when run inside a parity worktree (keep the
     explicit flags as overrides).
   - Four spec classifications: REGRESSION (pass-baseline/fail-merge), INHERITED/FLAKY
     (fail-both), UPSTREAM-TEST (test absent on baseline — detect via pytest collect-only
     exit 4 / "no tests ran"), ORDER-POLLUTION (pass-both). Rename/extend the current enum.
   - Multiple tests with **bounded parallelism**: default 2 workers, `--jobs N` capped at
     `os.cpu_count()//2` (ThreadPoolExecutor is fine; runner is subprocess-bound).
   - REGRESSION-classified tests re-run ONCE before final report (flake detection; a pass
     on re-run demotes to FLAKY).
   - `--from-file -` reads pytest node ids from stdin (one per line); `--from-file PATH` too.
   - Prints a classification TABLE at the end.

3. **`gates`**:
   - `--resume`: skip stages already green in `.parity-state.json` at the CURRENT tree-SHA;
     start at first invalid/missing. Persist per-stage results (ok, at, tree_sha) into state.
   - `--strict` for stage 3 (traps): findings become a hard fail instead of warnings.
   - CI reminder checklist must list the four named CI-owned gates (gitleaks, contributor-
     check, config-migration dry-run, tsc on apps/desktop incl. the zsh-zle/exit-194 trap)
     and read the pinned gitleaks version from `.github/workflows/*.y*ml` for display —
     if the parse fails print an explicit "⚠️ could not determine pinned gitleaks version"
     line, NEVER a silent skip.
   - Every stage failure prints its exact repro command.

4. **`finish`**: must actually CREATE the merge commit (verify HEAD is an in-progress
   merge / MERGE_HEAD exists OR the merge was staged --no-commit; commit with the template
   body: target SHA, merge-base, bucket stats, an empty "## Behavior changes" section for
   the operator), then push the branch (`git push origin <branch>`; remote name from state
   or `--remote`), then print the `gh pr create` command with `--body-file` pointing at a
   GENERATED body file (write it to the worktree), not inline --body. Refuse if fork/main
   moved since `start` (compare recorded `fork_main_at_start` to live; print recovery).

5. **`start`**: verify it writes the full spec state schema: created, target_sha,
   merge_base, fork_main_at_start, branch, rollback SHAs, buckets. Emit the
   conflict-bucketing report to `docs/sync/review/conflict-buckets.md` in the worktree
   (per-file hunk count + bucket MECHANICAL/SEMANTIC/ARCH-SPLIT + totals). No /tmp mirror
   of rollback SHAs (state file is the sole home).

6. **Startup**: git ≥ 2.38 version check (clear error if older).

7. **Tests**: extend `tests/scripts/test_hermes_parity.py` to cover every new pure
   function (classification incl. absent-on-baseline, fail-behind exit code, resume
   stage-skip logic, gitleaks version parse incl. failure path, from-file parsing,
   bounded-jobs computation). Fixture repos in tmpdir as before. All tests must pass via
   `scripts/run_tests.sh tests/scripts/test_hermes_parity.py`.

## Definition of done
- `python3.11 -m hermes_parity <every subcommand> --help` works from repo root
- unit tests green
- `python3.11 -m py_compile scripts/hermes_parity/*.py hermes_parity.py`
- update `docs/sync/README-hermes-parity.md` to match the final surface
- do NOT commit
