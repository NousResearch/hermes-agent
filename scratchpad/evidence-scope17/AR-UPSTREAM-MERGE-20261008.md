# Scope17 upstream merge review and validation

Merged code HEAD: `4731c0373588aa7fdf620452ce9f1968afb271aa`.
Merge parents: repair/evidence `693834b0c8` and upstream `08165d58931841cee713468ae89032af7c57060a`.
Normal two-parent merge; no conflicts, rebase, push, installed-checkout edit, Manager edit, provider call, or runtime action.

Independent read-only AR: `/root/sdk_upstream_sync_slice/final_merge_ar`, exact merged code HEAD, all 13 upstream paths plus combined auxiliary repair and repair tests: **NO FINDINGS**.
The reviewer traced upstream desktop scroll/composer changes, CLI context-cache guards, and TUI model-switch/compression behavior with the unchanged repair interfaces. Its preliminary direct test run (52 passes) is supplemental only, excluded from canonical validation.

Canonical validation uses `scripts/run_tests.sh`, the requested installed test interpreter via HERMES_PYTHON, and `--file-retries 0`.

- Initial ten-file batch: 78 passes, 86 failures. Eighty-five failures concern the two existing E2E suites described below. One cancellation test timed out waiting five seconds for its silent transport during concurrent checks; separate canonical isolated rerun: 13 passes, zero failures.
- Separate focused receipt excluding the two guard-blocked E2E suites: eight files, 73 passes, zero failures; one worker and zero retries.
- Related upstream TUI suite: eight files, 43 passes, zero failures.
- Full scripts/check on merged code HEAD: 11 checks, all passed; code health zero blocking/advisory.
- Git diff whitespace check passed.

## Parent-equivalent HomeIOGuard validation limit

The actual_auxiliary_routing suite reports 6 passes and 81 failures; explicit_base_anthropic reports four failures. The failure occurs before transport behavior in bootstrap import-time update recovery, which resolves a worktree Git journal in the installed repository's shared Git metadata. Canonical HERMES_HOME isolation is already active; changing that home does not move worktree Git metadata. No guard bypass was applied.

First failing chain: auxiliary client HTTP construction -> agent.process_bootstrap import -> hermes_bootstrap._settle_interrupted_update -> hermes_cli._early_recovery.restore_interrupted_pull -> journal.is_file -> HomeIOGuard.refuse (tests/home_io_guard.py:140).

Exact blocked path:
`/mnt/HC_Volume_105578801/hermes/hermes-agent-home/.hermes/hermes-agent/.git/worktrees/hermes-sdk-scope17-upstream-20261008/hermes-update-pull`

Exact error prefix: `AssertionError: TEST BUG: file I/O against the REAL hermes home`.

Parent proof: detached premerge HEAD `693834b0c8`, same canonical interpreter/runner, zero retries, one worker; selected actual_key_reload_keeps_yaml_endpoint and explicit_base_unknown_host_keeps_anthropic_path both fail at the same HomeIOGuard path. Returned to merged branch immediately afterward. Source, test files, bootstrap/recovery code, conftest, and guard are unchanged by this upstream merge. This demonstrates an existing worktree-isolation limit, not a newly introduced merge failure, and does not establish E2E behavior beyond the guard.

## Receipts

Log basenames are retained externally with SHA-256 hashes:

- scope17-upstream-merge-tests.log: `843a0430062ba7bcbb67b47d321e8160869e782d06bac73d5e46be8266cd256d`
- scope17-upstream-merge-check.log: `791263a95ac41c9876c6eb4d8f33b44494fd5a04afb37da464d661de47eed37e`
- scope17-upstream-related-tests.log: `d7266d5e2b61fe1bac4c681add618c00331e561b69559b1f5867fcda5b18708c`
- scope17-upstream-parent-limit.log: `5a8412d446fc0c4e961203cbe4d5a252b23294d5372aba17d4a898e9a6b1d7a4`
- scope17-upstream-focused-clean.log: `498f513c7e375d70e94b220d181ae3ef83ff9e8e24d4ea53df848fea9fb4d964`
- scope17-upstream-cancel-rerun.log: `945e11d08dc634e58688093bd51595d5e2c3e989df0b24488feb4a3fe1ca3323`

This review proves isolated branch integration only. Push, PR publication, installed runtime participation, real provider success, and Gordon activation remain outside this slice.
