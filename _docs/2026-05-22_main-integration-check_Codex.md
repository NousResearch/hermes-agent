# Main integration check

Date: 2026-05-22

## Overview

Verified the current feature branch against `origin/main` in a separate
integration worktree before any main-branch landing. The check resolved merge
conflicts, ran focused verification, and kept the original checkout and main
branch untouched.

## Background / Requirements

- The feature branch was behind `origin/main` and needed a main-integration
  check before it could be considered safe to merge.
- The branch already contained Discord gateway fixes, Harness CLI restoration,
  and Skills Hub uninstall path hardening.
- The integration branch was created from `origin/main` so the merge result
  matches the direction expected for a main landing.

## Assumptions / Decisions

- Use a separate worktree for conflict resolution so the original feature branch
  remains preserved.
- Resolve conflicts by keeping the newer main-branch Harness auto-start and
  Discord allow-all model while retaining the feature branch's safety fixes.
- Treat Windows-only test failures from line endings and path separators as real
  integration issues because the project is actively developed on Windows.
- Do not run a formatting rewrite across the touched files, because `ruff
  format --check` would reformat broad existing surfaces outside the functional
  integration concern.
- During the pre-merge review, fix required issues directly on the integration
  branch instead of leaving known regressions for main.

## Changed Files

- `gateway/platforms/discord.py`
- `hermes_cli/config.py`
- `hermes_cli/harness.py`
- `hermes_cli/main.py`
- `tools/skills_guard.py`
- `tools/skills_hub.py`
- `tests/gateway/test_discord_component_auth.py`
- `tests/gateway/test_discord_connect.py`
- `tests/gateway/test_discord_slash_auth.py`
- `tests/hermes_cli/test_harness.py`
- `tests/tools/test_skills_hub.py`
- `_docs/2026-05-22_discord-command-sync-limit_Codex.md`
- `_docs/2026-05-22_harness-cli-restore_Codex.md`
- `_docs/2026-05-22_skills-uninstall-path-guard_codex.md`
- `_docs/2026-05-22_main-integration-check_Codex.md`

## Implementation Details

- Resolved Discord conflicts by combining main's explicit allow-all flags with
  the wildcard allowlist behavior and guarded command sync.
- Resolved Harness conflicts by retaining the restored CLI surface and adding
  main's `ensure_harness_running()` auto-start guard.
- Replaced the inline `harness` argparse block in `hermes_cli/main.py` with the
  reusable `register_harness_subparser()` function.
- Normalized optional skill bundle keys to POSIX-style paths.
- Added canonical text content hashing so Skills Hub bundle hashes and installed
  skill hashes are stable across Windows CRLF and Unix LF checkouts while binary
  assets remain byte-preserved.
- Fixed a review finding in `stop_harness_daemon()`: an unhealthy harness
  process that fails `/status` can still be listening on the configured port,
  so `hermes harness stop` must still terminate the process instead of treating
  it as already stopped.

## Commands Run

```powershell
git worktree add -b codex/hermes-main-integration-check-20260522 C:\Users\downl\Desktop\hermes-main-integration-check-20260522 origin/main
git merge --no-commit --no-ff origin/codex/hermes-skills-uninstall-path-guard-20260522
py -3.12 -m py_compile gateway\platforms\discord.py hermes_cli\harness.py hermes_cli\main.py hermes_cli\config.py tools\skills_guard.py tools\skills_hub.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py tests\tools\test_skills_hub.py tests\hermes_cli\test_startup_plugin_gating.py
py -3.12 -m pytest tests\tools\test_skills_hub.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py tests\hermes_cli\test_startup_plugin_gating.py -o addopts= -p no:randomly -q
py -3.12 -m ruff check tools\skills_guard.py tools\skills_hub.py gateway\platforms\discord.py hermes_cli\harness.py hermes_cli\main.py hermes_cli\config.py tests\tools\test_skills_hub.py tests\hermes_cli\test_harness.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py
py -3.12 -m ruff format --check tools\skills_guard.py tools\skills_hub.py gateway\platforms\discord.py hermes_cli\harness.py hermes_cli\main.py hermes_cli\config.py tests\tools\test_skills_hub.py tests\hermes_cli\test_harness.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py
py -3.12 -m hermes_cli.main harness --help
py -3.12 -m hermes_cli.main harness status
py -3.12 scripts\run_tests_parallel.py -j 4 --file-timeout 300
py -3.12 scripts\run_tests_parallel.py -j 4 tests\tools\test_skills_hub.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py tests\hermes_cli\test_startup_plugin_gating.py -- -o addopts= -p no:randomly -q
git diff --check
```

## Test / Verification Results

- `py_compile`: passed for the integration touch set.
- Focused pytest suite before review fix: `241 passed, 1 skipped`.
- Focused pytest suite after review fix: `242 passed, 1 skipped`.
- Per-file isolation run for the touched test files: `6 files`, `242 tests
  passed`, `0 failed`.
- Skipped test: directory symlink escape case, because this Windows host lacks
  directory symlink privilege (`WinError 1314`).
- `ruff check`: passed.
- `ruff format --check`: reported that the touched files would be reformatted;
  no formatting rewrite was applied to avoid unrelated churn.
- `hermes harness --help`: exits 0 and lists `start`, `stop`, `restart`, and
  `status`.
- `hermes harness status`: exits 1 as expected on this machine and reports
  Harness offline at `http://127.0.0.1:18794`.
- `git diff --check`: no whitespace errors.
- Full `tests/` per-file isolation was attempted with `-j 4 --file-timeout
  300`, but exceeded a 30-minute wall-clock cap on this Windows host and was
  stopped. No failed test result was produced from that run.

## Residual Risks

- The complete repository test suite did not complete on this Windows host;
  verification remains focused on the merge-conflict and changed-behavior
  surfaces plus per-file isolation for the touched test files.
- Harness runtime startup still depends on a valid configured daemon script.
- The directory symlink path-escape case remains environment-skipped on this
  Windows host because elevated symlink privileges are unavailable.

## Recommended Next Actions

- If broader release confidence is required beyond the focused integration
  scope, run the full project test suite on CI or a developer host with the
  standard Hermes test environment and enough wall-clock budget.
