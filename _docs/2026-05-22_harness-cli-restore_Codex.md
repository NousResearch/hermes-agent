# Harness CLI Restore

## Overview

Restored the top-level `hermes harness` command so `hermes harness status`,
`start`, `stop`, and `restart` are accepted by argparse again.

## Background / Requirements

The reported failure was:

```text
hermes: error: argument command: invalid choice: 'harness'
```

History showed `hermes_cli/harness.py` existed in an older local commit, but the
current branch no longer had the module or subparser registration. The current
vendor tree also lacks the actual `harness_daemon.py` script, so the CLI must
parse cleanly while reporting that runtime dependency clearly.

## Assumptions / Decisions

- This change restores only daemon management, not agent-facing harness tools.
- `status` must work even when optional process-management dependencies are
  missing from a local virtual environment.
- `start` should fail with a clear missing-script message when the daemon script
  is absent.
- `auto_start` defaults to false so a future caller must opt into background
  daemon startup explicitly.

## Changed Files

- `hermes_cli/harness.py`
- `hermes_cli/main.py`
- `hermes_cli/config.py`
- `tests/hermes_cli/test_harness.py`

## Implementation Details

- Added `hermes_cli.harness` with URL/config resolution, status probing,
  background start, stop by configured port, restart, and argparse registration.
- Added `harness` to `_BUILTIN_SUBCOMMANDS` so plugin CLI discovery remains
  skipped for this built-in command.
- Added `harness` to `_coalesce_session_name_args()` command awareness so
  session-name coalescing does not consume it as a resume title token.
- Added default config under `DEFAULT_CONFIG["harness"]`.
- Made `psutil` a lazy import, because `status` and parser registration should
  not fail in an incomplete local venv.
- Wired command return codes through `SystemExit` for `hermes harness` only.

## Commands Run

```powershell
py -3.12 -m py_compile gateway\platforms\discord.py hermes_cli\harness.py hermes_cli\main.py hermes_cli\config.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py
py -3.12 -m pytest tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py tests\hermes_cli\test_startup_plugin_gating.py -o addopts= -p no:randomly -q
py -3.12 -m hermes_cli.main harness --help
py -3.12 -m hermes_cli.main harness status
git diff --check -- gateway\platforms\discord.py hermes_cli\config.py hermes_cli\main.py hermes_cli\harness.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py
```

## Test / Verification Results

- Focused tests: `117 passed in 11.95s`.
- `py_compile`: passed.
- `hermes harness --help`: exits 0 and lists `start`, `stop`, `restart`, and
  `status`.
- `hermes harness status`: exits 1 as expected on this machine, reports OFFLINE,
  and prints the missing daemon path rather than an argparse invalid-choice error.
- `git diff --check`: no whitespace errors; Git emitted only local line-ending
  warnings for existing Windows checkout behavior.

## Residual Risks

- The daemon cannot actually start until
  `vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py`
  exists or `harness.script_path` / `HYPURA_HARNESS_SCRIPT` points to a valid
  script.
- Verification used `py -3.12` because the repo-local `.venv` does not currently
  provide the full pytest stack.

## Recommended Next Actions

- Restore or intentionally replace the Hypura Harness daemon script if runtime
  startup is required.
- If agent-facing harness tools are desired, add them separately with explicit
  tests and safety boundaries rather than silently re-enabling old tool exposure.
