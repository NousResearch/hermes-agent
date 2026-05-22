# Discord Command Sync Limit Fix

## Overview

Fixed Discord startup reconciliation so stale global slash commands are deleted
before new commands are created, avoiding a startup failure when the Discord
application already has 100 global commands. Also made `DISCORD_ALLOWED_USERS=*`
an explicit allow-all wildcard instead of treating `*` as a username to resolve.

## Background / Requirements

- Runtime log showed `400 Bad Request (error code: 30032): Maximum number of application commands reached (100)` during `upsert_global_command`.
- Runtime log also showed `Resolving 1 username(s): *` followed by `Could not resolve usernames: *`.
- Apply the project `AGENTS.md`, PC-wide MILSPECLLMOps baseline, SOP Application Development, and SOP Python guidance.
- Keep the change narrow to Discord gateway behavior and tests.

## Assumptions / Decisions

- The active local command tree currently registers 46 Discord global commands, so the 100-command failure is consistent with stale server-side commands left from older registrations, not with the current desired command count.
- Desired command counts above 100 now fail before any Discord mutation, preventing partial or misleading reconciliation.
- `DISCORD_ALLOWED_USERS=*` means allow all users and should not request the privileged members intent or trigger username resolution.

## Changed Files

- `gateway/platforms/discord.py`
- `tests/gateway/test_discord_connect.py`
- `tests/gateway/test_discord_component_auth.py`
- `tests/gateway/test_discord_slash_auth.py`

## Implementation Details

- Added constants for the Discord global command limit and Discord allowlist wildcard.
- Changed safe slash-command sync to:
  - validate desired global command count before fetching or mutating remote commands;
  - delete stale remote commands first;
  - then create missing desired commands and update or recreate changed commands.
- Preserved mutation pacing between each Discord write.
- Normalized `DISCORD_ALLOWED_USERS=*` to a single wildcard entry.
- Skipped members intent for wildcard-only user allowlists.
- Allowed wildcard through message, slash, component, and voice-member inference authorization checks.
- Added slash-command regression coverage so the wildcard path remains aligned with message authorization.

## Commands Run

```powershell
py -3.12 -m py_compile gateway\platforms\discord.py hermes_cli\harness.py hermes_cli\main.py hermes_cli\config.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py
py -3.12 -m pytest tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py tests\hermes_cli\test_startup_plugin_gating.py -o addopts= -p no:randomly -q
git diff --check -- gateway\platforms\discord.py hermes_cli\config.py hermes_cli\main.py hermes_cli\harness.py tests\gateway\test_discord_connect.py tests\gateway\test_discord_component_auth.py tests\gateway\test_discord_slash_auth.py tests\hermes_cli\test_harness.py
```

## Test / Verification Results

- `117 passed in 11.95s` for the targeted Discord gateway, harness CLI, and startup plugin gating tests with `-p no:randomly`.
- `py_compile` completed successfully for all changed Python files.
- `git diff --check` completed successfully; Git printed line-ending warnings only.

## Residual Risks

- The repo-local `.venv` currently lacks `pytest`, so verification used `py -3.12` from the machine install.
- Running without `-p no:randomly` on this machine still triggers an unrelated `pytest-randomly` / NumPy seed error before the changed tests can run.
- This change does not delete remote Discord commands immediately by itself; it changes the next safe sync so it can clear stale commands before creating missing ones.

## Recommended Next Actions

- Restart the Discord gateway so `_run_post_connect_initialization()` performs the safe reconciliation.
- If startup still reports command-count pressure, inspect the Discord application's existing global commands and compare them with the 46-command local desired set.
