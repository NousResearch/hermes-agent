# AgentCyber Live USB + Upstream Sync Ledger

Started: 2026-06-21
Repo: `/home/kbun/Desktop/hermes-agentcyber`

## Mission

Finish/verify the AgentCyber Live USB feature and keep the fork synchronized with upstream Hermes without disturbing default Hermes.

## Initial findings

- `tools/cyber_live_usb.py` exists.
- `tests/cyber/test_live_usb_tool.py` exists.
- Focused current test passed before this ledger was created: `9 passed in 0.27s`.
- `git rev-list --count HEAD..upstream/main` returned `0`, so the fork is currently not behind upstream.
- `git rev-list --count upstream/main..HEAD` returned `60`, so AgentCyber has downstream commits ahead of upstream.

## Boundaries

- Work only in `/home/kbun/Desktop/hermes-agentcyber` unless doing read-only inspection.
- Do not modify default `~/.hermes`, default gateway, default cron, or default profiles.
- Do not write to a USB device, build ISO as root, install packages, use sudo, or touch block devices without explicit approval.
- Do not delete files. If cleanup is needed, record a proposal only.
- Do not push/merge unless tests pass and the change is scoped to AgentCyber.
- Preserve Cyber Edition files during upstream sync.
- Redact secrets.

## Live USB objectives

- Verify the `live_usb` tool is visible but disabled by default in standalone AgentCyber.
- Expand tests for safe status/list behavior, non-root guardrails, invalid action handling, and no accidental block-device writes.
- Check docs/runbook for clear operator steps.
- Add missing health/status commands if needed.
- Keep destructive/hardware actions gated.

## Upstream sync objectives

- Fetch upstream each run.
- If behind upstream, create a guarded sync branch and merge upstream safely.
- Preserve Cyber Edition files and run focused tests.
- If no upstream drift, record that and continue Live USB work.

## Run log

### 2026-06-21T19:50:55Z — guarded upstream sync branch

**Commands / status**

- `git status --short --branch` at start: `## main...origin/main` plus untracked `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`.
- `git fetch upstream main --prune && git fetch origin main --prune`: upstream advanced from `2ab09a6c5` to `1f4c5aed6`; origin fetched cleanly.
- Drift after fetch: `git rev-list --count HEAD..upstream/main` -> `172`; `git rev-list --count upstream/main..HEAD` -> `60`; `git rev-list --count HEAD..origin/main` -> `0`; `git rev-list --count origin/main..HEAD` -> `0`.
- Created guarded branch: `agentcyber/upstream-sync-20260621-194355`.
- Ran `git merge --no-ff upstream/main`; one conflict occurred in `website/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator.md`.

**Changed files**

- Broad upstream merge staged many upstream Hermes changes, including platform plugin moves, docs/i18n updates, gateway/cron/tool changes, desktop updates, and tests.
- Manual conflict resolution kept the existing valid Spotify docs link in `website/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator.md`.
- Fixed staged upstream whitespace/diff-check issues in:
  - `tests/hermes_cli/test_setup.py`
  - `tests/run_agent/test_codex_app_server_integration.py`
  - `tests/run_agent/test_in_place_compaction.py`
  - `website/docs/user-guide/profile-distributions.md`
- Added/updated this ledger: `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`.
- Review confirmed key AgentCyber/Live USB files still exist and had no staged/unstaged modifications from the merge lane: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `scripts/agentcyber`, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, `live-usb/build_iso.sh`, `live-usb/write_usb.sh`, `live-usb/provision.sh`.

**Verification**

- First combined `uv run --frozen python -m pytest ... -q -o addopts= --tb=short` showed one order-dependent failure in `tests/hermes_cli/test_tools_config.py::test_get_platform_tools_recovers_non_configurable_toolsets_from_composite`; rerunning that test alone passed. The isolated project wrapper was used for acceptance.
- Focused acceptance wrapper: `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py` -> `119 tests passed, 0 failed`.
- After whitespace fixes, expanded wrapper: `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_setup.py tests/run_agent/test_codex_app_server_integration.py tests/run_agent/test_in_place_compaction.py` -> `165 tests passed, 0 failed`.
- `git diff --cached --check` -> passed with no output.
- `git diff --check` -> passed with no output.
- Subagent spec review: `PASS`; no unresolved conflicts; required AgentCyber/Live USB files present and tracked.
- Subagent quality re-review after fixes: `APPROVED`; no critical or important issues.

**Blockers / boundaries**

- No USB/block-device, root, sudo, package install, cron mutation, gateway mutation, external security, cloud, hardware, or secret-disclosure actions were performed.
- No full-suite run was attempted because this cron lane stayed focused on AgentCyber/Live USB plus directly touched tests.

**Commit / push**

- Committed guarded upstream merge: `b2e66a619595f3c210ed8082275f8150aa23f059` (`merge: sync upstream Hermes into AgentCyber`).
- Pushed branch to origin without force: `origin/agentcyber/upstream-sync-20260621-194355`.
- Verified local and remote branch tips matched immediately after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `b2e66a619595f3c210ed8082275f8150aa23f059`.
- Post-merge drift check on the sync branch: `git rev-list --count HEAD..upstream/main` -> `0`; `git rev-list --count upstream/main..HEAD` -> `61`.
- GitHub reported PR creation URL: `https://github.com/breakingcircuits1337/hermes-agentcyber/pull/new/agentcyber/upstream-sync-20260621-194355`.

**Next lane**

- Open/review the pushed guarded sync branch and merge it into AgentCyber main when approved; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger before taking any new implementation lane.

### 2026-06-21T20:14:51Z — update sync branch and gate Live USB mutations

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune`: upstream advanced from `1f4c5aed6` to `8e4d2fd23`; origin fetched cleanly.
- Drift after fetch: `HEAD..upstream/main` -> `6`; `upstream/main..HEAD` -> `62`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `174`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; upstream files changed included `hermes_cli/backup.py`, `hermes_cli/kanban_db.py`, `tests/hermes_cli/test_backup.py`, `tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py`, `tests/hermes_cli/test_plugins.py`, `ui-tui/README.md`, and `website/docs/guides/build-a-hermes-plugin.md`.
- Post-merge drift before local commit: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `63`; branch ahead of remote sync branch by `7` commits.

**Changed files**

- `tools/cyber_live_usb.py`: added `HERMES_AGENTCYBER_LIVE_USB_APPROVAL` fail-closed token gating for `build`, `write`, and `provision`; approval is checked after root and before script execution, and for `write` before block-device checks. `status` and `list_usb` remain approval-free read-only actions. Updated schema text to say build/write/provision require root plus operator approval.
- `tests/cyber/test_live_usb_tool.py`: added root-simulated fail-closed tests proving build/write/provision do not invoke scripts without approval; added an approved write-path command-construction test with `_run`, block-device, and ISO checks mocked so no real USB/block-device write occurs.
- `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`: documented the live USB operator approval token, read-only status/list behavior, and cron repair prohibitions.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `13 passed in 0.35s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_backup.py tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py tests/hermes_cli/test_plugins.py` -> `351 tests passed, 0 failed`.
- Re-ran the same `scripts/run_tests.sh ...` command after the schema-description cleanup -> `351 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, and secret fields reported as booleans only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent next-gap review found the operator approval gap in the Live USB tool.
- Subagent spec review: `PASS`.
- Subagent quality review: `APPROVED`; minor schema wording note was fixed before final verification.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- A status command contacted the configured local Ollama health endpoint only (`http://192.168.1.120:11434/api/tags`); no secrets were printed.

**Commit / push**

- Committed scoped Live USB guard/runbook/ledger changes: `d600340a0202f22cccad42ad9ef209dbed37d264` (`fix: gate AgentCyber live USB mutations`).
- Pushed guarded sync branch to origin without force: `origin/agentcyber/upstream-sync-20260621-194355`.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `d600340a0202f22cccad42ad9ef209dbed37d264`.
- Post-push drift: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `64`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- This ledger was updated after push with the new SHA/remote verification, then committed as ledger-only follow-up `0470c880f3792bc00142dddd74832c3df4d1e9da` (`docs: record AgentCyber live USB sync verification`).
- Verified after that follow-up push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `0470c880f3792bc00142dddd74832c3df4d1e9da`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.

**Next lane**

- Open/review the pushed guarded sync branch and merge it into AgentCyber main when approved; do not force-push.
- After the pushed branch is reviewed/merged into AgentCyber main, future cron runs should be verification/no-op unless upstream drifts again or a new focused Live USB gap is found.
