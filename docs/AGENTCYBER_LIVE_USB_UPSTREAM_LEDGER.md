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

### 2026-06-21T20:38:29Z — sync upstream and redact Live USB command logs

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune`: upstream advanced from `8e4d2fd23` to `f72690825`; origin fetched cleanly.
- Drift after fetch: `HEAD..upstream/main` -> `6`; `upstream/main..HEAD` -> `66`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `184`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit before local Live USB work was `7254ba30a`.
- Post-merge preservation review confirmed the required AgentCyber/Live USB files still exist, are tracked, and were not touched by the upstream merge: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.

**Changed files**

- Upstream merge changed 14 upstream Hermes files, including desktop/bootstrap updater files, gateway stream handling/tests, `hermes_cli/main.py`, `tests/hermes_cli/test_tui_npm_install.py`, `tests/hermes_cli/test_update_zip_atomic_replace.py`, and kanban dashboard dist/tests.
- `tools/cyber_live_usb.py`: added `_redacted_command_for_log()` and routed `_run()` command logging through it so `--telegram-token`, `--model-key`, and live-USB/operator approval token flag values are redacted in logs while the original command still goes to `subprocess.run()`.
- `tests/cyber/test_live_usb_tool.py`: added regression tests proving raw command arguments are preserved for execution, secrets are absent from logs, redaction markers appear, and separate plus `--flag=value` forms are covered for each sensitive flag.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `26 passed in 0.46s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/gateway/test_stream_consumer.py tests/hermes_cli/test_tui_npm_install.py tests/hermes_cli/test_update_zip_atomic_replace.py tests/plugins/test_kanban_dashboard_plugin.py` -> `359 tests passed, 0 failed`.
- `scripts/agentcyber status --json` summary -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, `local_runtime_health_ok: true`.
- `scripts/agentcyber hermes tools list | grep -Ei 'cyber|live_usb'` -> `cyber` enabled and `live_usb` disabled.
- `python scripts/check-windows-footguns.py --all` -> `No Windows footguns found (686 file(s) scanned)`.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent upstream preservation review: `PASS`.
- Subagent redaction spec re-review after coverage fixes: `PASS`.
- Subagent redaction quality review: `APPROVED`; only a future malformed-input hardening idea was noted as non-blocking.
- Post-verification drift before commit/push: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `67`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `7`.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Committed upstream merge plus Live USB log-redaction changes on the guarded sync branch: `b5d015abe3aedcd22849a44ff4db77bf6e14b540` (`fix: redact AgentCyber live USB command logs`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `b5d015abe3aedcd22849a44ff4db77bf6e14b540`.
- Post-push drift: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `68`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- This ledger was updated after push with the commit SHA and remote verification, then committed as ledger-only follow-up `92734946969460a7c6b28ef1c7b32ac1c48d1465` (`docs: record AgentCyber live USB log redaction verification`).
- Verified after that follow-up push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `92734946969460a7c6b28ef1c7b32ac1c48d1465`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `69`.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger before taking a new implementation lane.

### 2026-06-21T20:56:44Z — sync upstream and guard Live USB provision targets

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune` plus read-only fetch of the active sync branch: upstream advanced from `f72690825` to `f79e0a706`; origin fetched cleanly.
- Drift after fetch: `HEAD..upstream/main` -> `10`; `upstream/main..HEAD` -> `70`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `194`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy. Upstream changes touched `cli.py`, gateway WhatsApp/email/auth paths, `hermes_cli/{config,gateway,main,web_server}.py`, platform adapters, release script, TUI gateway server, and upstream tests.
- Post-merge drift before local Live USB work: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `71`; branch ahead of remote sync branch by `11` commits.
- An initial combined status summarization command was blocked by the approval guard because it piped tool-list output into `python`; it was not approved and was rerun without that pipe.

**Changed files**

- Upstream merge changed 20 upstream Hermes files, including `cli.py`, `gateway/slash_commands.py`, `gateway/whatsapp_identity.py`, `hermes_cli/config.py`, `hermes_cli/gateway.py`, `hermes_cli/main.py`, `hermes_cli/web_server.py`, `plugins/platforms/{email,whatsapp}/adapter.py`, `scripts/release.py`, `tools/approval.py`, `tui_gateway/server.py`, and directly related upstream tests.
- `tools/cyber_live_usb.py`: `_provision()` now validates `Path(device).is_block_device()` after root/operator approval and after `device` is present, but before resolving or running `provision.sh`.
- `tests/cyber/test_live_usb_tool.py`: added provision guard regressions proving missing approval does not touch `Path.is_block_device()`/`_run`, approved non-block targets fail before `_script()`/`_run()`, and an approved mocked provision path preserves command construction without touching real devices.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `28 passed in 0.49s` after the final test-device wording cleanup.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/tools/test_approval_interrupt.py tests/gateway/test_whatsapp_to_jid.py tests/gateway/test_whatsapp_connect.py tests/gateway/test_email.py tests/cli/test_cli_init.py tests/gateway/test_model_command_expensive_confirm.py tests/hermes_cli/test_dashboard_auth_ws_auth.py tests/hermes_cli/test_update_concurrent_quarantine.py` -> `375 tests passed, 0 failed`.
- `scripts/agentcyber status --json` summary -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, `local_runtime_health_ok: true`; the status output reported secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec review: `PASS`.
- Subagent quality review: `APPROVED`; its minor note to use whole-device wording in provision tests was fixed before the final focused and expanded verification reruns.
- Post-verification drift before commit/push: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `71`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `11`.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Committed scoped Live USB provision guard/tests/ledger changes on the guarded sync branch: `a886bf011f0fed8d1b373d6d44f4c995412d2fec` (`fix: guard AgentCyber live USB provisioning`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `a886bf011f0fed8d1b373d6d44f4c995412d2fec`.
- Post-push drift: `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `72`.
- This is the one bounded ledger-only follow-up recording the post-push facts; after pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger before taking a new implementation lane.

### 2026-06-21T21:25:38Z — sync upstream and require removable Live USB targets

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune` plus read-only fetch of the active sync branch: upstream advanced from `f79e0a706` to `6f0ecf37d`; origin fetched cleanly.
- Drift after fetch: `HEAD..upstream/main` -> `4`; `upstream/main..HEAD` -> `73`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `207`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy. Upstream changes touched redaction, gateway stream consumption, URL safety, and related tests.
- Post-merge preservation review confirmed the required AgentCyber/Live USB files still exist and are tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.

**Changed files**

- Upstream merge changed 6 upstream Hermes files: `agent/redact.py`, `gateway/stream_consumer.py`, `tools/url_safety.py`, `tests/agent/test_redact.py`, `tests/gateway/test_stream_consumer.py`, and `tests/tools/test_url_safety.py`.
- `tools/cyber_live_usb.py`: added fail-closed removable-device verification for approved `write` and `provision` actions. The guard resolves the operator-supplied device with `strict=True`, rejects unresolved/non-`/dev` targets, requires Linux removable metadata flag exactly `1`, and returns a canonical `/dev/<block_name>` path that is passed to `write_usb.sh --device` or `provision.sh --usb` instead of any user-supplied alias.
- `tests/cyber/test_live_usb_tool.py`: expanded from 28 to 39 tests covering removable flag success, resolve failure, non-`/dev` resolved targets, non-removable flags, missing/unreadable flags, and alias canonicalization before `_run()` for both write and provision; tests remain fully mocked and perform no USB/block-device writes.
- `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`: documented that AgentCyber `write` and `provision` tool calls require verifiable removable Linux block-device metadata and that unverifiable edge-case media should be handled manually outside unattended tool control.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Initial baseline after upstream merge: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `28 passed in 0.41s`.
- Initial expanded wrapper after upstream merge: `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/agent/test_redact.py tests/tools/test_url_safety.py tests/gateway/test_stream_consumer.py` -> `439 tests passed, 0 failed`.
- After removable-guard implementation: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `39 passed in 0.57s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Final expanded wrapper: `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py tests/agent/test_redact.py tests/tools/test_url_safety.py tests/gateway/test_stream_consumer.py` -> `450 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, and secret fields reported as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec review after initial implementation: `PASS`.
- Subagent quality review after initial implementation: `REQUEST_CHANGES` for alias/symlink canonicalization before `_run()`; this was fixed with canonical `/dev/<block_name>` command construction tests.
- Subagent spec re-review after canonicalization fixes: `PASS`.
- Subagent quality re-review after canonicalization fixes: `APPROVED`; no critical, important, or minor issues.
- Post-verification drift before commit/push: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `74`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `5`.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Committed scoped upstream merge plus Live USB removable-target guard/runbook/tests/ledger changes on the guarded sync branch: `419c473bb46040ac9f77ed838c60a77c6c2f74cd` (`fix: require removable AgentCyber live USB targets`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `419c473bb46040ac9f77ed838c60a77c6c2f74cd`.
- Post-push drift: `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `75`.
- This is the bounded ledger-only follow-up recording post-push facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger before taking a new implementation lane.

### 2026-06-21T21:42:11Z — verify upstream and require exact Live USB approval tokens

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: all fetched cleanly with no upstream advancement this run.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `76`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `214`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Starting HEAD/local and remote sync branch: `37b190b29`; upstream main: `6f0ecf37d`; origin main: `480559a7e`.

**Changed files**

- `tools/cyber_live_usb.py`: made live USB approval-token comparison exact by removing `.strip()` normalization for both `HERMES_AGENTCYBER_LIVE_USB_APPROVAL` and supplied approval args while preserving `hmac.compare_digest`; blank/missing env or approval values still fail closed.
- `tests/cyber/test_live_usb_tool.py`: added regression coverage for `build`, `write`, and `provision` proving wrong token, wrong case, leading whitespace, and trailing whitespace all deny before `_script`, `_run`, or block-device checks.
- `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`: documented that high-consequence live USB approval tokens must match exactly with no whitespace trimming or case normalization.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Implementer subagent initially added wrong-token/wrong-case tests; focused test output before the whitespace hardening: `45 passed in 0.63s`.
- After exact whitespace hardening: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `51 passed in 0.69s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/hermes_cli/test_tools_config.py` -> `161 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, and secret fields reported as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec re-review after whitespace exactness: `PASS`.
- Subagent quality re-review after whitespace exactness: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Pre-commit diff stat: `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` (2-line wording update), `tests/cyber/test_live_usb_tool.py` (+51 lines), `tools/cyber_live_usb.py` (exact-comparison cleanup), and this ledger.
- Committed scoped exact-approval/runbook/tests/ledger changes on the guarded sync branch: `ba5eb84b91056d92825770e0ac77c6e47c99fc3c` (`fix: require exact AgentCyber live USB approval`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `ba5eb84b91056d92825770e0ac77c6e47c99fc3c`.
- Post-push drift: `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `77`.
- This is the bounded ledger-only follow-up recording post-push facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If this guarded branch is merged and no new Live USB gaps are found, treat future runs as verification/no-op lanes.

### 2026-06-21T21:59:52Z — verify upstream and align Live USB toolset safety wording

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: all fetched cleanly with no upstream advancement this run.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `78`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `216`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Starting HEAD/local and remote sync branch: `4172efe1c133`; upstream main: `6f0ecf37dad0`; origin main: `480559a7edeb`; origin sync branch: `4172efe1c133`.

**Changed files**

- `toolsets.py`: updated the `live_usb` registry description to state that `status/list` are safe, `build/write/provision` require root plus exact operator approval, and `write/provision` require verified removable Linux block-device metadata.
- `hermes_cli/tools_config.py`: updated the `live_usb` configurator/UI description to mention safe `status/list`, gated `build/write/provision`, root plus operator approval, and removable USB requirements for write/provision.
- `tests/hermes_cli/test_tools_config.py`: added invariant-style checks that AgentCyber `cyber`/`live_usb` stay default-off and that both configurator and registry descriptions include the Live USB safety gates without snapshotting full text.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `uv run --frozen python -m pytest tests/hermes_cli/test_tools_config.py -q -o addopts= --tb=short` -> `98 passed, 8 warnings in 13.34s`.
- `uv run --frozen python -m ruff check toolsets.py hermes_cli/tools_config.py tests/hermes_cli/test_tools_config.py` -> `All checks passed!`.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py -q -o addopts= --tb=short` -> `161 passed, 8 warnings in 16.72s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py` -> `161 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this lane had uncommitted repo-local changes, and secret fields reported as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec review: `PASS`.
- Subagent quality review: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This entry will be included in the scoped Live USB clarity commit. After that push, use one bounded ledger-only follow-up to record the new commit SHA, remote-tip verification, and final drift/status, then stop rather than amending the ledger again solely to mention the ledger-only commit.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If this guarded branch is merged and no new Live USB gaps are found, treat future runs as verification/no-op lanes.
