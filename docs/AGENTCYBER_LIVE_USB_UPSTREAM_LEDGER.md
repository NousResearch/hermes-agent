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

- Committed scoped Live USB toolset wording/tests/ledger changes on the guarded sync branch: `44b93aab43b2f3d30320afa7c2ae15f1683b9c5f` (`docs: clarify AgentCyber live USB safety gates`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `44b93aab43b2f3d30320afa7c2ae15f1683b9c5f`.
- Post-push drift: `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `79`.
- This is the bounded ledger-only follow-up recording post-push facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If this guarded branch is merged and no new Live USB gaps are found, treat future runs as verification/no-op lanes.

### 2026-06-21T22:16:04Z — sync upstream and verify Live USB lane remains complete

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree at `cc99ac2d449a`.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: upstream advanced from `6f0ecf37d` to `bb59075b2`; origin fetched cleanly.
- Drift after fetch before merge: `HEAD..upstream/main` -> `2`; `upstream/main..HEAD` -> `80`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `218`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `cec4eb06dd725defc8309c19747de6bf7f265d98`.
- Post-merge drift before ledger commit: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `81`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `3`.

**Changed files**

- Upstream merge changed only `hermes_constants.py` and `tests/test_hermes_constants.py`.
- Required AgentCyber/Live USB files were preserved and still present: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.
- No Live USB implementation or runbook code/docs were changed this run because focused review found no new smallest safety/test/doc gap.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/test_hermes_constants.py -q -o addopts= --tb=short` -> `212 passed, 8 warnings in 15.70s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/test_hermes_constants.py` -> `212 tests passed, 0 failed`.
- `scripts/agentcyber status --json` summary -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields reported as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker check: `^<<<<<<< ` and `^>>>>>>> ` searches returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent Live USB next-gap review: `PASS`; no focused safety/test/doc gap worth changing this run.
- Subagent upstream preservation review: `PASS`; required AgentCyber files present, merge touched only upstream constants files, focused tests passed in the subagent context.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.
- Deviation noted: one status/tool-list summarization used temporary evidence files under `/tmp` (`/tmp/agentcyber-status.json`, `/tmp/agentcyber-tools-list.txt`) before correction; they contained status/tool-list output only, no approval tokens or secrets were printed. Per the no-delete boundary, no cleanup was attempted in this cron run.

**Commit / push**

- Upstream merge commit already created by `git merge`: `cec4eb06dd725defc8309c19747de6bf7f265d98`.
- This is the bounded ledger-only follow-up recording the merge and verification facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If this guarded branch is merged and no upstream drift or new Live USB gap is found, treat the lane as verification/no-op.

### 2026-06-21T22:34:21Z — sync upstream and verify Live USB lane remains complete

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree at `e6955345330b`.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: upstream advanced from `bb59075b2` to `624580e83`; origin fetched cleanly.
- Drift after fetch before merge: `HEAD..upstream/main` -> `5`; `upstream/main..HEAD` -> `82`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `222`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `7b4a5182be3eba9b5907f97cc4d1020b4a8718b6` with `HEAD^2` equal to upstream `624580e8363f5dcd5903a01d483d6f006f5be9d9`.
- Post-merge drift before ledger commit: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `83`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `6`.

**Changed files**

- Upstream merge changed 5 upstream Hermes files: `gateway/session.py`, `scripts/release.py`, `tests/gateway/test_session.py`, `tests/tools/test_browser_orphan_reaper.py`, and `tools/browser_tool.py`.
- Required AgentCyber/Live USB files were preserved and still present: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.
- No Live USB implementation, toolset, or runbook code/docs were changed this run because focused review found no new smallest safety/test/doc gap.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/gateway/test_session.py tests/tools/test_browser_orphan_reaper.py -q -o addopts= --tb=short` -> `275 passed, 7 warnings in 16.90s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/gateway/test_session.py tests/tools/test_browser_orphan_reaper.py` -> `275 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields reported as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker line check outside `.git`, `.venv`, `venv`, `node_modules`, and `.agentcyber-home` -> `0` matches for lines starting `<<<<<<< ` or `>>>>>>> `.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent Live USB next-gap review: `PASS`; no focused safety/test/doc gap worth changing this cron run.
- Subagent upstream preservation review: `PASS`; required AgentCyber files present, no unmerged files, `HEAD^2` equals upstream, and merge changes were limited to expected upstream files.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Upstream merge commit already created by `git merge`: `7b4a5182be3eba9b5907f97cc4d1020b4a8718b6`.
- This is the bounded ledger-only follow-up recording the merge and verification facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If this guarded branch is merged and no upstream drift or new Live USB gap is found, treat the lane as verification/no-op.

### 2026-06-21T22:57:15Z — sync upstream and document Live USB agent gates in README

**Commands / status**

- `git status --short --branch`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree at `a5aaa2b24d239433bfd505e14eb7ce279a627882`.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: upstream advanced from `624580e83` to `745c4db23`; origin fetched cleanly. The combined tip-print command ended with `fatal: Needed a single revision`, but the fetches and drift counts completed and individual tip checks were rerun successfully.
- Drift after fetch before merge: `HEAD..upstream/main` -> `1`; `upstream/main..HEAD` -> `84`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `229`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `ea9567b65bb804f28a748863b51c9b173bf20e2d` with `HEAD^2` equal to upstream `745c4db235bdb09beb19564f66727dc1f43e4fe2`.
- Post-merge drift before the README/docs lane: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `85`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `2`.

**Changed files**

- Upstream merge changed 5 upstream desktop/i18n files: `apps/desktop/electron/main.cjs`, `apps/desktop/src/i18n/en.ts`, `apps/desktop/src/i18n/ja.ts`, `apps/desktop/src/i18n/zh-hant.ts`, and `apps/desktop/src/i18n/zh.ts`.
- `README.md`: clarified that manual Live USB shell examples are operator procedures; documented that AgentCyber `live_usb` is disabled by default; `status`/`list_usb` are read-only; `build`/`write`/`provision` require root plus exact operator approval through `HERMES_AGENTCYBER_LIVE_USB_APPROVAL` and `operator_approval`; `write`/`provision` require verified removable Linux block-device metadata and canonical `/dev/...` targets; unattended cron lanes must not set the approval token, run `sudo`, build, write, or provision.
- `tests/cyber/test_live_usb_docs.py`: added README Live USB docs invariant tests and a regression against root/sudo-alone wording.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Pre-gap baseline after upstream merge: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/test_desktop_electron_pin.py tests/test_desktop_mac_entitlements.py tests/hermes_cli/test_gui_command.py -q -o addopts= --tb=short` -> `212 passed, 8 warnings in 19.08s`.
- Pre-gap baseline wrapper: `scripts/run_tests.sh tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/test_desktop_electron_pin.py tests/test_desktop_mac_entitlements.py tests/hermes_cli/test_gui_command.py` -> `212 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after merge -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` after merge -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` -> `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` after merge -> passed with no output.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked, no unmerged/conflict state, and merge changes limited to expected upstream desktop/i18n files.
- Subagent Live USB next-gap review: `REQUEST_CHANGES`; README Live USB section still implied root/sudo was sufficient for agent `build`/`write` and omitted exact approval/removable metadata guidance from tool-call examples. This was fixed in the README/docs-test lane.
- Initial docs regression run after adding the test failed as intended on an over-specific test phrase (`canonical /dev/` with Markdown backticks in README); the test was corrected to assert `canonical` and `/dev/` separately.
- Final focused docs/toolset run: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py -q -o addopts= --tb=short` -> `151 passed, 8 warnings in 15.06s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py` -> `All checks passed!`.
- Final expanded wrapper: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/test_desktop_electron_pin.py tests/test_desktop_mac_entitlements.py tests/hermes_cli/test_gui_command.py` -> `214 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after docs changes -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because README/test/ledger changes were uncommitted, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` after docs changes -> `cyber` enabled and `live_usb` disabled.
- Final pre-commit `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent docs spec re-review: `PASS`.
- Subagent docs quality review: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Upstream merge commit already created by `git merge`: `ea9567b65bb804f28a748863b51c9b173bf20e2d`.
- Committed scoped README/docs-test/ledger changes on the guarded sync branch: `52144539ea733438f8b5dbe5399dc47ecc7cfaa6` (`docs: document AgentCyber live USB approval gates`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified local and remote branch tips matched after push: `git rev-parse HEAD` and `git rev-parse origin/agentcyber/upstream-sync-20260621-194355` both returned `52144539ea733438f8b5dbe5399dc47ecc7cfaa6`.
- Post-push drift: `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `86`.
- This is the bounded ledger-only follow-up recording the post-push facts. After pushing this ledger commit, final verification should check local HEAD equals the remote branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Push this bounded ledger-only follow-up to the guarded sync branch and verify local HEAD equals the remote branch tip.
- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.

### 2026-06-21T23:19:23Z — verify upstream and fix Live USB non-root guidance

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse --short HEAD && git rev-parse HEAD`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `6ace3111194a73a092ddff51b081ac51f64dc49b` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: all fetched cleanly with no upstream advancement; the combined multi-ref tip-print command ended with `fatal: Needed a single revision`, and individual tip checks were rerun successfully.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `87`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `233`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Individual tips: `HEAD=6ace31111`; `upstream/main=745c4db23`; `origin/main=480559a7e`; `origin/agentcyber/upstream-sync-20260621-194355=6ace31111`.

**Changed files**

- `tools/cyber_live_usb.py`: updated the module action summary and non-root `build`, `write`, and `provision` responses so AgentCyber tool-call guidance says high-consequence live USB actions require root plus exact operator approval, and explicitly says root alone is not sufficient. Removed the prior sudo-only destructive write hint from the non-root tool response.
- `tests/cyber/test_live_usb_tool.py`: added a parametrized regression proving non-root `build`, `write`, and `provision` responses mention root plus operator approval/root-alone insufficiency, do not include the old sudo-only write command, and fail before `_script()`, `_run()`, or block-device checks.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Baseline focused run before the messaging patch: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py -q -o addopts= --tb=short` -> `163 passed, 8 warnings in 18.08s`.
- Baseline wrapper before the messaging patch: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py` -> `163 tests passed, 0 failed`.
- `scripts/agentcyber status --json` before patch -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` before patch -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` -> `0` matches.
- Pre-patch `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked, no unmerged/conflict state, and local `HEAD` matched the origin sync branch.
- Subagent Live USB next-gap review: `REQUEST_CHANGES`; non-root `build`/`write`/`provision` responses still implied root/sudo was the missing condition. This run fixed that gap.
- RED regression before implementation: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py::TestLiveUsbTool::test_destructive_actions_non_root_guidance_requires_more_than_sudo -q -o addopts= --tb=short` -> `3 failed` because the responses omitted `operator approval`.
- GREEN regression after implementation: same focused test -> `3 passed in 0.21s`.
- Focused docs/toolset/tool run after implementation: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py tests/hermes_cli/test_tools_config.py -q -o addopts= --tb=short` -> `154 passed, 8 warnings in 15.93s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Expanded wrapper after implementation: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py` -> `166 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after implementation -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this lane had uncommitted repo-local changes, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` after implementation -> `cyber` enabled and `live_usb` disabled.
- Final pre-commit `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec re-review after implementation: `PASS`.
- Subagent quality re-review after implementation: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Pre-commit diff stat: `tools/cyber_live_usb.py` and `tests/cyber/test_live_usb_tool.py`, plus this ledger.
- Committed and pushed scoped non-root guidance/ledger lane as `a1cc0ac5d24f9fa0ee50c68c4e44a54577cd59d9` (`fix: clarify AgentCyber live USB root gate`) on `origin/agentcyber/upstream-sync-20260621-194355`.
- The next run verified local `HEAD` and `origin/agentcyber/upstream-sync-20260621-194355` both at `a1cc0ac5d24f` before making its own ledger-only update.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, treat the lane as verification/no-op.

### 2026-06-21T23:35:36Z — no-op upstream and Live USB verification

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with a clean worktree.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: all fetched cleanly with no upstream advancement.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `88`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `234`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Tips after fetch: `HEAD=a1cc0ac5d24f9fa0ee50c68c4e44a54577cd59d9`; `upstream/main=745c4db235bd`; `origin/main=480559a7edeb`; `origin/agentcyber/upstream-sync-20260621-194355=a1cc0ac5d24f`.

**Changed files**

- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: recorded the previous pushed non-root guidance commit facts and this no-op verification run.
- No Live USB implementation, toolset, README, or runbook code/docs were changed because both focused review subagents found no new sync or Live USB gap.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py -q -o addopts= --tb=short` -> `166 passed, 8 warnings in 18.19s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py` -> `166 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker searches for lines starting `<<<<<<< ` and `>>>>>>> ` -> `0` matches.
- Required Live USB files were present: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Subagent Live USB next-gap review: `PASS`; no focused safety/test/docs change needed. Its isolated blocked pipe-to-`python` convenience attempt was not approved and it reran direct checks.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files are present/tracked, local branch matches remote sync branch, and no sync action is needed.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This is a bounded ledger-only follow-up for the no-op verification run. After committing and pushing this ledger entry, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T00:51:29Z — harden Live USB approval redaction and audit logs

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with local `HEAD` ahead of the origin sync branch by 9 commits and an unexpected dirty redaction/audit lane in `agent/redact.py`, `gateway/builtin_hooks/cyber_audit.py`, `tests/agent/test_redact.py`, and `tests/gateway/test_cyber_audit_hook.py`.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly; upstream advanced from `c768c4b71` to `73340d8be`.
- Drift after fetch before this commit: `HEAD..upstream/main` -> `19`; `upstream/main..HEAD` -> `90`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `244`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `9`.
- Starting tips: `HEAD=ba6d6158c41d`; `upstream/main=73340d8be650`; `origin/main=480559a7edeb`; `origin/agentcyber/upstream-sync-20260621-194355=fb602389b0cc`.

**Changed files**

- `agent/redact.py`: broadened forced secret redaction for AgentCyber Live USB/operator approval tokens while preserving broad URL pass-through and code-file source visibility. Added exact/canonical approval-key handling for env, prose, CLI flag, form body, URL query, access-log, JSON/Python repr, percent-encoded key variants, quoted values with escaped delimiters, shell separators, and `x-amz-signature` form-key compatibility.
- `gateway/builtin_hooks/cyber_audit.py`: routed audit string values and tool result previews through forced redaction; redacts exact approval-token keys while preserving benign approval metadata keys.
- `tests/agent/test_redact.py`: expanded regression coverage for Live USB approval redaction, encoded keys, quoted/unquoted values with spaces, suffix/separator preservation, code-file false positives, single-field form false positives, hyphenated sensitive keys, and prior URL pass-through invariants.
- `tests/gateway/test_cyber_audit_hook.py`: covered audit redaction of approval keys/commands/results and benign approval metadata preservation.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Initial focused verification before review fixes: `uv run --frozen python -m pytest tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py -q -o addopts= --tb=short` -> `269 passed, 7 warnings`.
- Iterative reviewer-driven RED/GREEN fixes covered code-file false positives, single-field form false positives, encoded approval keys, mixed/escaped quoted values, suffix preservation, larger approval-like key false positives, exact audit key handling, `x-amz-signature`, and shell separator preservation.
- Final focused redaction/audit run: `uv run --frozen python -m pytest tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py -q -o addopts= --tb=short` -> `140 passed in 1.92s`.
- `uv run --frozen python -m ruff check agent/redact.py gateway/builtin_hooks/cyber_audit.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `All checks passed!`.
- Expanded wrapper: `scripts/run_tests.sh tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py` -> `306 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, secret fields as booleans/presence only, and git `dirty: true` only because this lane was uncommitted.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec review initially found concrete gaps; after fixes, final spec review: `PASS`.
- Subagent quality review initially returned `REQUEST_CHANGES`; after fixes, final quality review: `APPROVED`.

**Blockers / boundaries**

- Upstream drift exists (`HEAD..upstream/main` -> `19` before this local commit), so the next safe action is to commit this verified dirty lane, then merge upstream on the guarded sync branch and rerun focused tests.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Pending: commit this verified redaction/audit/ledger lane, merge upstream drift, rerun focused tests, push without force, then verify local `HEAD` equals the remote sync branch tip.

**Next lane**

- Commit this verified redaction/audit lane and merge current upstream drift on the guarded sync branch.
- After merge, preserve AgentCyber/Live USB files and rerun focused tests before push.

### 2026-06-22T01:10:26Z — finish upstream merge and verify Live USB lane remains complete

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` with local `HEAD` ahead of the origin sync branch by 10 commits and an in-progress upstream merge conflict in `website/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator.md`.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly; `upstream/main` remained `73340d8be6504425b008a3d56daeeac979ae5fa6` for this merge.
- Drift before resolving the merge: `HEAD..upstream/main` -> `19`; `upstream/main..HEAD` -> `91`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `10`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `245`.
- Resolved the generated skill-doc conflict by dropping stale `kanban-orchestrator`/`kanban-worker` links removed upstream while preserving the valid local `spotify` docs link.
- `git add website/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator.md` cleared the unmerged index; `git diff --name-only --diff-filter=U` returned no files.
- `git commit --no-edit` completed the upstream merge as `067df7c13a4f941fdd30c996e9bfbf0841b2b8dc`; `MERGE_HEAD` no longer exists.
- Drift after the merge commit: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `92`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `30`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `265`.

**Changed files**

- Previous interrupted lane commit preserved on the local branch: `28ede95dce9eb898918e9f5e842f79253ed32e4e` (`fix: redact AgentCyber live USB approvals`).
- Upstream merge `067df7c13a4f941fdd30c996e9bfbf0841b2b8dc` brought in upstream changes across prompt/skill utils, desktop updater/composer/UI files, gateway status/WhatsApp tests, kanban/process/code-execution files, mem0 plugin files, release script, generated skills docs, and related tests/docs.
- Manual conflict resolution touched `website/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator.md` only.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- No Live USB implementation, toolset, README, or runbook behavior was changed this run because read-only review found no new focused safety/test/docs gap.

**Verification**

- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/gateway/test_status.py tests/gateway/test_whatsapp_connect.py tests/gateway/test_whatsapp_bridge_pidfile.py tests/hermes_cli/test_kanban_core_functionality.py tests/hermes_cli/test_kanban_goal_mode.py tests/tools/test_kanban_tools.py tests/tools/test_process_registry.py tests/tools/test_code_execution.py -q -o addopts= --tb=short` -> `839 passed, 8 warnings in 53.46s`.
- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/gateway/test_status.py tests/gateway/test_whatsapp_connect.py tests/gateway/test_whatsapp_bridge_pidfile.py tests/hermes_cli/test_kanban_core_functionality.py tests/hermes_cli/test_kanban_goal_mode.py tests/tools/test_kanban_tools.py tests/tools/test_process_registry.py tests/tools/test_code_execution.py` -> `839 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the merge commit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` -> `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files are present/tracked, no unmerged files remain, no conflict markers remain, shell scripts retain executable index modes, and required Live USB files were not modified by the staged upstream merge.
- Subagent Live USB safety/docs next-gap review: `PASS`; no smallest safety/test/docs gap found.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Completed upstream merge commit: `067df7c13a4f941fdd30c996e9bfbf0841b2b8dc`.
- This is the bounded ledger-only follow-up recording the merge and verification facts. After pushing this ledger commit, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Push this bounded ledger-only follow-up to the guarded sync branch and verify local `HEAD` equals the remote branch tip.
- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.

### 2026-06-22T01:38:14Z — sync upstream and fix Live USB status readiness gates

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `78e669b2c2fe2db700984771cd3df75de3bd8e52`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly; upstream advanced from `73340d8be` to `2b3a4f0af`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `1`; `upstream/main..HEAD` -> `93`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `266`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `3484d74c5794d67a6a9ac36545da431b94f73d54` with `HEAD^2` equal to upstream `2b3a4f0af80f`.
- Drift after merge before local status-readiness fix: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `94`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `2`.

**Changed files**

- Upstream merge changed 3 upstream files: `agent/agent_runtime_helpers.py`, `tests/run_agent/test_deepseek_reasoning_content_echo.py`, and `tests/run_agent/test_run_agent.py`.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- `tools/cyber_live_usb.py`: changed `status` output so dependency readiness is reported separately as `build_dependencies_ready` / `write_dependencies_ready`, while legacy `can_build` / `can_write` now also require root plus the live USB approval env gate. Added explicit `operation_gates` text for root plus exact operator approval, removable Linux block-device metadata, canonical `/dev` targets, and safe `status`/`list_usb` actions.
- `tests/cyber/test_live_usb_tool.py`: added a regression for non-root/no-approval status with all dependencies mocked present, proving `can_build` and `can_write` remain false and gate guidance is present.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Initial combined focused pytest after upstream merge showed the known order-dependent `tests/hermes_cli/test_tools_config.py::test_get_platform_tools_recovers_non_configurable_toolsets_from_composite` failure (`terminal` missing from `enabled` in the combined process); rerunning that test alone passed: `1 passed, 8 warnings in 4.19s`.
- Pre-fix wrapper acceptance after upstream merge: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/run_agent/test_deepseek_reasoning_content_echo.py tests/run_agent/test_run_agent.py` -> `734 tests passed, 0 failed`.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked, no unmerged entries, no conflict markers, and merge changed only the expected upstream files.
- Subagent Live USB next-gap review: `REQUEST_CHANGES`; `status` reported `can_write: true` from `dd` availability alone on a non-root/no-approval run. This run fixed that gap.
- After the fix: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `55 passed in 0.85s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- A repeated combined `uv run --frozen python -m pytest ... -q -o addopts= --tb=short` again hit the same known order-dependent `test_tools_config.py` failure, while the isolated wrapper path passed.
- Final wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/run_agent/test_deepseek_reasoning_content_echo.py tests/run_agent/test_run_agent.py` -> `735 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, secret fields as booleans/presence only, and git `dirty: true` only because this lane was uncommitted.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent spec review after the fix: `PASS`.
- Subagent quality review after the fix: `APPROVED`; no critical or important issues, only future positive-case coverage suggestions.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.
- One attempted inline `python -c` status summarization command was blocked by the approval guard and was not approved; verification relied on tests and safe status/tool-list commands instead.

**Commit / push**

- Committed upstream merge plus scoped Live USB status-readiness/ledger lane as `b06fcbe00d2174e740c515670aef298bc8ac2055` (`fix: clarify AgentCyber live USB status gates`).
- Pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force.
- Verified immediately after push: local `HEAD` and `origin/agentcyber/upstream-sync-20260621-194355` both returned `b06fcbe00d2174e740c515670aef298bc8ac2055`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `95`.
- This is the bounded ledger-only follow-up recording the post-push facts at `2026-06-22T01:38:56Z`. After pushing this ledger commit, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T01:53:29Z — no-op upstream and Live USB verification

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `27425c38d7f239091e31c77033ebe30b6d09afee`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly with no upstream advancement.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `96`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `270`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Tips after fetch: `HEAD=27425c38d7f2`; `upstream/main=2b3a4f0af80f`; `origin/main=480559a7edeb`; `origin/agentcyber/upstream-sync-20260621-194355=27425c38d7f2`.

**Changed files**

- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: recorded this no-op verification run.
- No Live USB implementation, toolset, README, or runbook code/docs were changed because focused verification and review found no new sync or Live USB gap.

**Verification**

- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `307 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker check outside `.git`, `.venv`, `venv`, `node_modules`, and `.agentcyber-home` -> `0` matches for lines starting `<<<<<<< ` or `>>>>>>> `.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Subagent upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked, no unmerged files, local branch matches the remote sync branch, and no upstream sync action is needed.
- Subagent Live USB safety/docs next-gap review: `PASS`; no smallest safety/test/docs gap worth changing this run.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This is a bounded ledger-only no-op verification entry. After committing and pushing this ledger entry, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T02:11:52Z — sync upstream and verify Live USB lane remains complete

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly; upstream advanced from `2b3a4f0af` to `41fe086eb`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `6`; `upstream/main..HEAD` -> `97`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `271`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `a42bc2413823f4136acdd9b5a41364f2b7de6be1` with `HEAD^2` equal to upstream `41fe086eb6f5a96da909d1127e40aef8829dbf18`.
- Drift after merge before this follow-up: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `98`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `7`.

**Changed files**

- Upstream merge changed 16 upstream files: `docker/s6-rc.d/dashboard/run`, `gateway/platforms/api_server.py`, `gateway/run.py`, `hermes_cli/mcp_security.py`, `hermes_cli/security_audit_startup.py`, `hermes_cli/subcommands/dashboard.py`, `hermes_cli/web_server.py`, `mini_swe_runner.py`, `tests/docker/test_dashboard.py`, `tests/gateway/test_weak_credential_guard.py`, `tests/hermes_cli/test_dashboard_auth_gate.py`, `tests/hermes_cli/test_mcp_security.py`, `tests/hermes_cli/test_security_audit_startup.py`, `trajectory_compressor.py`, `website/docs/user-guide/docker.md`, and `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/docker.md`.
- `mini_swe_runner.py`: fixed one trailing-whitespace line introduced by the upstream merge after `git diff --check HEAD~1..HEAD` reported `mini_swe_runner.py:680: trailing whitespace`.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- No Live USB implementation, toolset, README, or runbook behavior was changed because focused verification and review found no new safety/test/docs gap.

**Verification**

- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/docker/test_dashboard.py tests/gateway/test_weak_credential_guard.py tests/hermes_cli/test_dashboard_auth_gate.py tests/hermes_cli/test_mcp_security.py tests/hermes_cli/test_security_audit_startup.py` -> `379 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false` at that point, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required file presence check found all required Live USB files present.
- Conflict marker check for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.
- `git diff --check` and `git diff --cached --check` passed after the whitespace cleanup.
- Subagent upstream preservation review: `PASS`; no unmerged entries, no conflict markers, required AgentCyber files present/tracked, AgentCyber path count unchanged across the merge, and no AgentCyber paths changed by the merge.
- Subagent Live USB safety/docs next-gap review: `PASS`; no smallest safety/test/docs gap worth changing this run.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Completed upstream merge commit: `a42bc2413823f4136acdd9b5a41364f2b7de6be1`.
- This is the bounded follow-up recording the merge, whitespace cleanup, and verification facts. After committing and pushing this follow-up, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the follow-up commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T02:29:15Z — sync upstream and verify Live USB lane remains complete

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `5ccec7f522dfac77186c25aa32c129d80624b1bf`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly; upstream advanced from `41fe086eb` to `5bf23ff25`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `4`; `upstream/main..HEAD` -> `99`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `279`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `adeb8d0a5f153db18daf5f47ad571bf217d80a9d` with `HEAD^2` equal to upstream `5bf23ff251ed54961f5560d2d2f95474dcc09386`.
- Drift after merge before this follow-up: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `100`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `5`.

**Changed files**

- Upstream merge changed 6 upstream files: `hermes_cli/banner.py`, `hermes_cli/config.py`, `scripts/release.py`, `tests/hermes_cli/test_banner.py`, `tests/tools/test_process_registry.py`, and `tools/process_registry.py`.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- No Live USB implementation, toolset, README, or runbook behavior was changed because focused verification and review found no new safety/test/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/hermes_cli/test_banner.py tests/tools/test_process_registry.py` -> `402 tests passed, 0 failed`.
- `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Subagent upstream preservation review: `PASS`; no merge/conflict state, required AgentCyber/Live USB files present/tracked, executable modes preserved for `scripts/agentcyber` and `live-usb/*.sh`, and latest merge did not touch required Live USB files.
- Subagent Live USB safety/docs next-gap review: `PASS`; no smallest safety/test/docs gap worth changing this run.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Completed upstream merge commit: `adeb8d0a5f153db18daf5f47ad571bf217d80a9d`.
- This is the bounded ledger-only follow-up recording the merge and verification facts. After committing and pushing this follow-up, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending the ledger again solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T02:48:31Z — fail-close direct Live USB scripts on unverifiable media

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `f1f6d45322324d892910493c8e6a4d6fde275ded`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune && git fetch origin main --prune && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune`: fetched cleanly with no upstream advancement.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `101`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `285`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Initial focused verification before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/hermes_cli/test_banner.py tests/tools/test_process_registry.py` -> `402 tests passed, 0 failed`.
- Initial `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- Initial `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.

**Changed files**

- `live-usb/write_usb.sh`: added direct-script canonical removable-disk validation before destructive writes. The target now must resolve via `readlink -f --` to a canonical `/dev/...` path, be a block-device whole disk (`lsblk TYPE=disk`), and have `/sys/class/block/<device>/removable` exactly `1`; the previous warning-only non-removable path was removed.
- `live-usb/provision.sh`: added the same canonical whole-disk removable-media validation before mounting/provisioning; added `_partition_path` so digit-ending disk names such as NVMe/MMC use `p3` partition syntax.
- `tests/cyber/test_live_usb_docs.py`: added direct-script fail-closed safety invariants.
- `README.md` and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`: documented that the direct scripts also reject non-removable, partition, mapper, symlink-only, or unverifiable targets and that root/sudo alone is not sufficient.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Subagent upstream preservation review before edits: `PASS`; required Live USB files tracked, executable modes preserved, no merge/conflict state, no upstream merge needed.
- Subagent Live USB next-gap review before edits: `REQUEST_CHANGES`; direct `live-usb/write_usb.sh` only warned on non-removable media and `live-usb/provision.sh` lacked removable/canonical guards, so direct manual paths could bypass Python wrapper checks.
- `bash -n live-usb/write_usb.sh` and `bash -n live-usb/provision.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `58 passed in 0.71s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/hermes_cli/test_banner.py tests/tools/test_process_registry.py` -> `403 tests passed, 0 failed`.
- `git diff --check && git diff --cached --check` -> passed with no output.
- Subagent spec re-review after fix: `PASS`.
- Subagent quality review after fix: `APPROVED`; no critical or important issues. Minor note: future mocked shell/Bats-style tests could reduce brittleness; `shellcheck` was not installed for that reviewer.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped direct-script guardrail and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T03:08:59Z — sync upstream and clarify direct-script root guidance

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `e938b31e24a72a3784c1b2f430ed8201de295bbf`; no `MERGE_HEAD` or unmerged files.
- `git fetch --all --prune --no-tags`: fetched origin/upstream read-only; upstream `main` advanced from `5bf23ff25` to `b0a25980f`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `4`; `upstream/main..HEAD` -> `102`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `a657c20175ffd55cc33b1b58bbf16de53d8dc77e` with `HEAD^2` equal to upstream `b0a25980f89fc42b495d7d6ec17bf879c9b5d5c3`.
- Drift after merge before this follow-up: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `103`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `5`.

**Changed files**

- Upstream merge changed 76 upstream files, primarily removing in-tree Gemini/Antigravity Code Assist/OAuth provider code and tests and updating auth/provider config, local environment blocklist behavior, desktop settings, release docs, provider docs, and related tests.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- `live-usb/write_usb.sh`: non-root destructive path now says root/sudo alone is not sufficient and that the target must canonicalize to a whole removable `/dev` disk with `removable=1`, before the sudo usage hint.
- `live-usb/provision.sh`: expanded the non-root guard to the same explicit root/sudo-alone-is-not-sufficient warning before any canonicalization or mount/provision side effect.
- `tests/cyber/test_live_usb_docs.py`: added a direct-script invariant asserting both scripts include the complete root/sudo-alone-is-not-sufficient guidance.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Initial focused baseline after merge: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/tools/test_local_env_blocklist.py tests/hermes_cli/test_auth_commands.py tests/hermes_cli/test_config.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_provider_catalog.py tests/hermes_cli/test_web_oauth_dispatch.py tests/agent/test_gemini_fast_fallback.py tests/agent/transports/test_chat_completions.py tests/agent/transports/test_codex_app_server_runtime.py` -> `711 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the merge before local edits -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before local edits.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required AgentCyber/Live USB files present/tracked, executable modes preserved for `scripts/agentcyber` and `live-usb/*.sh`, and the latest merge did not touch required Live USB files.
- Read-only Live USB next-gap review before the fix: `REQUEST_CHANGES`; the direct scripts' non-root failures still implied “just sudo it” before explaining root/sudo alone is insufficient.
- RED check after adding the invariant only: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py -q -o addopts= --tb=short` -> failed as expected because the new root/sudo guidance was absent from the scripts.
- After the script guidance fix: `bash -n live-usb/write_usb.sh live-usb/provision.sh live-usb/build_iso.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `58 passed in 0.82s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix using the same 16-file set above -> `711 tests passed, 0 failed`.
- Spec re-review after the fix: `PASS`.
- Quality review after the fix: `APPROVED`; no critical or important issues.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output after the fix.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- Completed upstream merge commit: `a657c20175ffd55cc33b1b58bbf16de53d8dc77e`.
- This scoped direct-script root-guidance follow-up and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T03:25:31Z — sync upstream prompt-compose/reasoning updates and verify Live USB lane

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `e3f62ead9341f48057247f7cdd1bdb18783db1bb`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched cleanly; upstream `main` advanced from `b0a25980f` to `e448b2141`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `3`; `upstream/main..HEAD` -> `104`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `292`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit `cbdd037dcf99a7c789e2d7daa02f35533303369c` with `HEAD^2` equal to upstream `e448b21414b9dece9b74c3281f04ba4f5c79a771`.
- Drift after merge before this follow-up: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `105`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `4`.

**Changed files**

- Upstream merge changed 16 upstream files: `cli.py`, `hermes_cli/cli_commands_mixin.py`, `hermes_cli/commands.py`, `hermes_cli/config.py`, `hermes_cli/main.py`, `hermes_cli/web_server.py`, `tests/gateway/test_reasoning_command.py`, `tests/hermes_cli/test_prompt_compose_command.py`, `tests/hermes_cli/test_reasoning_full_command.py`, `tests/test_tui_gateway_server.py`, `tui_gateway/server.py`, `ui-tui/src/__tests__/createSlashHandler.test.ts`, `ui-tui/src/app/interfaces.ts`, `ui-tui/src/app/slash/commands/core.ts`, `ui-tui/src/app/useMainApp.ts`, and `website/docs/user-guide/features/web-dashboard.md`.
- Required AgentCyber/Live USB files remained present and tracked: `tools/cyber_live_usb.py`, `tests/cyber/test_live_usb_tool.py`, `tests/cyber/test_live_usb_docs.py`, `scripts/agentcyber`, this ledger, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- No Live USB implementation, toolset, README, or runbook behavior was changed because focused verification and read-only reviews found no new safety/test/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/gateway/test_reasoning_command.py tests/hermes_cli/test_prompt_compose_command.py tests/hermes_cli/test_reasoning_full_command.py tests/test_tui_gateway_server.py` -> `617 tests passed, 0 failed`.
- `scripts/agentcyber status --json` summary -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, and secret fields summarized as booleans/status only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required AgentCyber/Live USB files present/tracked, executable modes preserved for `scripts/agentcyber` and `live-usb/*.sh`, and the latest merge did not delete required AgentCyber files.
- Read-only Live USB safety/docs next-gap review: `PASS`; status/list remain safe, build/write/provision remain root/operator-approval gated and fail closed, direct scripts fail closed for non-root/non-removable/unverifiable media, tests avoid real block-device writes, and docs do not imply sudo alone is enough.

**Blockers / boundaries**

- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.
- One status-summary command mistakenly wrote transient evidence to `/tmp/agentcyber-status-current.json` outside the repo-local write boundary; it contained status JSON only, no delete was performed due the no-delete boundary, and subsequent evidence capture avoided external writes.

**Commit / push**

- Completed upstream merge commit: `cbdd037dcf99a7c789e2d7daa02f35533303369c`.
- This bounded ledger-only follow-up records the merge and verification facts. After committing and pushing this follow-up, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T03:43:30Z — remove sudo from read-only Live USB list docs

**Commands / status**

- `git status --short --branch && git remote -v && git branch --show-current`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `146beb3ed1e8d5f3e749efc3bc42177c1204f6e2`; no `MERGE_HEAD` or unmerged files.
- `git fetch --prune origin && git fetch --prune upstream`: fetched origin/upstream read-only; upstream `main` did not advance beyond `e448b21414b9dece9b74c3281f04ba4f5c79a771`.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `106`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `297`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Baseline focused wrapper before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `308 tests passed, 0 failed`.
- Read-only upstream preservation review: `PASS`; required Live USB files present/tracked, executable modes preserved, branch clean, and no upstream drift.
- Read-only Live USB safety/docs next-gap review before the fix: `REQUEST_CHANGES`; `README.md` still showed `sudo live-usb/write_usb.sh --list` for read-only USB discovery, which normalized unnecessary sudo for a safe list action.

**Changed files**

- `README.md`: changed the Step 3 discovery example to `# Find your USB drive (read-only; no sudo required)` and `live-usb/write_usb.sh --list` without `sudo`.
- `tests/cyber/test_live_usb_docs.py`: added a targeted docs invariant rejecting `sudo live-usb/write_usb.sh --list` in the Live USB README section.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- RED check after adding only the invariant: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py -q -o addopts= --tb=short` -> failed as expected because `sudo live-usb/write_usb.sh --list` was still present.
- After README fix: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `58 passed in 0.75s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `308 tests passed, 0 failed`.
- `scripts/agentcyber status --json` summary -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, and git `dirty: true` only because this docs/test lane was in progress.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Spec review after the fix: `PASS`.
- Quality review after the fix: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped README docs/test fix and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T04:01:45Z — verification/no-op Live USB and upstream drift check

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355`.
- `git fetch upstream --prune && git fetch origin --prune`: fetched read-only; no `upstream/main` advancement (`upstream/main` remained `e448b21414b9dece9b74c3281f04ba4f5c79a771`). A non-main upstream branch `hermes/hermes-6735a859` advanced; this lane did not merge it.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `107`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `298`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Merge state check: no `.git/MERGE_HEAD`, no unmerged paths.

**Changed files**

- No Live USB implementation, toolset, README, runbook, or test behavior changed because focused verification and read-only reviews found no new smallest safety/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this verification/no-op run entry.

**Verification**

- Focused wrapper acceptance: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `308 tests passed, 0 failed`.
- `scripts/agentcyber status --json` before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `511f1ae24ceff1b20b294d344c30de0bf5c00e0a`, and secret fields summarized as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files present and tracked; executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required files present/tracked, executable modes preserved, and upstream/main is already an ancestor of HEAD.
- Read-only Live USB safety/docs next-gap review: `PASS`; no worthwhile remaining smallest safety/test/docs gap found.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This bounded ledger-only verification entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T04:16:48Z — verification/no-op Live USB and upstream drift check

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `4c1f649dc2e77b7671210c3817fa844bd6bdde3a`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `108`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `299`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.

**Changed files**

- No Live USB implementation, toolset, README, runbook, or test behavior changed because focused verification and read-only reviews found no new smallest safety/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this verification/no-op run entry.

**Verification**

- Focused wrapper acceptance: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `308 tests passed, 0 failed`.
- `scripts/agentcyber status --json` before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `4c1f649dc2e77b7671210c3817fa844bd6bdde3a`, and secret fields summarized as booleans/presence only.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files present and tracked; executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required files present/tracked, executable modes preserved, `upstream/main` is an ancestor of `HEAD`, and diff checks passed.
- Read-only Live USB safety/docs next-gap review: `PASS`; status/list remain safe, build/write/provision remain root/operator-approval gated and fail closed, direct scripts reject non-removable/unverifiable targets, tests avoid real block-device writes, and no worthwhile remaining smallest safety/test/docs gap was found.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This bounded ledger-only verification entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T04:41:12Z — guard Live USB build output targets

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `d8d0e47ad2ea64450f657ae5d16a5542b52380fa`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `109`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `300`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Baseline focused wrapper before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `308 tests passed, 0 failed`.
- Baseline `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `d8d0e47ad2ea64450f657ae5d16a5542b52380fa`, and secret fields summarized as booleans/presence only.
- Baseline `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.

**Changed files**

- `tools/cyber_live_usb.py`: added a fail-closed build output guard that rejects ISO output targets that are existing block devices or canonicalize under `/dev`, after root/operator approval and before resolving/running `build_iso.sh`.
- `live-usb/build_iso.sh`: added `reject_unsafe_output_target`; it rejects existing block-device outputs and paths that canonicalize under `/dev`, runs after the root gate before dependency/build work, and re-runs immediately before both amd64 and arm64 `xorriso -o "${OUTPUT}"` calls.
- `tests/cyber/test_live_usb_tool.py`: added approved-build regressions for `/dev/...` output and mocked existing block-device output, both proving the tool fails before `_script()`/`_run()`.
- `tests/cyber/test_live_usb_docs.py`: added an invariant for the direct build script output guard and repeated pre-`xorriso` checks.
- `README.md`: documented that direct `build_iso.sh` refuses existing block-device outputs and paths that canonicalize under `/dev`.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `bash -n live-usb/build_iso.sh live-usb/write_usb.sh live-usb/provision.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `61 passed in 0.83s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `311 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the fix before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this build-output guard lane was in progress.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required files present/tracked, executable modes preserved, and `upstream/main` is an ancestor of `HEAD`.
- Read-only Live USB next-gap review before the fix: `REQUEST_CHANGES`; approved `build` could pass `--output /dev/...` or an output alias resolving to `/dev` through to `xorriso` without a specific output-target guard.
- Spec review after the first fix: `PASS`.
- Quality review after the first fix: `REQUEST_CHANGES`; direct script output validation needed to be repeated immediately before each delayed `xorriso -o "${OUTPUT}"` invocation.
- Quality re-review after the follow-up: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped build-output guard and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T05:00:36Z — tighten post-build Live USB write/provision guidance

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `ea4e8a44425b7329b8c5baa26f1d26696038e055`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `110`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `301`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Baseline focused wrapper before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `311 tests passed, 0 failed`.
- Baseline `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `ea4e8a44425b7329b8c5baa26f1d26696038e055`, and secret fields summarized as booleans/presence only.
- Baseline `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.

**Changed files**

- `live-usb/build_iso.sh`: changed the post-build completion output from a terse `sudo ./write_usb.sh` / provision example into an explicit “operator-approved removable media only” next-step block that states root/sudo alone is not sufficient, requires a canonical whole removable `/dev` disk with `removable=1`, and frames write/provision commands as target-verified operator actions.
- `tests/cyber/test_live_usb_docs.py`: added a regression invariant for the post-build completion guidance, requiring the safer wording and rejecting the stale `Write to USB:  sudo ./write_usb.sh` form.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- RED check after adding only the invariant: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py -q -o addopts= --tb=short` -> failed as expected because the new post-build guidance was not present yet.
- `bash -n live-usb/build_iso.sh live-usb/write_usb.sh live-usb/provision.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `62 passed in 0.78s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `312 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the fix before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this guidance lane was in progress.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required files present/tracked, executable modes preserved, and `upstream/main` is an ancestor of `HEAD`.
- Read-only Live USB safety/docs next-gap review before the fix: `REQUEST_CHANGES`; `live-usb/build_iso.sh` post-build output still framed the next write step as `sudo ./write_usb.sh` without nearby operator-approved/removable/root-alone-not-sufficient wording.
- Spec/safety re-review after the fix: `PASS`.
- Quality re-review after the fix: `APPROVED`; no critical or important issues.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped post-build guidance fix and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T05:19:26Z — verification/no-op Live USB and upstream drift check

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `3682979055a5c298cc53e43b1dfec4b5ff87cb4a`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `111`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `302`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `scripts/agentcyber status --json` before ledger edit reported `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `3682979055a5c298cc53e43b1dfec4b5ff87cb4a`, and secret fields summarized as booleans/presence only.
- `scripts/agentcyber hermes tools list` showed `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker scan for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.

**Changed files**

- No Live USB implementation, toolset, README, runbook, or test behavior changed because focused verification and read-only reviews found no new smallest safety/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this verification/no-op run entry.

**Verification**

- Focused wrapper acceptance: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `312 tests passed, 0 failed`.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; no merge/conflict state, required AgentCyber/Live USB files present/tracked, executable modes preserved, `upstream/main` is an ancestor of `HEAD`, and diff hygiene checks passed.
- Read-only Live USB safety/docs next-gap review: `PASS`; `live_usb` remains default-off, `status`/`list_usb` remain safe/read-only, `build`/`write`/`provision` remain root plus exact-operator-approval gated and fail closed, write/provision target checks require canonical removable `/dev` metadata, direct scripts enforce comparable guardrails, and no new smallest safety/docs gap was found.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This bounded ledger-only verification entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the ledger-only commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T05:44:11Z — require whole-disk Live USB targets in agent guard

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `70589680f368e991f00f1e2db9d5c7b5529cbb2f`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `112`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `303`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- Baseline focused wrapper before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `312 tests passed, 0 failed`.
- Baseline `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `70589680f368e991f00f1e2db9d5c7b5529cbb2f`, and secret fields summarized as booleans/presence only.
- Baseline `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned no matches.

**Changed files**

- `tools/cyber_live_usb.py`: added `_linux_block_device_type()` and made `_require_verifiably_removable_block_device()` reject targets unless `lsblk -dn -o TYPE -- <canonical_device>` reports `disk`, before removable-flag checks or script execution. Also updated status/list/schema wording and made `write_dependencies_ready` include `lsblk`.
- `tests/cyber/test_live_usb_tool.py`: added mocked regressions proving approved `write` and `provision` reject non-disk target types (`part`, `lvm`) before `_script()`/`_run()` and before removable-flag reads; added a status readiness regression proving missing `lsblk` keeps `can_write` false.
- `tests/cyber/test_live_usb_docs.py`: updated the README gate invariant for whole removable `/dev` disk metadata.
- `README.md`, `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`, and `toolsets.py`: updated Live USB wording from removable block-device metadata to verified whole removable `/dev` disk metadata.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- RED check from the implementer after adding only the new non-disk regression: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py::TestLiveUsbTool::test_approved_non_disk_target_fails_before_script_and_run -q -o addopts= --tb=short` -> failed as expected with four `AttributeError` failures because `_linux_block_device_type` did not exist yet.
- After implementation: `uv run --frozen python -m pytest tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py -q -o addopts= --tb=short` -> `67 passed in 0.89s`.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py toolsets.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `317 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the fix before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this whole-disk guard lane was in progress.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Spec re-review after the fix: `PASS`.
- Final quality re-review after the status dependency cleanup: `APPROVED`; no critical, important, or minor issues.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped whole-disk guard/status dependency fix and ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T06:01:37Z — sync upstream and verify Live USB lane remains complete

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `540eeafd711148353a563c987a5e8c94069b604f`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only; upstream advanced from `e448b2141` to `5ff11a689`.
- Drift after fetch before merge: `HEAD..upstream/main` -> `10`; `upstream/main..HEAD` -> `113`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `304`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`.
- `git merge --no-ff upstream/main`: merged cleanly with the `ort` strategy; merge commit before ledger edit was `f64756b404cec648bf88ba6cc7ec6814b9f93faa`.
- Drift after merge before ledger edit: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `114`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `11`.
- `scripts/agentcyber status --json` after the merge before ledger edit reported `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `f64756b404cec648bf88ba6cc7ec6814b9f93faa`, and secret fields summarized as booleans/presence only.
- `scripts/agentcyber hermes tools list` showed `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.

**Changed files**

- Upstream merge changed 44 upstream Hermes files, including agent auxiliary/runtime helpers, gateway auth/config/document handling, `cli.py`, Hermes CLI config/container/runtime/web-server paths, platform adapters, web chat title code, website messaging/security docs, and related upstream tests.
- No Live USB implementation, toolset, README, runbook, or test behavior changed because focused verification and read-only reviews found no new smallest safety/docs gap.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this upstream-sync verification entry.

**Verification**

- Focused wrapper acceptance after the upstream merge: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py tests/agent/test_auxiliary_client.py tests/gateway/test_config.py tests/gateway/test_discord_document_handling.py tests/gateway/test_document_cache.py tests/gateway/test_telegram_documents.py tests/gateway/test_telegram_group_gating.py tests/gateway/test_unauthorized_dm_behavior.py tests/hermes_cli/test_config.py tests/hermes_cli/test_container_boot.py tests/hermes_cli/test_ctrlg_editor_submit.py tests/hermes_cli/test_nous_auth_keepalive.py tests/hermes_cli/test_runtime_provider_resolution.py tests/hermes_cli/test_timestamps_command.py tests/run_agent/test_provider_parity.py tests/test_tui_gateway_server.py tests/tools/test_terminal_config_env_sync.py` -> `1463 tests passed, 0 failed`.
- Conflict marker scan for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before the ledger edit.
- `bash -n live-usb/build_iso.sh live-usb/write_usb.sh live-usb/provision.sh` -> passed.
- `uv run --frozen python -m ruff check tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py tests/cyber/test_live_usb_docs.py toolsets.py hermes_cli/tools_config.py` -> `All checks passed!`.
- Read-only upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked/readable, executable modes preserved, and the worktree/index were clean after merge.
- Read-only Live USB safety/docs next-gap review: `PASS`; `live_usb` remains default-off, `status`/`list_usb` remain safe/read-only, `build`/`write`/`provision` remain root plus exact-operator-approval gated and fail closed, write/provision target checks require canonical whole removable `/dev` disk metadata, direct scripts enforce comparable guardrails, and no new smallest safety/docs gap was found.

**Blockers / boundaries**

- Upstream drift was present and was merged cleanly on the existing guarded sync branch; no conflict recovery was needed.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped upstream merge plus ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T06:25:38Z — fix Live USB provisioning of standalone AgentCyber config dirs

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `931e8caa60021f152f51009900d9541281eb7fb1`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `115`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `316`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `upstream/main` is an ancestor of `HEAD`.
- Baseline focused wrapper before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `317 tests passed, 0 failed`.
- Baseline `scripts/agentcyber status --json` -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `931e8caa60021f152f51009900d9541281eb7fb1`, and secret fields summarized as booleans/presence only.
- Baseline `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker scan for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.

**Changed files**

- `live-usb/provision.sh`: changed config-directory provisioning to stage source contents under a top-level `.hermes/` before writing `hermes-config.tar.gz`, so provisioning from `.agentcyber-home` populates `/home/hermes/.hermes` on first boot. Added `TMP_CFG` cleanup in the existing exit trap and cleared it after successful explicit cleanup to avoid retaining copied config/secrets on failure paths.
- `README.md`: documented that config directories such as `.agentcyber-home` are repacked under `.hermes`, while prebuilt tarballs must already contain a top-level `.hermes/` directory.
- `tests/cyber/test_live_usb_docs.py`: added an invariant for the provision config-dir archive layout, README guidance, and temporary staging cleanup.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- `bash -n live-usb/build_iso.sh live-usb/write_usb.sh live-usb/provision.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `68 passed in 0.87s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `318 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the fix before ledger edit -> `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this config-dir provisioning lane was in progress.
- `scripts/agentcyber hermes tools list` -> `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Read-only upstream preservation review: `PASS`; required AgentCyber/Live USB files present/tracked, executable modes preserved, local and remote guarded branch tips matched before edits, and upstream/main is already an ancestor of HEAD.
- Read-only Live USB safety/docs next-gap review before the fix: `REQUEST_CHANGES`; `README.md` recommended `config=".agentcyber-home"`, but `live-usb/provision.sh` archived config directories using their original basename while first boot expects `/home/hermes/.hermes/config.yaml`.
- Spec re-review after the fix and cleanup follow-up: `PASS`.
- Quality re-review after the cleanup follow-up: `APPROVED`; no critical or important issues.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted by this cron run.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped provision config-dir fix plus ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T07:14:59Z — recover and verify direct Live USB approval-gate changes

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `c98ce7c3d7c022f3f16921247af1af220c7d1c2b` with a dirty worktree from an already-scoped Live USB recovery/follow-up lane; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `116`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `317`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `upstream/main` is an ancestor of `HEAD`.
- `scripts/agentcyber status --json` before ledger edit summarized: `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this direct-approval recovery lane was in progress, head `c98ce7c3d7c022f3f16921247af1af220c7d1c2b`, and secret fields summarized as booleans/presence only.
- `scripts/agentcyber hermes tools list` showed `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker scan for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.

**Changed files**

- `live-usb/build_iso.sh`, `live-usb/write_usb.sh`, and `live-usb/provision.sh`: added exact direct-script operator approval gates using `HERMES_AGENTCYBER_LIVE_USB_APPROVAL`, `--operator-approval`, `--operator-approval-stdin`, or an interactive silent prompt; the scripts clear approval material after a successful gate and before long-running/destructive work. `write_usb.sh --list` remains approval-free/read-only.
- `tools/cyber_live_usb.py`: changed the Python tool wrapper to keep its own exact operator approval check and pass the already-approved value to direct scripts through stdin with `--operator-approval-stdin`, not argv/logs.
- `README.md` and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md`: documented direct-script operator approval, token handling, silent prompt use, default-off `live_usb`, and that root/sudo alone is not sufficient for build/write/provision.
- `tests/cyber/test_live_usb_docs.py` and `tests/cyber/test_live_usb_tool.py`: added direct-script approval-gate, no-token-disclosure, stdin-approval, and mocked positive/negative path coverage without touching real USB/block devices.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this recovery/verification entry.

**Verification**

- `bash -n live-usb/build_iso.sh live-usb/write_usb.sh live-usb/provision.sh` -> passed.
- `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `72 passed in 1.01s`.
- `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tools/cyber_live_usb.py tests/cyber/test_live_usb_tool.py` -> `All checks passed!`.
- Focused wrapper acceptance: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `322 tests passed, 0 failed`.
- Direct non-root token-leak probe with a dummy approval value for `build_iso.sh`, `write_usb.sh`, and `provision.sh` -> no dummy approval token appeared in the emitted errors.
- `git diff --check && git diff --cached --check` -> passed with no output before the ledger edit.
- Read-only spec/safety review: `PASS`; `status`/`list_usb` remain read-only, build/write/provision are root plus exact-operator-approval gated, direct scripts enforce comparable gates, and token material is not logged or echoed by the reviewed paths.
- Read-only quality review: `APPROVED`; wrappers pass approval via stdin after their own gate, direct scripts clear approval material before long-running/destructive children, docs/errors do not imply root alone is sufficient, and tests cover negative/positive paths without real device writes.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted by this cron run.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped direct Live USB approval-gate recovery plus ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.

### 2026-06-22T07:32:17Z — isolate Live USB list tests from host block-device metadata

**Commands / status**

- Read this ledger and `docs/AGENTCYBER_STANDALONE_RUNBOOK.md` before acting.
- `git status --short --branch && git remote -v && git branch --show-current && git rev-parse HEAD`: started clean on `agentcyber/upstream-sync-20260621-194355...origin/agentcyber/upstream-sync-20260621-194355` at `f805d283eaa29574666d52efd2238fa84e0d4d83`; no `MERGE_HEAD` or unmerged files.
- `git fetch upstream main --prune --no-tags && git fetch origin main --prune --no-tags && git fetch origin agentcyber/upstream-sync-20260621-194355 --prune --no-tags`: fetched read-only.
- Drift after fetch: `HEAD..upstream/main` -> `0`; `upstream/main..HEAD` -> `117`; `HEAD..origin/main` -> `0`; `origin/main..HEAD` -> `318`; `HEAD..origin/agentcyber/upstream-sync-20260621-194355` -> `0`; `origin/agentcyber/upstream-sync-20260621-194355..HEAD` -> `0`; `upstream/main` is an ancestor of `HEAD`.
- Baseline `scripts/agentcyber status --json` before edits reported `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: false`, head `f805d283eaa29574666d52efd2238fa84e0d4d83`, and secret fields summarized as booleans/presence only.
- Baseline `scripts/agentcyber hermes tools list` showed `cyber` enabled and `live_usb` disabled.
- Required AgentCyber/Live USB files were tracked with executable modes preserved for `scripts/agentcyber` and `live-usb/{build_iso.sh,write_usb.sh,provision.sh}`.
- Conflict marker search for lines starting `<<<<<<< ` or `>>>>>>> ` returned `0` matches.

**Changed files**

- `tests/cyber/test_live_usb_tool.py`: changed `test_list_usb_returns_removable_devices` to monkeypatch `shutil.which` and `subprocess.run`, feed fixture `lsblk --json` output, and assert the expected command/timeout and parsed removable device result instead of allowing the test to read host block-device metadata.
- `tests/cyber/test_live_usb_docs.py`: added `_list_usb_stub_env()` with temp `lsblk` and `column` stubs, and used it for the direct `write_usb.sh --list` assertion so that read-only script behavior is tested without invoking the host's real `lsblk`/device listing.
- `docs/AGENTCYBER_LIVE_USB_UPSTREAM_LEDGER.md`: added this run entry.

**Verification**

- Baseline focused wrapper acceptance before edits: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `322 tests passed, 0 failed`.
- Read-only upstream preservation review before the fix: `PASS`; no merge/conflict state, required AgentCyber/Live USB paths present/tracked, executable modes preserved, `upstream/main` is an ancestor of `HEAD`, and guarded branch matched origin.
- Read-only Live USB safety/docs next-gap review before the fix: `REQUEST_CHANGES`; two tests still reached real host block-device listing (`tests/cyber/test_live_usb_tool.py::test_list_usb_returns_removable_devices` via `_handle({"action":"list_usb"})`, and `tests/cyber/test_live_usb_docs.py::test_direct_live_usb_approval_gate_behavior_is_fail_closed` via `live-usb/write_usb.sh --list`).
- Focused unit/docs tests after the fix: `uv run --frozen python -m pytest tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py -q -o addopts= --tb=short` -> `72 passed in 1.02s`.
- Focused lint after the fix: `uv run --frozen python -m ruff check tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tools/cyber_live_usb.py` -> `All checks passed!`.
- Read-only re-review after the fix: `PASS`; the two requested no-real-USB/block-device-listing issues were resolved and no critical/important regression was found.
- Focused wrapper acceptance after the fix: `scripts/run_tests.sh tests/cyber/test_live_usb_docs.py tests/cyber/test_live_usb_tool.py tests/hermes_cli/test_tools_config.py tests/hermes_cli/test_agentcyber_cmd.py tests/hermes_cli/test_agentcyber_wrapper.py tests/agent/test_redact.py tests/gateway/test_cyber_audit_hook.py` -> `322 tests passed, 0 failed`.
- `scripts/agentcyber status --json` after the fix before ledger edit reported `live_usb_visible: true`, `live_usb_enabled: false`, `cyber_enabled: true`, local runtime health `ok: true`, git `dirty: true` only because this test-isolation lane was in progress.
- `scripts/agentcyber hermes tools list` after the fix showed `cyber` enabled and `live_usb` disabled.
- `git diff --check && git diff --cached --check && git diff --check HEAD~1..HEAD` -> passed with no output before the ledger edit.

**Blockers / boundaries**

- No upstream drift on `upstream/main` was present, so no upstream merge was needed this run.
- No cron jobs were scheduled, created, updated, paused, resumed, or removed.
- No default `~/.hermes`, default gateway, default cron, or default profiles were modified.
- No files were deleted by this cron run.
- No USB/block-device writes, ISO builds as root, `sudo`, package installs, hardware actions, external security actions, cloud spend, credential access/disclosure, or public disclosure were performed.
- Status commands contacted only the configured local Ollama health endpoint and printed booleans/status fields, not secrets.

**Commit / push**

- This scoped Live USB test-isolation fix plus ledger entry should be committed and pushed to `origin/agentcyber/upstream-sync-20260621-194355` without force. After pushing, final verification should check local `HEAD` equals the remote sync branch tip and stop rather than amending this ledger solely to mention the commit SHA.

**Next lane**

- Open/review/merge the guarded sync branch into AgentCyber main only after human approval; do not force-push.
- Future runs should re-check upstream drift, focused Live USB tests, toolset/status visibility, and this ledger. If no upstream drift or new Live USB gap is found, continue treating the lane as verification/no-op.
