# HQ Priority A bugfix stack detailed review — 2026-07-05

## Scope and evidence

- Repo: `C:/Users/82109/AppData/Local/hermes/hermes-agent`
- Branch: `main`
- HEAD: `8b97f5e51`
- origin/main: `7fde19afc`
- Source inventory: `reports/hq_local_ahead_74_commit_inventory_20260705.json`
- Parent classification: `reports/hq_local_ahead_74_commit_classification_20260705.md`
- This review is read-only plus report generation. No merge/rebase/reset/cherry-pick/branch creation was performed.

```text
## main...origin/main [ahead 74, behind 84]
?? reports/hq_hermes_pre_update_audit_20260705_043838.md
?? reports/hq_local_ahead_74_commit_classification_20260705.md
?? reports/hq_local_ahead_74_commit_inventory_20260705.json
git rev-list --left-right --count origin/main...HEAD = 84	74
git diff --shortstat origin/main..HEAD = 240 files changed, 11456 insertions(+), 6063 deletions(-)
```

## Priority A stack summary

- Priority A bugfix commits identified: `33`
- `A4 Windows Gateway restart hardening`: 1 commits, 2 touched paths
- `A5 Desktop/TUI websocket stall diagnostics`: 1 commits, 8 touched paths
- `A2 Runtime event/result boundary hardening`: 4 commits, 15 touched paths
- `A1 Clarify approval/choice UX correctness`: 23 commits, 29 touched paths
- `A3 Desktop startup and prompt-submit resilience`: 4 commits, 10 touched paths

## Recommended clean-candidate port order

1. **A4 Windows Gateway restart hardening** — contained operational reliability fix.
2. **A5 Desktop/TUI websocket stall diagnostics** — supports diagnosing live Desktop/TUI stalls before larger UI changes.
3. **A2 Runtime event/result boundary hardening** — prevents tool-result/event loss before clarify UI depends on that path.
4. **A1 Clarify approval/choice UX correctness** — highest user-facing value, but broad; port as one coherent stack, not repeated raw commits.
5. **A3 Desktop startup and prompt-submit resilience** — polish/resilience layer after core event/clarify paths are stable.

## Stack details

### A4 Windows Gateway restart hardening

- Risk/value: **High value / low-medium risk**
- Why: HQ Windows operational reliability; contained to gateway restart path plus tests.
- Porting recommendation:
  - Port early because HQ runs on Windows and restart reliability is operationally important.

Commits:
- `c1710d52e` 2026-06-28 — fix(gateway): harden Windows restart path

Net-diff paths still changed vs `origin/main`:
- `M` `gateway/run.py`
- `M` `tests/gateway/test_restart_drain.py`

Focused verification commands after porting this stack:
```bash
python -m pytest tests/gateway/test_restart_drain.py -v --tb=short -n 0
```

### A5 Desktop/TUI websocket stall diagnostics

- Risk/value: **High value / medium risk**
- Why: Useful for diagnosing live-loop stalls; touches web server/TUI gateway concurrency paths.
- Porting recommendation:
  - Port early if current Desktop/TUI sessions still show stalls; otherwise keep as diagnostics stack with tests.

Commits:
- `2875dad0c` 2026-07-03 — fix: harden desktop ws loop stall diagnostics

Net-diff paths still changed vs `origin/main`:
- `M` `hermes_cli/web_server.py`
- `M` `tests/test_tui_gateway_ws.py`
- `M` `tests/tui_gateway/test_inline_rpc_gil_starvation.py`
- `A` `tests/tui_gateway/test_loop_monitor.py`
- `A` `tui_gateway/loop_monitor.py`
- `M` `tui_gateway/server.py`
- `M` `tui_gateway/ws.py`
Touched by commits but not currently net-diffing vs `origin/main` (possible superseded/canceled/test-only movement):
- `tests/test_web_server.py`

Focused verification commands after porting this stack:
```bash
python -m pytest tests/test_tui_gateway_ws.py tests/test_web_server.py tests/tui_gateway/test_inline_rpc_gil_starvation.py tests/tui_gateway/test_loop_monitor.py -v --tb=short -n 0
```

### A2 Runtime event/result boundary hardening

- Risk/value: **High value / medium risk**
- Why: Prevents lost/delayed tool-result confusion; interacts with agent executor and Desktop event stream.
- Porting recommendation:
  - Port before A1 so Desktop/Gateway can reliably deliver tool-result events that clarify depends on.

Commits:
- `1492c1718` 2026-06-30 — fix(desktop): distinguish lost gateway and bound tool result events duplicate-subject
- `18e8f4465` 2026-06-30 — fix(desktop): bound live terminal output events duplicate-subject
- `307924c72` 2026-06-30 — fix(desktop): distinguish lost gateway and bound tool result events duplicate-subject
- `6527445b1` 2026-06-30 — fix(desktop): bound live terminal output events duplicate-subject

Net-diff paths still changed vs `origin/main`:
- `M` `agent/tool_executor.py`
- `M` `apps/desktop/src/app/gateway/hooks/use-gateway-boot.ts`
- `M` `apps/desktop/src/components/boot-failure-overlay.tsx`
- `M` `apps/desktop/src/components/gateway-connecting-overlay.test.tsx`
- `M` `apps/desktop/src/i18n/en.ts`
- `M` `apps/desktop/src/i18n/ja.ts`
- `M` `apps/desktop/src/i18n/types.ts`
- `M` `apps/desktop/src/i18n/zh-hant.ts`
- `M` `apps/desktop/src/i18n/zh.ts`
- `M` `apps/desktop/src/store/boot.ts`
- `M` `tests/run_agent/test_tool_call_incremental_persistence.py`
- `M` `tests/test_tui_gateway_server.py`
- `M` `tui_gateway/server.py`
Touched by commits but not currently net-diffing vs `origin/main` (possible superseded/canceled/test-only movement):
- `apps/desktop/electron/titlebar-overlay-width.cjs`
- `apps/desktop/src/app/session/hooks/use-message-stream.ts`

Focused verification commands after porting this stack:
```bash
python -m pytest tests/test_tui_gateway_server.py tests/run_agent/test_run_agent.py -v --tb=short -n 0
cd apps/desktop && npm run test:ui -- src/app/gateway/hooks/use-gateway-boot.test.tsx src/components/gateway-connecting-overlay.test.tsx
```

### A1 Clarify approval/choice UX correctness

- Risk/value: **High value / medium risk**
- Why: Touches tool schema, Desktop UI, TUI/Gateway callback semantics, and messaging adapters. Needs all adapters/tests together to avoid partial UX regressions.
- Porting recommendation:
  - Port final net behavior as a single coherent clarify stack. Do not cherry-pick every duplicate/restaged clarify commit one by one.
  - Keep backend constraints, Desktop staged multi-select UI, adapter buttons, stale-request cleanup, timeout/keyboard behavior, and tests synchronized.

Commits:
- `cf5cf315a` 2026-06-29 — fix(desktop): tolerate delayed clarify responses duplicate-subject
- `b7c7d78e7` 2026-06-29 — feat(clarify): restore native multi-select prompts duplicate-subject
- `336352acd` 2026-06-29 — fix(clarify): harden constrained-choice semantics duplicate-subject
- `3f17156b7` 2026-06-29 — fix(clarify): keep multi-select choice UX visible duplicate-subject
- `f2f675a46` 2026-06-29 — fix(clarify): enforce multi-select response bounds duplicate-subject
- `cb0410001` 2026-06-29 — fix(clarify): reject empty required multi-select replies duplicate-subject
- `dba26fbac` 2026-06-29 — fix(desktop): tolerate delayed clarify responses duplicate-subject
- `b3123618c` 2026-06-29 — feat(clarify): restore native multi-select prompts duplicate-subject
- `13c8445f4` 2026-06-29 — fix(clarify): harden constrained-choice semantics duplicate-subject
- `45e737a75` 2026-06-29 — fix(clarify): keep multi-select choice UX visible duplicate-subject
- `e78c7f85b` 2026-06-29 — fix(clarify): enforce multi-select response bounds duplicate-subject
- `fa522dae9` 2026-06-29 — fix(clarify): reject empty required multi-select replies duplicate-subject
- `d4ba30a27` 2026-07-01 — fix(clarify): enforce constrained selection responses duplicate-subject
- `35aa9f395` 2026-07-01 — fix(clarify): reject mixed constrained responses duplicate-subject
- `2aa38fba5` 2026-07-02 — fix(clarify): cover task report scope choices duplicate-subject
- `61505325e` 2026-06-29 — fix(desktop): tolerate delayed clarify responses duplicate-subject
- `e8368cfc7` 2026-07-01 — fix(clarify): enforce constrained selection responses duplicate-subject
- `6f3029374` 2026-07-01 — fix(clarify): reject mixed constrained responses duplicate-subject
- `ff795c13b` 2026-07-02 — fix(clarify): cover task report scope choices duplicate-subject
- `9cbdedeff` 2026-07-02 — fix(clarify): honor configured Desktop clarify timeout
- `51150b9c7` 2026-07-02 — fix(clarify): resolve latest-origin Desktop integration
- `4607d1758` 2026-07-02 — fix(desktop): expire stale clarify requests
- `d6c19f820` 2026-07-04 — fix(desktop): restore clarify keyboard shortcuts

Net-diff paths still changed vs `origin/main`:
- `M` `agent/agent_runtime_helpers.py`
- `M` `agent/tool_executor.py`
- `M` `apps/desktop/electron/update-relaunch.cjs`
- `M` `apps/desktop/electron/update-relaunch.test.cjs`
- `A` `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event.test.tsx`
- `M` `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event.ts`
- `A` `apps/desktop/src/components/assistant-ui/clarify-tool.test.tsx`
- `M` `apps/desktop/src/components/assistant-ui/clarify-tool.tsx`
- `M` `apps/desktop/src/i18n/en.ts`
- `M` `apps/desktop/src/i18n/ja.ts`
- `M` `apps/desktop/src/i18n/types.ts`
- `M` `apps/desktop/src/i18n/zh-hant.ts`
- `M` `apps/desktop/src/i18n/zh.ts`
- `M` `apps/desktop/src/store/clarify.ts`
- `M` `gateway/platforms/base.py`
- `M` `gateway/platforms/whatsapp_cloud.py`
- `M` `gateway/run.py`
- `M` `plugins/platforms/discord/adapter.py`
- `M` `plugins/platforms/telegram/adapter.py`
- `M` `tests/gateway/test_discord_clarify_buttons.py`
- `M` `tests/gateway/test_telegram_clarify_buttons.py`
- `M` `tests/gateway/test_whatsapp_cloud.py`
- `M` `tests/run_agent/test_run_agent.py`
- `M` `tests/test_tui_gateway_server.py`
- `M` `tests/tools/test_clarify_gateway.py`
- `M` `tests/tools/test_clarify_tool.py`
- `M` `tools/clarify_gateway.py`
- `M` `tools/clarify_tool.py`
- `M` `tui_gateway/server.py`

Focused verification commands after porting this stack:
```bash
python -m pytest tests/tools/test_clarify_tool.py tests/tools/test_clarify_gateway.py tests/test_tui_gateway_server.py tests/run_agent/test_run_agent.py -v --tb=short -n 0
python -m pytest tests/gateway/test_telegram_clarify_buttons.py tests/gateway/test_discord_clarify_buttons.py -v --tb=short -n 0
cd apps/desktop && npm run test:ui -- src/components/assistant-ui/clarify-tool.test.tsx src/app/session/hooks/use-message-stream/gateway-event.test.tsx
cd apps/desktop && npm run typecheck
```

### A3 Desktop startup and prompt-submit resilience

- Risk/value: **Medium value / low-medium risk**
- Why: Mostly UI resilience/copy and startup readiness handling; verify Desktop focused tests.
- Porting recommendation:
  - Port after core event/clarify stacks; it is mostly resilience/polish but may conflict with current Desktop prompt code.

Commits:
- `2bd044507` 2026-06-29 — fix(desktop): explain prompt submit timeouts duplicate-subject
- `33f588e87` 2026-06-29 — fix(desktop): tolerate slow backend readiness during startup duplicate-subject
- `3051d4f0d` 2026-06-29 — fix(desktop): explain prompt submit timeouts duplicate-subject
- `2e7d55643` 2026-06-29 — fix(desktop): tolerate slow backend readiness during startup duplicate-subject

Net-diff paths still changed vs `origin/main`:
- `M` `apps/desktop/electron/backend-ready.test.cjs`
- `M` `apps/desktop/electron/main.cjs`
- `M` `apps/desktop/src/app/session/hooks/use-prompt-actions/index.test.tsx`
- `M` `apps/desktop/src/app/session/hooks/use-prompt-actions/submit.ts`
- `M` `apps/desktop/src/app/session/hooks/use-prompt-actions/utils.ts`
- `M` `apps/desktop/src/i18n/en.ts`
- `M` `apps/desktop/src/i18n/ja.ts`
- `M` `apps/desktop/src/i18n/types.ts`
- `M` `apps/desktop/src/i18n/zh-hant.ts`
- `M` `apps/desktop/src/i18n/zh.ts`

Focused verification commands after porting this stack:
```bash
cd apps/desktop && npm run test:ui -- src/app/session/hooks/use-prompt-actions/index.test.tsx
cd apps/desktop && npm test -- electron/backend-ready.test.cjs
```

## Cross-stack conflict risks

- A1 and A2 both touch `agent/tool_executor.py`, Desktop event-stream code, and TUI/Gateway callback paths; port A2 first, then A1.
- Multiple duplicate clarify subjects exist, so a raw cherry-pick sequence risks reintroducing older intermediate states. Use final net diff and tests as the source of truth.
- Desktop tests/build may require current `apps/desktop` dependencies; verify in the target clean candidate environment, not only this divergent branch.
- Gateway adapter changes for Telegram/Discord should be ported with tests together; partial porting can make `clarify` work in Desktop but fail on Gateway surfaces.

## Suggested next step

Prepare a clean-candidate plan that starts from `origin/main`, creates a preservation tag for current `main`, then ports Priority A in the order above with focused tests after each stack. Do not push or merge until the candidate has been verified.

## Verification note for this report

This report itself is a Markdown audit artifact. Code tests were not run during report generation because no product code was modified in this step; artifact consistency checks should verify the report file and that referenced commits exist.
