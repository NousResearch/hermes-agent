# Capy WebUI visual QA loop

Use this runbook for WebUI/Capy Spaces sprints and any user-visible UI change. It is intentionally lightweight enough to run every sprint.

## Trigger

Run visual QA when any of these change:

- WebUI pages, components, styling, routes, workspace/file browsing, or chat UX.
- Gateway/WebUI integration such as sessions, workspace selection, dashboard data, or status indicators.
- Capy Spaces demos or production Spaces surfaces.

## Loop

1. Start from code/tests.
   - Run targeted unit/API tests for the changed files.
   - Check `git diff --check`.

2. Verify service health.
   - Local WebUI: `curl -fsS http://127.0.0.1:8787/health`
   - Tailnet WebUI when available: `curl -fsS https://capy.tail9c6e3.ts.net/health`

3. Open the user-visible app.
   - Use the assistant browser tools for automated QA.
   - For Brendan-facing local apps, also open the visible macOS browser when requested or when visual review is the deliverable.

4. Browser QA checklist.
   - Load the changed page from a clean state.
   - Capture a screenshot with annotated elements when layout or interaction changed.
   - Exercise the happy path.
   - Exercise one failure/empty state.
   - Check browser console for errors and failed requests.
   - Verify important text and status indicators match backend state.

5. Report.
   - Include the URL tested, actions taken, screenshot path if created, console/API errors, and any remaining blockers.
   - Do not treat mock/demo UI as sufficient for Capy Spaces acceptance unless the user explicitly asks for a mock.

## Workspace-aware checks

When the change touches workspace handling:

- Confirm WebUI workspace selector and gateway `/workspace` agree on the active project path where expected.
- Confirm terminal/file/code/delegation tools resolve relative paths from the selected workspace.
- Confirm `/status` and `/capy` display the session workspace.

## Suggested commands

```bash
cd /Users/bschmidy10/.hermes/hermes-agent
./venv/bin/python3 -m pytest tests/gateway/test_workspace_sessions.py tests/gateway/test_capy_status.py -q
git diff --check
curl -fsS http://127.0.0.1:8787/health
```
