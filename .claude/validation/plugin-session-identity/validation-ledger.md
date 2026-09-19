# Plugin Session Identity Validation Ledger

## Scope

- Task: `plugin-session-identity`
- Branch: `hermes/plugin-session-identity`
- Baseline: `e818827191c6ff44493124b76e3b07ddf208d20a`
- Environment: isolated worktree; canonical `scripts/run_tests.sh`; `HERMES_TEST_WORKERS=3`; Python 3.11.16.
- App/browser matrix: not applicable. This is a Python dispatcher change with no UI, deployed environment, user role, locale, database, or network contract.

## Literal RED evidence

Before the production edit, both new distinct-session cases failed for the intended reason:

```text
FAILED tests/hermes_cli/test_plugins.py::TestForceReloadSymmetry::test_concurrent_distinct_session_starts_both_run
E AssertionError: assert ['session-a'] == ['session-a', 'session-b']
WARNING hermes_cli.plugins:plugins_dispatch.py:265 Hook 'on_session_start' callback recorder skipped after previous timeout or while still running

FAILED tests/hermes_cli/test_plugins.py::TestForceReloadSymmetry::test_concurrent_distinct_pre_verify_sessions_both_run
E AssertionError: assert ['verify-a'] == ['verify-a', 'verify-b']
WARNING hermes_cli.plugins:plugins_dispatch.py:265 Hook 'pre_verify' callback recorder skipped after previous timeout or while still running
```

Pre-fix controls in the same implementation state: `4 passed, 0 failed` for same-session dedupe, real-timeout callback-global suppression, distinct tool calls, and turn-scoped session end.

## GREEN evidence

After adding `session_id` as the final identity fallback:

- Focused identity/safety tests: `16 passed, 0 failed`.
- Complete `tests/hermes_cli/test_plugins.py`: `94 passed, 0 failed`.
- Neighboring hook/session suites: `69 passed, 0 failed`.
- Relevant plugin and Honcho suites: `2,538 passed, 0 failed, 23 skipped`.
- Final combined matrix: `2,701 passed, 0 failed, 23 skipped` across 175 files.
- Repeated concurrency subset: `30/30` clean runs.
- Ruff lint: passed.
- Python syntax compilation: passed.
- `git diff --check`: passed.
- Full-file Ruff format check has pre-existing baseline drift in both target files; the changed ranges were formatted without rewriting unrelated code.

## Additional sequential rounds

Each round executed the same complete relevant matrix plus Ruff, syntax compilation, and diff checks. Profiles/locales/data variation is not applicable to this non-app dispatcher unit; each round exercises concurrent unique sessions, same-session duplication, real timeout, reduced and canonical session-end shapes, pre-verify, and tool-hook priority.

| Round | Time (-03) | Revision under test | Matrix | Static checks | Gaps |
|---:|---|---|---|---|---|
| 1 | 2026-09-19 07:47:33 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 2 | 2026-09-19 07:48:41 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 3 | 2026-09-19 07:49:56 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 4 | 2026-09-19 07:51:09 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 5 | 2026-09-19 07:52:10 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 6 | 2026-09-19 07:55:18 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 7 | 2026-09-19 07:56:24 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 8 | 2026-09-19 07:57:28 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 9 | 2026-09-19 07:58:35 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 10 | 2026-09-19 07:59:36 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 11 | 2026-09-19 08:00:46 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 12 | 2026-09-19 08:01:46 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 13 | 2026-09-19 08:02:44 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 14 | 2026-09-19 08:03:43 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |
| 15 | 2026-09-19 08:04:45 -0300 | `e8188271` + working tree | 175 files / 2,701 passed / 0 failed / 23 skipped | Ruff, syntax, diff check passed | No gap |

## Mandatory checklist

| Area | Status | Evidence / justification |
|---|---|---|
| Data and numbers | Not applicable | Dispatcher gate has no business data, calculations, dates, currencies, or UI totals. |
| Internationalization | Not applicable | No user-visible copy or locale behavior changed. |
| RBAC | Not applicable | No authorization surface changed; fail-closed `pre_tool_call` regression remains green. |
| Centralized logging | Approved | No logging payload added; existing redacted hook/callback warning contract unchanged. |
| Design system | Not applicable | No UI. |
| APIs / API Registry | Not applicable | No endpoint or schema change. |
| Hover / context menus | Not applicable | No UI. |
| App Registry / Copilot | Not applicable | No app resource or Copilot contract. |
| Database | Not applicable | No persistence or migration. |
| Product photos | Not applicable | No ecommerce/media surface. |
| Regressions and publication | Locally approved; publication blocked by scope | 2,701-test matrix and 15 additional rounds passed. Push/PR/merge/deploy are explicitly controller-owned. |

## Gap ledger

No confirmed implementation gap remains. Publication onto current `origin/main` is pending controller action, not performed here. Upstream advanced by 11 unrelated commits during validation; neither target file changed upstream.
