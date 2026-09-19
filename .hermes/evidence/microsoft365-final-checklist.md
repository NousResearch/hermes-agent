# Microsoft 365 Tasks 13–16 acceptance evidence

This checklist records the repository state after the Tasks 13–15 documentation/artifact changes. It deliberately does not modify `MICROSOFT365_PLUGIN_REVIEW_BRIEF.md`, which was a pre-existing untracked user artifact and is outside this work.

## Plan completion criteria

| Criterion | Evidence | Status |
|---|---|---|
| Host-owned `pre_tool_call` approval; no direct plugin approval call | `plugins/microsoft365/__init__.py:register`; `tests/plugins/test_microsoft365_plugin.py:test_pre_tool_call_emits_host_approval_directive_for_enabled_write`; host denial test | verified |
| Unsupported operations blocked before approval/client/token and absent from model schema | `backend.py:operation_support`, `tools.py:_run/schema`, tests for Teams send and empty enum | verified |
| Preflight rejects `user_id=me`, missing fields, unsupported selections | `backend.py:preflight`; `test_preflight_is_not_ready_for_me_or_unsupported_selection` | verified |
| Action enums contain only enabled and supported operations | `__init__.py:supported_operations`; `test_schema_only_contains_enabled_supported_operations` | verified |
| Binary upload/download work under explicit limits | `tools.py:_decode_upload`, `_safe_path`, download branch; `test_microsoft365_tasks_6_12.py` transfer tests | verified |
| Search is implemented as a bounded/query-bearing operation | `_request_configuration` and Teams generated search request; search tests | verified |
| Planner and To Do labels/endpoints are distinct | `backend.py:OPERATIONS`; Planner branch in `tools.py`; distinction test | verified |
| Permission claims have endpoint/auth evidence | `references/graph-permissions.md`, `backend.py:OPERATION_PERMISSIONS`; permission tests | verified locally; tenant consent not tested |
| Secret handling follows a canonical mechanism | `plugin.yaml` secret marker; `tools.py:_settings` scoped-secret fallback; redaction test | verified for runtime/redaction; deployment persistence remains environment-specific |
| Result normalization/redaction is bounded and tested | `backend.py:safe_result`; nested header/secret test | verified |
| Dependency range matches tested SDK floor | `plugin.yaml`, `pyproject.toml`, metadata test; SDK 1.62.0 environment | verified for tested floor; no matrix for other 1.x releases |
| README and review evidence distinguish implemented/unsupported/planned | `plugins/microsoft365/README.md`; this checklist | verified; review brief intentionally untouched per task scope |
| Test evidence is reproducible without hidden selectors | Commands below and captured results | verified |
| Teams Bot Framework / `teams_pipeline` unchanged | `git diff --name-only origin/main...HEAD`; no adapter files in range; existing Teams tests | verified by scope review |
| Final diff receives maintainer/security review | Task 16 review below, diff checks and secret scan | verified locally; no PR update/push performed |

## Documentation requirements (Tasks 13–15)

- README leads with business value and explains the one-plugin rationale.
- The complete operation matrix includes auth mode, permission, approval, status, and notes.
- App-only versus delegated auth, admin consent, preflight semantics, host approval seam, policy-not-sandbox trust model, secret storage, binary limits, Planner/To Do distinction, Teams adapter preservation, and limitations are explicit.
- `assets/architecture.svg` is self-contained, sanitized, uses no external images, and shows administrator flags → registered actions → host approval → Graph client → services, plus the unsupported stop and separate Teams adapter.
- Full requested operation definitions remain in `backend.py`; `teams.send_messages` is explicitly retained and blocked rather than removed.

## Task 16 review notes

- Scope: only Microsoft 365 plugin docs/artifact were added in Tasks 13–15; the pre-existing review brief was not touched.
- No `.env`, config, credentials, tokens, tenant data, network calls, PR updates, or pushes were performed.
- No production-code fix was required by the Task 16 review; therefore no new TDD cycle was needed for code changes.
- The existing implementation uses official Graph SDK builders/models and the declared tested floor `msgraph-sdk>=1.62.0,<2`.
- The README avoids claims that delegated auth, remote consent verification, automatic save/apply preflight, complete Graph coverage, large-file sessions, or encrypted config persistence are implemented.

## Required command evidence

Run from `C:/Users/fabio/hermes-wt-microsoft365`:

```text
py -3.11 -m pytest tests/plugins/test_microsoft365_plugin.py -q
py -3.11 -m pytest tests/plugins/test_microsoft365_tasks_6_12.py -q
py -3.11 -m pytest tests/tools/test_microsoft_graph_client.py tests/tools/test_microsoft_graph_auth.py -q
py -3.11 -m hermes_cli.plugin_validate plugins/microsoft365/plugin.yaml
py -3.11 -m compileall -q plugins/microsoft365
git diff --check origin/main...HEAD
```

Observed on `e7a6a41` (before the evidence-only commit below):

| Command | Exit | Observed result |
|---|---:|---|
| `py -3.11 -m pytest tests/plugins/test_microsoft365_plugin.py -q` | 0 | 21 passed in 2.93s |
| `py -3.11 -m pytest tests/plugins/test_microsoft365_tasks_6_12.py -q` | 0 | 8 passed in 1.61s |
| `py -3.11 -m pytest tests/tools/test_microsoft_graph_client.py tests/tools/test_microsoft_graph_auth.py -q` | 1 | 16 passed, 1 failed, 8 warnings; the known `[trio]` test passes `asyncio.gather` to Trio and raises `TypeError` |
| `py -3.11 -m hermes_cli.plugin_validate plugins/microsoft365/plugin.yaml` | 0 | no output |
| `py -3.11 -m compileall -q plugins/microsoft365` | 0 | no output |
| `git diff --check origin/main...HEAD` | 0 | no whitespace errors |

The same Graph failure was observed in `.hermes/evidence/microsoft365-baseline.md`; that baseline explicitly did not establish a clean-origin comparison, so this remains an unresolved environment/upstream attribution rather than a claim that the failure is pre-existing. It is reported rather than silently filtered. The final evidence commit does not alter production code or test behavior.
