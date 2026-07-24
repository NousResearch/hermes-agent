# HQ Harness Dashboard Adapter Build Packet

Planner source: `cron_331ecf7312ab_20260613_000025`, assistant message `4457`.
Build scope: worktree-only dry-run adapter for `hq_harness_eval_latest.json`.

## Purpose

`HQ Harness Evidence Dashboard Adapter v0.3` reads the existing dry-run harness report and emits a compact status block for Build Packet and Review Dispatch. It is an observability adapter only: it does not change Gateway, cron, tool permissions, MCP scopes, live enforcement, secrets, or ELIOS.

## Inputs

Default report path:

```text
C:/Users/82109/AppData/Local/hermes/reports/hq_harness_eval_latest.json
```

Required report fields:

- `run_id`
- `mode`
- `total_fixtures`
- `passed`
- `failed`
- `failures`
- `decisions`
- `safety_invariants`

Required true safety invariants:

- `synthetic_only`
- `no_live_tool_execution`
- `no_secret_access`
- `no_cron_mutation`
- `no_gateway_runtime_change`

## Status labels

| Status | Meaning | Review action |
|---|---|---|
| `OK` | `dry_run_only`, all fixtures pass, all required safety invariants true | Continue dry-run review; live enforcement still requires approval |
| `REVIEW` | Fixture failure, schema mismatch, missing/unknown field, non-dry-run mode, or empty decisions | Inspect report/fixtures before broadening autonomy |
| `BLOCKED` | Any required safety invariant is false or missing | Stop capability expansion and recover dry-run safety invariants |
| `RECOVER` | Report missing, unreadable, or malformed JSON | Regenerate the dry-run harness report |

## Files prepared

- `scripts/hq_harness_dashboard_adapter.py`
- `tests/scripts/test_hq_harness_dashboard_adapter.py`
- `references/hq-harness-dashboard-adapter-build-packet.md`

## Test plan

```bash
uv run --with pytest --python 'C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe' \
  python -m pytest tests/scripts/test_hq_harness_validator.py tests/scripts/test_hq_harness_dashboard_adapter.py -q -o addopts=''

uv run --with pytest --python 'C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe' \
  python -m py_compile scripts/hq_harness_validator.py scripts/hq_harness_dashboard_adapter.py

uv run --with pytest --python 'C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe' \
  python scripts/hq_harness_dashboard_adapter.py --report 'C:/Users/82109/AppData/Local/hermes/reports/hq_harness_eval_latest.json'
```

Expected baseline from the latest known report: status `OK`, mode `dry_run_only`, fixture pass `15/15`, failed count `0`, all required safety invariants true.

## Safety classification

Auto-allowed under HQ Supervised Auto-Approve Policy v1:

- Worktree-only branch code edit with tests and rollback notes.
- Read-only local report inspection.
- Documentation-only Build Packet artifact.
- Deterministic local pytest/py_compile/dry-run execution.

Still approval-gated:

- Integrating this adapter into live `hq_health_dashboard.py` outside the worktree.
- Connecting status to Gateway pre-dispatch enforcement or tool-call blocking.
- Changing cron schedules, `context_from`, delivery targets, or permissions.
- Adding MCP write scopes or raw secret access.

## Rollback

Because this is branch/worktree-only, rollback is:

```bash
git restore --staged scripts/hq_harness_dashboard_adapter.py tests/scripts/test_hq_harness_dashboard_adapter.py references/hq-harness-dashboard-adapter-build-packet.md
git restore scripts/hq_harness_dashboard_adapter.py tests/scripts/test_hq_harness_dashboard_adapter.py references/hq-harness-dashboard-adapter-build-packet.md
# if files are untracked:
# remove only these three files after preserving any desired patch externally
```

Do not use destructive repository resets unless Kihoon explicitly approves.

## Review Dispatch input template

```text
- Planner focus: Harness Evidence Dashboard Adapter v0.3
- Worktree: C:/Users/82109/AppData/Local/hermes/worktrees/hq-harness-validator-v0
- Files changed: scripts/hq_harness_dashboard_adapter.py; tests/scripts/test_hq_harness_dashboard_adapter.py; references/hq-harness-dashboard-adapter-build-packet.md
- Current harness baseline: dry_run_only, 15/15 pass, failed=0
- Adapter dry-run status: <fill from command output>
- Tests: <fill from pytest/py_compile>
- Approval-needed: live dashboard integration and any runtime enforcement remain approval-gated
- Rollback: restore/remove the three prepared files in the worktree; no Gateway/cron mutation occurred
```
