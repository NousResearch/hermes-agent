# HQ Memory Quality Gate Review Prep

Generated: 2026-06-23 KST
Scope: review-prep artifact for approved `clean-worktree 및 quality gate 정리` cleanup.
Safety: reference document only; no live enforcement changes.

## 1. Review objective

Prepare the `hq-harness-validator-v0` worktree for a clean, auditable review pass by grouping the current quality-gate work into explicit review themes, tests, and rollback paths.

## 2. Current change groups

| Group | Files | Purpose | Risk |
|---|---|---|---|
| Eval gate scaffold | `agent/eval_gate.py`, `tests/agent/test_eval_gate.py` | Deterministic allow/confirm/reject/skip gate decisions. | Low while disabled/audit-only by default. |
| Cron pre-dispatch hook | `cron/scheduler.py`, `tests/cron/test_eval_gate_pre_dispatch.py` | Evaluate cron jobs before dispatch; fail open except explicit configured skip. | Medium; dispatch boundary code, but default inert. |
| Gateway pre-dispatch hook | `gateway/run.py`, `tests/gateway/test_pre_gateway_dispatch.py` | Evaluate user-originated gateway events before auth/session start. | Medium; dispatch boundary code, but default inert. |
| Config/plugin registration | `hermes_cli/config.py`, `hermes_cli/plugins.py` | Add `eval_gate` config and `pre_cron_dispatch` hook name. | Low; default disabled. |
| Harness fixtures | `scripts/hq_harness_validator.py`, `tests/scripts/test_hq_harness_validator.py` | Add policy/red-team fixtures and validator coverage. | Low/medium; local validation only. |
| Reference docs/reports | `references/*`, `reports/*` | Preserve planner/build evidence and rollout rationale. | Low; docs only. |

## 3. Quality checks performed

Command used after installing missing local test plugins (`pytest-asyncio`, `pytest-timeout`) into the Hermes venv:

```bash
cd C:/Users/82109/AppData/Local/hermes/worktrees/hq-harness-validator-v0
export PYTHONPATH="$PWD"
python -m pytest \
  tests/agent/test_eval_gate.py \
  tests/cron/test_eval_gate_pre_dispatch.py \
  tests/gateway/test_pre_gateway_dispatch.py \
  tests/scripts/test_hq_harness_validator.py \
  -q -o 'addopts=' --tb=short
```

Observed result:

```text
21 passed in 1.34s
```

## 4. Review checklist

- [x] Default config is disabled/audit-only: `eval_gate.enabled=false`, `gateway_enabled=false`, `cron_enabled=false`, `audit_only=true`, `enforce=false`.
- [x] Cron hook returns a BLOCKED doc only when deterministic gate returns `skip`.
- [x] Gateway hook short-circuits only on `skip`; exceptions fail open with warning.
- [x] Plugin hook registry includes `pre_cron_dispatch` without removing existing hooks.
- [x] Harness fixtures include approval-required and explicit-approval cases.
- [x] Tests cover allow/confirm/reject and pre-dispatch skip behavior.
- [x] No main/master merge performed.
- [x] No Gateway restart performed.
- [x] No live enforcement enabled.

## 5. Remaining reviewer questions

1. Should `eval_gate` stay in core `agent/` or move under a plugin-specific module before PR?
2. Should cron BLOCKED output count as `last_status=failed` or a distinct blocked status in future schema work?
3. Should gateway pre-dispatch run before or after pairing/auth for all platforms? Current branch keeps it before auth to evaluate dispatch boundary.
4. Should synthetic red-team fixtures be separated from policy fixtures in output JSON for clearer dashboards?

## 6. Rollback plan

Rollback is branch-local:

```bash
git revert <cleanup-commit>
```

If only live-dispatch hook integration needs rollback, revert these files first:

```text
cron/scheduler.py
gateway/run.py
hermes_cli/config.py
hermes_cli/plugins.py
agent/eval_gate.py
```

Docs/reports can be removed independently if they are too noisy for the final PR.

## 7. Approval boundary reminder

The operator approval phrase authorizes clean-worktree and quality-gate organization only:

```text
clean-worktree 및 quality gate 정리 승인
```

Still not authorized here: raw secrets, ELIOS access, destructive cleanup outside this worktree, main/master merge, Gateway restart, or enabling live enforcement in production config.
