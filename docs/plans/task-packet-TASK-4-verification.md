# Task Packet — TASK-4

## Title
Verification gate: tests, compile checks, and hygiene audit

## Objective
Produce fresh verification evidence for merge recommendation.

## Scope In
- Run targeted tests for changed modules
- Run compile checks on touched Python files
- Run hygiene scan (branch name, artifacts, diff scope)

## Scope Out
- No new feature behavior

## Target Files
- test files under `tests/gateway/` + `tests/hermes_cli/`
- touched Python modules under `gateway/` and `hermes_cli/`

## Required Tests
```bash
source venv/bin/activate
python -m pytest tests/gateway/test_routing_policy.py tests/gateway/test_command_rbac_audit.py tests/gateway/test_exec_owner_commands.py tests/gateway/test_discord_attachments.py tests/hermes_cli/test_model_validation.py -q
python -m py_compile gateway/run.py gateway/platforms/discord.py hermes_cli/models.py
```

## Acceptance Criteria
- [ ] All targeted tests pass
- [ ] Compile checks pass
- [ ] Hygiene checks clear

## Commit Target
`test(gateway): refresh verification evidence for feature tranche`

## Rollback Note
N/A (verification only).
