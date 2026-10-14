# Test plan: completion gate through real Kanban API
Max-iterations: 6

## TC-001: typed deployed card with missing artifact stays running
Type: cli
Run: `HERMES_HOME=$(mktemp -d) .venv/bin/python docs/qa/t_7c16c4ed/run_smoke.py`, creating a running `deployed` card and calling `complete_task` without `metadata.verification.artifact`.
Expect: completion raises `CompletionVerificationError`, report contains `missing_artifact`, and the card status remains `running`.
Wiring: Kanban completion entry point -> `kanban_db.complete_task` -> `verify_completion` -> persisted task status and blocking event.

## TC-002: typed deployed card with required artifact completes
Type: cli
Run: `HERMES_HOME=$(mktemp -d) .venv/bin/python docs/qa/t_7c16c4ed/run_smoke.py`, creating a running `deployed` card and calling `complete_task` with a valid deployed artifact including URL and successful smoke-test command.
Expect: completion returns successfully and the card status is `done`.
Wiring: Kanban completion entry point -> `kanban_db.complete_task` -> artifact contract validation -> durable done transition and verification report.

## TC-003: targeted regression suite remains green
Type: cli
Run: `scripts/run_tests.sh tests/hermes_cli/test_kanban_completion_verification_gate.py tests/hermes_cli/test_kanban_pr_acceptance.py`
Expect: all tests pass.
Wiring: repository test runner -> completion-gate behavior tests -> gate and PR-acceptance implementation.
