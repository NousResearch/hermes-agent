# Read-only Mission Control action API

## Goal

Expose a narrow Mission Control API for approved read-only operational checks without reusing the dashboard's existing mutating action endpoints.

## Scope

- Add an allowlisted action registry for read-only actions only.
- Add an API endpoint that invokes a known action id and returns structured JSON.
- Support these approved checks:
  - data refresh/read
  - backup status check
  - latest cron output retrieval
  - Hermes health check
  - route/plugin health checks
- Reject unknown or unsafe action ids.
- Use bounded timeouts/captured output for subprocess-backed checks.

## Non-goals

- No restart, update, deploy, delete, write, install, enable, disable, pause, resume, trigger, or profile mutation behavior.
- No shell access and no user-supplied command strings.
- No frontend wiring in this task.

## Expected files

- `hermes_cli/web_server.py`
- `tests/hermes_cli/test_web_server.py`

## Test/verification strategy

- Add API tests first for all approved action ids, unsafe action rejection, structured response shape, path traversal rejection, command allowlist enforcement, and subprocess timeout/error capture.
- Run targeted pytest for web server tests.
- Run diff checks.

## Progress

- [x] Repo state inspected and feature branch confirmed.
- [x] Spec/plan written.
- [x] Failing tests added.
- [x] API implemented.
- [x] Targeted tests pass.
- [x] Independent review completed.
- [ ] Human review handoff recorded.
