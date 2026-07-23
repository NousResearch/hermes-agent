# Verification — shared goal outcomes surface

## Change under test

- Added optional same-session filtering to `list_reusable_outcome_receipts()`.
- Added a shared read-only formatter and `/goal outcomes` / `/goal learning`
  transport wiring for CLI, gateway, and TUI.
- The surface returns only currently passing, same-session receipts and states
  that it does not mutate prompts, memory, skills, or receipt state.

## Executed verification

```text
C:\Users\82109\AppData\Local\hermes\git\bin\bash.exe -lc \
  "export HERMES_PYTHON=C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe; \
   scripts/run_tests.sh tests/hermes_cli/test_goals.py \
     tests/agent/test_verification_evidence.py \
     tests/tui_gateway/test_goal_command.py -q"
```

Result: 3 files, 152 tests passed, 0 failed. The ordinary `bash.exe` is a
WSL shim without an installed distribution on this host; the repository-local
Git Bash path above successfully ran the mandated test wrapper with the
existing Python test environment.

## Independent-review repair

The first independent review found that CLI and gateway command dispatch did
not yet have direct regression tests. Added a CLI dispatch test and a gateway
dispatch test, then restored an accidentally displaced pre-existing gateway
confirmation test before re-verification.

The review-expanded eight-file run passed all seven goal/receipt/transport and
approval-routing modules. `tests/tools/test_approval.py` executed 310 tests
but had two unrelated Windows portability failures: a POSIX `/tmp` expectation
and unavailable symlink privilege (`WinError 1314`). They do not exercise the
outcome surface and no approval-policy code was changed in this branch.

## Final focused rerun

```text
C:\Users\82109\AppData\Local\hermes\git\bin\bash.exe -lc \
  "export HERMES_PYTHON=C:/Users/82109/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe; \
   scripts/run_tests.sh tests/agent/test_verification_evidence.py \
     tests/hermes_cli/test_goals.py \
     tests/gateway/test_goal_max_turns_config.py \
     tests/tui_gateway/test_goal_command.py -q"
```

Result: 4 files, 158 tests passed, 0 failed. This includes the final
fail-closed workspace-root test and direct CLI/gateway/TUI outcome dispatch
coverage.

## New contracts covered

- Same-workspace reusable receipts can be scoped to one session.
- The shared formatter passes the active session id and remains pull-only.
- TUI `/goal outcomes` uses the current session's cwd and session key.
