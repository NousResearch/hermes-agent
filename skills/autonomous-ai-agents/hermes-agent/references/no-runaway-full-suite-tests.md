# No runaway full-suite tests

Use when Al asks to stop agents from running giant test suites in live Telegram/gateway/resumed CLI sessions, while keeping `pytest` usable.

## Rule

Keep pytest available. Reduce blast radius.

Full-suite `pytest tests/` is allowed only in:

- approved release/final-validation phase, or
- explicit Al approval for that exact full-suite command.

It is not a reflex for normal live support/debug/build turns.

## Default test ladder

1. One function bugfix → run nearest exact node ID:
   ```bash
   pytest tests/path/test_file.py::test_case -q
   ```
2. One file/module change → run matching test file/module:
   ```bash
   pytest tests/path/test_file.py -q
   ```
3. Plugin/tool change → run that plugin/tool slice only:
   ```bash
   pytest tests/plugins/test_name.py -q -o 'addopts='
   ```
4. Infra/config/core dependency change → run directly affected package tests only.
5. Release/final validation → full suite allowed if approved:
   ```bash
   pytest tests/ -q -o 'addopts='
   ```

## Stop conditions

Stop and report instead of escalating when:

- targeted tests pass but broad suite shows many unrelated failures;
- full suite hangs or runs long;
- same full-suite command already failed/hung once;
- session is Telegram/live gateway/resumed CLI and Al did not approve full suite;
- `/goal` or high-turn budget starts drifting into repeated broad test attempts.

## Reporting shape

Use this short shape:

```text
PROVEN:
- focused test command + result

GAP:
- full suite not run because <live session / no approval / broad unrelated failures>

NEXT:
- approve exact full-suite command only if this is release/final validation
```

## Runaway recovery proof

If runaway already happened, prove:

- exact Hermes PID/session/log tag;
- exact child pytest command/PID;
- whether it was full-suite or targeted;
- gateway stayed running and was not restarted;
- no `pytest tests/` process remains after stop.

## Bad pattern to avoid

Do not say “verification requires full suite” in a live session. Say:

> Focused tests prove the touched change. Full suite is deferred to release/final validation unless Al approves it now.
