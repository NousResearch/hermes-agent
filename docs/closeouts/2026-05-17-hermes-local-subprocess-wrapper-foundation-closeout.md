# Hermes Local Subprocess Wrapper Foundation Closeout

Date: 2026-05-17
Scope: A-slice local foundation only

## What changed

- Added `agent/cmh_subprocess/` local foundation package.
- Added CLI flag verification for Claude help evidence.
- Added profile-aware envelope state helpers.
- Added profile-aware halt flag helpers.
- Added disabled-by-default wrapper preflight helpers.
- Added pure budget display formatter.
- Added focused pytest coverage.
- Added hardening discovered during review: expired-window budget checks, atomic envelope writes, strict halt-flag boolean validation, unsafe prompt rejection, and structured state-error returns for malformed envelope state.

## Verification

- `python -m pytest tests/agent/test_cmh_subprocess_flags.py tests/agent/test_cmh_subprocess_envelope.py tests/agent/test_cmh_subprocess_halt_flags.py tests/agent/test_cmh_subprocess_wrappers.py tests/agent/test_cmh_subprocess_budget_display.py -q -o 'addopts='` passed with 37 tests.
- `python -m py_compile agent/cmh_subprocess/__init__.py agent/cmh_subprocess/result.py agent/cmh_subprocess/flags.py agent/cmh_subprocess/envelope.py agent/cmh_subprocess/halt_flags.py agent/cmh_subprocess/wrappers.py agent/cmh_subprocess/budget_display.py` passed.
- `git diff --check` passed.
- Changed-file scope was limited to `agent/cmh_subprocess/`, `tests/agent/test_cmh_subprocess_*.py`, and this closeout.

## Activation status

No activation occurred.

Not changed:

- Telegram sends or Telegram verbs.
- Gateway callbacks, gateway restart, or gateway config.
- Cron, launchd, daemon, MCP, AgentMail, provider, or profile config.
- `~/.hermes/wrappers/` or `~/.hermes/bin/` runtime scripts.
- Cowork-headless spawn.
- Codex auto-dispatch.
- R109 fire.
- Git push, merge, deploy, or production mutation.

## Known dependencies for later phases

- Codex Wave 1.16.E verified flag docs are still required before activation.
- `codex` must be present on PATH or the Codex wrapper remains disabled.
- Any Telegram or gateway exposure requires separate exact approval.
