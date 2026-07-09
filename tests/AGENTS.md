# Tests Subtree Instructions

This file scopes test-suite guidance to `tests/` work. Root `AGENTS.md` still contains the non-negotiable project rules.

## Test runner

- Always use `scripts/run_tests.sh`; do not call `pytest` directly.
- The wrapper enforces CI parity: hermetic env, temp `HERMES_HOME`, UTC/C.UTF-8, xdist, and subprocess-per-test-file isolation.
- Tests must never write to the real `~/.hermes/`.

## Test quality

- Do not write change-detector tests that freeze model catalogs, config version literals, enum counts, or other expected-to-change data.
- Prefer behavior contracts and invariants: plumbing works, all catalog entries have required metadata, migrations reach the current version, unsafe overlaps do not exist.
- For config/profile tests, mock both `Path.home()` and `HERMES_HOME` so default and profile paths stay inside the temp dir.
- Add platform skip guards for POSIX-only behavior and patch `sys.platform` together with `platform.system()`/`release()` when simulating OS branches.
