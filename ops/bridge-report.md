# Bridge report: t_08e0d1db

## Root cause

The live-system guard correctly allows read-only `systemctl` commands to reach the real subprocess implementation. macOS does not provide an executable `systemctl`, so the four pass-through tests failed during executable lookup after the guard allowed them. The other 30 tests either reject commands before process launch or use platform-available primitives.

## Fix

Added a Darwin-only autouse fixture in the scoped test module. It prepends an executable no-op `systemctl` shim to `PATH`. The guard still inspects the original command shapes, blocked commands still raise before launch, no tests are skipped, and non-Darwin execution returns from the fixture without modifying `PATH`.

## Independent validation

The initial Droid/DeepSeek validator failed during harness startup without producing a review. A read-only Claude/Sonnet review then returned `VERDICT: PASS` with no findings. It specifically confirmed that Linux behavior is unchanged and that the shim cannot bypass command-shape guard coverage.

## Evidence

- Baseline: shared venv run produced 4 failures and 30 passes on Darwin.
- Focused verification: shared venv run produced 34 passes and 0 skips on Darwin.
- Ruff: scoped test file passed.
- Diff check: no whitespace errors.
- Environment note: the requested `venv/bin/python` path is absent in this worktree; the repository-supported shared fallback `~/.hermes/hermes-agent/venv/bin/python` was used.

**Result** done: 34/34 scoped tests pass on Darwin with zero skips; independent Claude review passed.
**Changed** tests/test_live_system_guard_self_test.py:23-40; ops/bridge-report.md:1
**Verified** light tier: `~/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_live_system_guard_self_test.py -x -v` -> 34 passed; `~/.hermes/hermes-agent/venv/bin/python -m ruff check tests/test_live_system_guard_self_test.py` -> passed; `git diff --check` -> passed; Claude/Sonnet read-only review -> PASS.
**Risk** Linux was not executed locally; unchanged behavior is established by the explicit `sys.platform != "darwin"` early return and independent review. The requested worktree-local `venv/bin/python` command remains unavailable.
