# Bridge report: t_13e11b7c

## Summary

Added the approved module-level pytest platform guard to `tests/gateway/test_systemd_notify.py`. On non-Linux platforms, including macOS, the systemd-specific module is collected and skipped instead of exercising Linux-only Unix socket behavior. The predicate is false on Linux, so Linux test execution is unchanged.

## Root cause

The test module assumes Linux systemd socket semantics. On Darwin, the filesystem socket test exceeded macOS's shorter `AF_UNIX` path limit and the abstract socket test attempted a Linux-only abstract namespace address. The pre-change run collected 12 tests and failed 2 for those platform differences.

## Validation

- macOS reproduction before the change: 2 failed, 10 passed.
- Exact requested test command after the change: 12 skipped, exit 0. This worktree did not contain `venv/`, so a transient `venv` symlink to the repository's shared virtualenv was created for the command and removed immediately afterward.
- `git diff --check`: exit 0.
- Independent validator: Claude Sonnet returned `VERDICT: PASS`, confirming the module-level guard skips all cases off Linux and is a no-op on Linux.

## Better-design option not implemented

A future follow-up could split platform-neutral watchdog state tests from Linux socket integration tests, retaining more cross-platform coverage. That is broader than the approved suite-level guard and was not implemented.

**Result** done: the scoped pytest command exits 0 on macOS and independent Claude review passed.
**Changed** `tests/gateway/test_systemd_notify.py:6-14`; `ops/bridge-report.md:1-25`.
**Verified** Light tier: `venv/bin/python -m pytest tests/gateway/test_systemd_notify.py` -> exit 0, 12 skipped; `git diff --check` -> exit 0; Claude Sonnet review -> PASS.
**Risk** No Linux host was available; unchanged Linux execution is established by inspection of the false-on-Linux `skipif` predicate, not a Linux test run.
