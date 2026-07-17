"""Regression test for the CLI exit-watchdog gap (PR #65998).

The 30s exit backstop (``_arm_exit_watchdog`` -> daemon timer -> ``os._exit(0)``)
used to be armed ONLY inside ``_run_cleanup``, which runs from the TUI
``finally`` block / ``atexit``. If the main thread wedges before ``app.run()``
returns, that ``finally`` never executes, the watchdog is never armed, and a
"dead" CLI spins forever (observed ~47 min at 4% CPU on a stalled ``hermes --tui``).

The fix arms the watchdog at both chat-entry points (interactive TUI ``run()``
and single-query mode) so the backstop is live from process startup and covers
the entire post-turn -> cleanup window.

These tests spawn REAL subprocesses that import the real ``_arm_exit_watchdog``
from ``cli.py`` and either wedge or exit cleanly, asserting the resulting
process lifetime. Use a short ``HERMES_EXIT_WATCHDOG_S`` so the test is fast.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# Resolve the repo root (tests/cli -> repo root) so the subprocess can import cli.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENV_PY = os.path.join(_REPO_ROOT, "venv", "bin", "python")

_SUBPROCESS_SRC = '''
import os, sys, time
sys.path.insert(0, {repo!r})
from cli import _arm_exit_watchdog
mode = os.environ["TEST_MODE"]
if mode in ("wedge_armed", "clean_exit"):
    # Mirrors the fix: watchdog armed at "startup", before any wedge window.
    _arm_exit_watchdog()
if mode in ("wedge_no_arm", "wedge_armed"):
    # Simulate the wedge: main thread parks before app.run() returns, so the
    # TUI finally/_run_cleanup (which also arms the watchdog) never runs.
    time.sleep(60)
sys.exit(0)
'''.format(
    repo=_REPO_ROOT
)


def _run(mode: str, timeout: float = 8.0):
    """Run the wedge/exit subprocess; return (exited: bool, rc_or_None, elapsed_s).

    The subprocess runs under the hermes venv python (the real runtime that can
    import ``cli``), NOT the test-runner interpreter.
    """
    runner = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable
    env = dict(
        os.environ,
        HERMES_EXIT_WATCHDOG_S="2",  # short so the test is fast (prod default 30)
        TEST_MODE=mode,
        PYTHONPATH=_REPO_ROOT,
    )
    # _arm_exit_watchdog early-returns when PYTEST_CURRENT_TEST is set (it must
    # not kill the live test worker). Strip it so the subprocess behaves like a
    # real CLI invocation rather than inheriting pytest's env.
    env.pop("PYTEST_CURRENT_TEST", None)
    p = subprocess.Popen([runner, "-c", _SUBPROCESS_SRC], env=env)
    t0 = time.time()
    try:
        rc = p.wait(timeout=timeout)
        return True, rc, time.time() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        return False, None, time.time() - t0


def test_pre_fix_wedge_hangs_without_startup_arm():
    """Reproduces the original bug: a wedged CLI with no startup-armed watchdog
    stays alive indefinitely (survives the 2s test window by a long margin)."""
    exited, _rc, _dt = _run("wedge_no_arm")
    assert not exited, "pre-fix wedge should NOT self-exit (that is the bug we fixed)"


def test_post_fix_wedge_self_exits_via_watchdog():
    """The fix: watchdog armed at startup forces a wedged process to exit."""
    # Prefer the venv python if present (matches how cli is actually run), else
    # fall back to the current interpreter.
    runner = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable
    env = dict(
        os.environ,
        HERMES_EXIT_WATCHDOG_S="2",
        TEST_MODE="wedge_armed",
        PYTHONPATH=_REPO_ROOT,
    )
    env.pop("PYTEST_CURRENT_TEST", None)  # see _run() note
    src = _SUBPROCESS_SRC.replace("sys.executable", repr(runner))
    p = subprocess.Popen([runner, "-c", src], env=env)
    t0 = time.time()
    try:
        rc = p.wait(timeout=8)
        dt = time.time() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        raise AssertionError("post-fix wedge should self-exit within the watchdog window")
    # Watchdog fires at ~2s -> os._exit(0), so the process must be gone well
    # before the 8s ceiling. Allow generous slack for CI slowness.
    assert dt < 6.0, f"wedged process took {dt:.1f}s to die; watchdog should have fired ~2s"
    assert rc == 0


def test_post_fix_clean_exit_unaffected():
    """A normal exit path (watchdog armed but main thread returns) must NOT be
    force-killed — it exits 0 promptly."""
    exited, rc, dt = _run("clean_exit")
    assert exited, "clean exit should return on its own"
    assert rc == 0, f"clean exit rc={rc}"
    assert dt < 4.0, f"clean exit took {dt:.1f}s; watchdog must not delay a real exit"
