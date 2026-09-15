"""Regression tests for the dashboard SIGTERM->SIGKILL grace (#111912).

The kill helper historically SIGKILLed after a hardcoded 3.0s grace, but the
dashboard's lifespan teardown budget is ~6s
(``stop_hosted_room_service(timeout=5.0)`` + ``hosted_room_start_thread.join(1.0)``
+ ``PTY_REGISTRY.close_all()``).  A SIGKILL landing inside that window skips
``close_all()`` and orphans ``ui-tui`` / ``tui_gateway.entry`` children that
keep holding ``state.db-wal``; the next startup then aborts with a FATAL
``DeletedWalGenerationError``.

These tests exercise the real ``_kill_pids_posix`` path against real child
processes: the grace must let a graceful teardown finish (no SIGKILL, teardown
marker lands) while still SIGKILLing a process that ignores SIGTERM.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_cli import dashboard_procs

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX kill semantics (os.kill / SIGTERM) only"
)

_GRACEFUL_CHILD = textwrap.dedent(
    """
    import pathlib
    import signal
    import sys
    import time

    marker = pathlib.Path(sys.argv[1])
    ready = pathlib.Path(sys.argv[2])
    teardown_seconds = float(sys.argv[3])

    def _on_term(_signum, _frame):
        # Simulate the lifespan teardown (hosted-room stop + thread join).
        time.sleep(teardown_seconds)
        marker.write_text("teardown-complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_term)
    ready.write_text("ready")
    time.sleep(300)
    """
)

_IGNORING_CHILD = textwrap.dedent(
    """
    import pathlib
    import signal
    import sys
    import time

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(300)
    """
)


def _spawn(script: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _await_ready(child: subprocess.Popen, ready_path) -> None:
    """Wait for the child to install its SIGTERM handler before killing it."""
    deadline = time.monotonic() + 10.0
    while not ready_path.exists():
        if child.poll() is not None:
            raise AssertionError(f"child exited before signaling ready: {child.returncode}")
        if time.monotonic() > deadline:
            raise AssertionError("child never signaled ready")
        time.sleep(0.02)


def _kill_and_reap(child: subprocess.Popen, monkeypatch, grace: float | None = None):
    if grace is not None:
        monkeypatch.setattr(dashboard_procs, "_POSIX_TERM_GRACE_SECONDS", grace)
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    start = time.monotonic()
    dashboard_procs._kill_pids_posix([child.pid], killed, failed)
    elapsed = time.monotonic() - start
    try:
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    return killed, failed, elapsed


def test_graceful_teardown_within_lifespan_budget_is_not_sigkilled(tmp_path, monkeypatch):
    """A teardown as long as the documented lifespan budget must complete.

    Simulates stop_hosted_room_service(5.0) + hosted_room_start_thread.join(1.0)
    + close_all() headroom (6.5s total).  With the production grace the child
    must finish its teardown itself: marker written, exit code 0, no SIGKILL.
    """
    marker = tmp_path / "marker.txt"
    ready = tmp_path / "ready.txt"
    child = _spawn(_GRACEFUL_CHILD, str(marker), str(ready), "6.5")
    _await_ready(child, ready)
    killed, failed, elapsed = _kill_and_reap(child, monkeypatch)

    assert failed == []
    assert marker.exists() and marker.read_text() == "teardown-complete"
    assert child.returncode == 0, f"child killed instead of exiting gracefully: {child.returncode}"
    assert killed == [child.pid]
    assert elapsed >= 6.0, f"kill helper returned before teardown finished ({elapsed:.2f}s)"


def test_sigterm_ignoring_process_is_sigkilled_after_grace(tmp_path, monkeypatch):
    """A process that ignores SIGTERM must still be force-killed at the deadline."""
    ready = tmp_path / "ready.txt"
    child = _spawn(_IGNORING_CHILD, str(ready))
    _await_ready(child, ready)
    killed, failed, elapsed = _kill_and_reap(child, monkeypatch, grace=0.6)

    assert failed == []
    assert child.returncode == -signal.SIGKILL
    assert killed == [child.pid]
    assert elapsed >= 0.5, f"grace expired early ({elapsed:.2f}s)"


def test_fast_graceful_exit_does_not_wait_out_full_grace(tmp_path, monkeypatch):
    """A quick graceful exit must return as soon as the process is gone."""
    marker = tmp_path / "marker.txt"
    ready = tmp_path / "ready.txt"
    child = _spawn(_GRACEFUL_CHILD, str(marker), str(ready), "0.3")
    _await_ready(child, ready)
    killed, failed, elapsed = _kill_and_reap(child, monkeypatch, grace=10.0)

    assert failed == []
    assert marker.read_text() == "teardown-complete"
    assert child.returncode == 0
    assert elapsed < 5.0, f"waited out the grace despite early exit ({elapsed:.2f}s)"
