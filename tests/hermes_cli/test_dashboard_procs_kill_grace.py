"""Dashboard stop waits for graceful teardown and reaps a wedged process tree.

A forced backend kill must not leave ui-tui / tui_gateway descendants holding a deleted
``state.db-wal`` inode. These tests use real child processes and real POSIX signals.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

from hermes_cli import dashboard_procs

pytestmark = pytest.mark.linux_only

_GRACEFUL_CHILD = textwrap.dedent(
    """
    import pathlib, signal, sys, time
    marker, ready, secs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
    def _on_term(_signum, _frame):
        time.sleep(secs)
        marker.write_text("teardown-complete")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_term)
    ready.write_text("ready")
    time.sleep(300)
    """
)

_IGNORING_CHILD = textwrap.dedent(
    """
    import pathlib, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(300)
    """
)

_TREE_PARENT = textwrap.dedent(
    """
    import pathlib, signal, subprocess, sys, time
    marker, ready, child_pid, mode = map(pathlib.Path, sys.argv[1:])
    child = subprocess.Popen([
        sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
    ])
    child_pid.write_text(str(child.pid))
    if mode.name == "graceful":
        def _on_term(_signum, _frame):
            time.sleep(0.3)
            child.kill()
            child.wait()
            marker.write_text("tree-teardown-complete")
            sys.exit(0)
        signal.signal(signal.SIGTERM, _on_term)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.write_text("ready")
    time.sleep(300)
    """
)


def _spawn_ready(script: str, ready_path, *args: str) -> subprocess.Popen:
    child = subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10.0
    while not ready_path.exists():
        if child.poll() is not None:
            raise AssertionError(f"child exited before signaling ready: {child.returncode}")
        if time.monotonic() > deadline:
            raise AssertionError("child never signaled ready")
        time.sleep(0.02)
    return child


def _kill_and_reap(child: subprocess.Popen):
    killed: list[int] = []
    failed: list[tuple[int, str]] = []
    dashboard_procs._kill_pids_posix([child.pid], killed, failed)
    try:
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    return killed, failed


def test_teardown_as_long_as_lifespan_budget_exits_gracefully(tmp_path):
    """A teardown spanning the 5s hosted-room stop + 1s join must not be SIGKILLed."""
    marker, ready = tmp_path / "marker", tmp_path / "ready"
    child = _spawn_ready(_GRACEFUL_CHILD, ready, str(marker), str(ready), "6.2")

    killed, failed = _kill_and_reap(child)

    assert failed == []
    assert child.returncode == 0, f"SIGKILLed mid-teardown (rc={child.returncode})"
    assert marker.read_text() == "teardown-complete"
    assert killed == [child.pid]


def test_sigterm_ignoring_process_is_still_sigkilled(tmp_path, monkeypatch):
    """The grace is a ceiling, not a wait: a process that ignores SIGTERM is force-killed."""
    monkeypatch.setattr(dashboard_procs, "_POSIX_TERM_GRACE_SECONDS", 0.6)
    ready = tmp_path / "ready"
    child = _spawn_ready(_IGNORING_CHILD, ready, str(ready))

    killed, failed = _kill_and_reap(child)

    assert failed == []
    assert child.returncode == -signal.SIGKILL
    assert killed == [child.pid]


def test_graceful_backend_teardown_waits_for_its_descendant(tmp_path):
    marker, ready, child_pid_path = (
        tmp_path / "marker", tmp_path / "ready", tmp_path / "child-pid")
    parent = _spawn_ready(
        _TREE_PARENT, ready, str(marker), str(ready), str(child_pid_path), "graceful")
    descendant_pid = int(child_pid_path.read_text())

    killed, failed = _kill_and_reap(parent)

    assert failed == []
    assert parent.returncode == 0
    assert marker.read_text() == "tree-teardown-complete"
    assert not psutil.pid_exists(descendant_pid)
    assert killed == [parent.pid]


def test_wedged_backend_force_kill_reaps_its_descendant(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_procs, "_POSIX_TERM_GRACE_SECONDS", 0.6)
    marker, ready, child_pid_path = (
        tmp_path / "marker", tmp_path / "ready", tmp_path / "child-pid")
    parent = _spawn_ready(
        _TREE_PARENT, ready, str(marker), str(ready), str(child_pid_path), "wedged")
    descendant_pid = int(child_pid_path.read_text())

    killed, failed = _kill_and_reap(parent)

    assert failed == []
    assert parent.returncode == -signal.SIGKILL
    assert not psutil.pid_exists(descendant_pid)
    assert killed == [parent.pid]
