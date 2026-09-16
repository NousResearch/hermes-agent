"""The dashboard SIGTERM→SIGKILL grace must outlast the lifespan teardown (#111912).

``hermes update`` / ``hermes dashboard --stop`` fall back to ``_kill_pids_posix`` for a
manually-started backend. Its lifespan teardown blocks on ``stop_hosted_room_service(timeout=5.0)``
before ``PTY_REGISTRY.close_all()`` runs; a SIGKILL inside that window orphans the ui-tui /
tui_gateway.entry children, which keep the deleted ``state.db-wal`` inode open until the next
start aborts with ``DeletedWalGenerationError``. Real child processes, real signals.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_cli import dashboard_procs

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics only")

# Mirrors the lifespan: sleep for the teardown budget on SIGTERM, then leave a marker and exit 0.
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


def _spawn_ready(script: str, ready_path, *args: str) -> subprocess.Popen:
    child = subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10.0
    while not ready_path.exists():  # the SIGTERM handler must be installed before we signal
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
