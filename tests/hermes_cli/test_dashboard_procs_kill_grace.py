"""The dashboard SIGTERM→SIGKILL grace must outlast the lifespan teardown (#111912), and a
descendant that outlives it must not survive the stop (#112631).

``hermes update`` / ``hermes dashboard --stop`` fall back to ``_kill_pids_posix`` for a
manually-started backend. Its lifespan teardown blocks on ``stop_hosted_room_service(timeout=5.0)``
before ``PTY_REGISTRY.close_all()`` runs; a SIGKILL inside that window orphans the ui-tui /
tui_gateway.entry children, which keep the deleted ``state.db-wal`` inode open until the next
start aborts with ``DeletedWalGenerationError``. A wedged descendant defeats any finite grace, so
the kill sequence sweeps the dashboard-owned tree after the root — but never the detached
messaging-gateway bots the dashboard launched with ``start_new_session``. Real child processes,
real signals, a real PTY.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_cli import dashboard_procs

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics only")

_IGNORING_CHILD = textwrap.dedent(
    """
    import pathlib, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text("ready")
    time.sleep(300)
    """
)

# A wedged hosted TUI: ignores SIGTERM and the SIGHUP its PTY master's close delivers.
_WEDGED_DESCENDANT = textwrap.dedent(
    """
    import os, pathlib, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(300)
    """
)

_DETACHED_BOT = textwrap.dedent(
    """
    import os, pathlib, sys, time
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(300)
    """
)

# Backend stand-in: a hosted TUI behind a real PTY (PtyBridge.spawn shape) plus a gateway bot
# launched detached (web_server_gateway / web_routers.messaging shape); its own SIGTERM teardown
# is wedged so the root gets SIGKILLed mid-teardown, exactly the incident.
_WEDGED_BACKEND = textwrap.dedent(
    f"""
    import pathlib, signal, subprocess, sys, time
    import ptyprocess
    tui_pid, bot_pid, ready = (pathlib.Path(p) for p in sys.argv[1:4])
    # Keep the handle: a collected PtyProcess terminates its child (SIGHUP…SIGKILL) from __del__.
    tui = ptyprocess.PtyProcess.spawn([sys.executable, "-c", {_WEDGED_DESCENDANT!r}, str(tui_pid)])
    subprocess.Popen([sys.executable, "-c", {_DETACHED_BOT!r}, str(bot_pid)], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while not (tui_pid.exists() and bot_pid.exists()):
        time.sleep(0.01)
    signal.signal(signal.SIGTERM, lambda *_: time.sleep(300))
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


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, check=False).stdout.strip()
    return bool(stat) and not stat.startswith("Z")




def test_sigterm_ignoring_process_is_still_sigkilled(tmp_path, monkeypatch):
    """The grace is a ceiling, not a wait: a process that ignores SIGTERM is force-killed."""
    monkeypatch.setattr(dashboard_procs, "_POSIX_TERM_GRACE_SECONDS", 0.6)
    ready = tmp_path / "ready"
    child = _spawn_ready(_IGNORING_CHILD, ready, str(ready))

    killed, failed = _kill_and_reap(child)

    assert failed == []
    assert child.returncode == -signal.SIGKILL
    assert killed == [child.pid]


@pytest.mark.live_system_guard_bypass  # the orphans are reparented out of the test subtree by design
def test_wedged_pty_descendant_is_gone_but_detached_bot_survives(tmp_path, monkeypatch):
    """#112631: when the stop returns, the hosted TUI that outlived the SIGKILLed backend is dead
    (it would hold the deleted state.db-wal inode), while the messaging-gateway bot the dashboard
    started with ``start_new_session`` is untouched."""
    pytest.importorskip("ptyprocess")
    monkeypatch.setattr(dashboard_procs, "_POSIX_TERM_GRACE_SECONDS", 0.6)
    tui_pid_file, bot_pid_file, ready = tmp_path / "tui.pid", tmp_path / "bot.pid", tmp_path / "ready"
    backend = _spawn_ready(_WEDGED_BACKEND, ready, str(tui_pid_file), str(bot_pid_file), str(ready))
    tui_pid, bot_pid = int(tui_pid_file.read_text()), int(bot_pid_file.read_text())
    try:
        killed, failed = _kill_and_reap(backend)

        assert (killed, failed) == ([backend.pid], [])
        assert backend.returncode == -signal.SIGKILL
        assert not _pid_running(tui_pid), "wedged hosted TUI outlived the dashboard stop"
        assert _pid_running(bot_pid), "detached gateway bot was killed with the dashboard"
    finally:
        for pid in (tui_pid, bot_pid):
            if _pid_running(pid):
                os.kill(pid, signal.SIGKILL)
