"""Viewer takeover linearizes at the PTY sink, including off-loop writes."""
import asyncio
import os
from pathlib import Path
import threading

import pytest

from hermes_cli.pty_session import PtySession, WS_CLOSE_SUPERSEDED
from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry


class Socket:
    def __init__(self):
        self.closed = []

    async def close(self, code, reason=None):
        self.closed.append(code)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at_sink", [False, True])
async def test_takeover_revokes_waiting_workers_and_waits_for_an_admitted_write(tmp_path, monkeypatch, pause_at_sink):
    import pty
    import select
    import tty
    from hermes_cli import pty_bridge
    from hermes_cli.profile_incarnation import ensure_profile_incarnation

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, ensure_profile_incarnation(profile), "sh", str(tmp_path))
    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = pty_bridge.PtyBridge.__new__(pty_bridge.PtyBridge)
    bridge._fd, bridge._closed, bridge._fd_lock = master, False, threading.Lock()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    old, new = Socket(), Socket()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_write = os.write

    def pause():
        entered.set()
        assert release.wait(5), "write barrier was not released"

    def sink(fd, data):
        if pause_at_sink and fd == master and bytes(data) == b"OLD\n":
            pause()
        return real_write(fd, data)

    def old_fence(write):
        try:
            if not pause_at_sink:
                pause()
            return owner.admit(write)
        finally:
            finished.set()

    monkeypatch.setattr(pty_bridge.os, "write", sink)
    writing = takeover = None
    try:
        assert await session.attach(old)
        writing = asyncio.create_task(session.write(old, b"OLD\n", fence=old_fence))
        assert await asyncio.to_thread(entered.wait, 3)
        takeover = asyncio.create_task(session.attach(new))
        if pause_at_sink:
            # The sink already admitted OLD: takeover waits, with the event loop
            # still free to run the bridge timeout/cancellation cleanup.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(takeover), 0.05)
            release.set()
        assert await asyncio.wait_for(takeover, 3)
        assert await asyncio.wait_for(writing, 3) is False
        assert old.closed == [WS_CLOSE_SUPERSEDED]
        assert await session.write(new, b"NEW\n", fence=owner.admit)
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        readable, _, _ = await asyncio.to_thread(select.select, [slave], [], [], 3)
        assert readable
        assert os.read(slave, 1024) == (b"OLD\nNEW\n" if pause_at_sink else b"NEW\n")
        assert session.alive
    finally:
        release.set()
        for task in (writing, takeover):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(finished.wait, 5)
        os.close(master)
        os.close(slave)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_host_reaper_releases_dead_shell_before_drain_observes_eof(tmp_path):
    import signal
    from hermes_cli.pty_bridge import PtyBridge

    bridge = PtyBridge.spawn(["/bin/sh", "-c", "trap '' HUP; sleep 60 & echo ready; exec sleep 60"], cwd=str(tmp_path))
    registry = HostTerminalRegistry()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    registry._sessions[session.key] = session
    registry.owners[session.key] = HostOwner(("loopback", "test"), tmp_path, None, "sh", str(tmp_path))
    viewer = Socket()
    try:
        async with asyncio.timeout(5):
            while b"ready" not in (await asyncio.to_thread(bridge.read, 0.05) or b""):
                await asyncio.sleep(0)
        assert await session.attach(viewer)
        os.kill(bridge.pid, signal.SIGKILL)
        async with asyncio.timeout(5):
            while bridge.is_alive():
                await asyncio.sleep(0.01)
        # The drain has not observed EOF. A helper can delay it indefinitely
        # on Linux; macOS can report EOF as soon as the leader exits.
        assert session.alive and session.attached
        await registry.reap_idle()
        assert not registry._sessions and not registry.owners
        assert bridge._closed
    finally:
        await registry.close_all()
        await asyncio.to_thread(bridge.close)
