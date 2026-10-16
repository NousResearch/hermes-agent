"""Viewer takeover linearizes at the PTY sink, including off-loop writes."""
import asyncio
import os
from pathlib import Path
import threading

import pytest

from hermes_cli.pty_session import PtySession, WS_CLOSE_SUPERSEDED
from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry
from tests.waiting import async_wait_until


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


class InputSocket(Socket):
    """A viewer that sends ``frames`` and then reads until it is closed."""

    def __init__(self, *frames):
        super().__init__()
        self.inbound = asyncio.Queue()
        for frame in frames:
            self.inbound.put_nowait({"type": "websocket.receive", "bytes": frame})

    async def receive(self):
        return await self.inbound.get()

    async def close(self, code, reason=None):
        await super().close(code, reason)
        self.inbound.put_nowait({"type": "websocket.disconnect"})


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_takeover_during_an_admitted_input_write_keeps_the_shell(tmp_path, monkeypatch):
    """The cancelled write returns False while the old socket is still the viewer (the takeover
    waits for the sink); the pump must neither close the shell nor let the old tab reconnect."""
    from types import SimpleNamespace
    from hermes_cli import pty_bridge
    from hermes_cli.profile_incarnation import ensure_profile_incarnation
    from hermes_cli.web_host_terminal_sessions import _pump_input

    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = HostOwner(("loopback", "test"), profile, ensure_profile_incarnation(profile), "sh", str(tmp_path))
    import pty
    import tty
    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = pty_bridge.PtyBridge.__new__(pty_bridge.PtyBridge)
    bridge._fd, bridge._closed, bridge._fd_lock = master, False, threading.Lock()
    session = PtySession("terminal", bridge, buffer_cap=32, read_timeout=0.01)
    entered, release = threading.Event(), threading.Event()
    real_write = os.write

    def sink(fd, data):
        if fd == master and bytes(data) == b"OLD\n":
            entered.set()
            assert release.wait(5), "write barrier was not released"
        return real_write(fd, data)

    monkeypatch.setattr(pty_bridge.os, "write", sink)
    removed = []

    async def remove(token):
        removed.append(token)

    old, new = InputSocket(b"OLD\n"), Socket()
    pump = takeover = None
    try:
        assert await session.attach(old)
        pump = asyncio.create_task(_pump_input(old, SimpleNamespace(remove=remove), "terminal", session, owner))
        assert await asyncio.to_thread(entered.wait, 3)
        takeover = asyncio.create_task(session.attach(new))
        # The takeover cancelled the write and waits for the sink; the old socket is still the viewer.
        await async_wait_until(lambda: session._input_task is None, timeout=3)
        assert session._ws is old
        release.set()
        assert await asyncio.wait_for(takeover, 3)
        async with asyncio.timeout(3):
            await pump
        assert removed == [] and session.alive and not bridge._closed
        assert old.closed == [WS_CLOSE_SUPERSEDED]
        assert await session.write(new, b"NEW\n", fence=owner.admit)
    finally:
        release.set()
        for task in (pump, takeover):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
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
        await async_wait_until(lambda: not bridge.is_alive(), timeout=5)
        # The drain has not observed EOF. A helper can delay it indefinitely
        # on Linux; macOS can report EOF as soon as the leader exits.
        assert session.alive and session.attached
        await registry.reap_idle()
        assert not registry._sessions and not registry.owners
        assert bridge._closed
    finally:
        await registry.close_all()
        await asyncio.to_thread(bridge.close)
