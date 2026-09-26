"""Ownership, replay ordering, and spawn reservation contracts for PTY sessions."""
import asyncio
import threading
import time

import pytest

from hermes_cli.pty_session import PtySession, PtySessionRegistry, RegistryFull


class Bridge:
    def __init__(self):
        self.closed = False
        self.written = []
        self.resized = []
        self.output = []

    def read(self, timeout):
        if self.output:
            return self.output.pop(0)
        time.sleep(0.001)
        return b""

    async def write(self, data):
        self.written.append(data)
        return True

    def resize(self, cols, rows):
        self.resized.append((cols, rows))

    def close(self):
        self.closed = True


class Socket:
    def __init__(self):
        self.frames = []
        self.closed = None

    async def send_text(self, text):
        self.frames.append(text)

    async def send_bytes(self, data):
        self.frames.append(data)

    async def close(self, code):
        self.closed = code


@pytest.mark.asyncio
async def test_metadata_replay_live_order_and_superseded_input_fence():
    bridge = Bridge()
    session = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    session.buffer.append(b"replay")
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowMetadata(Socket):
        async def send_text(self, text):
            entered.set()
            await release.wait()
            await super().send_text(text)

    old = SlowMetadata()
    attaching = asyncio.create_task(session.attach(old, initial_text="meta"))
    await entered.wait()
    bridge.output.append(b"live")
    await session.start()
    # Eventual drain observation, not a timing assertion.
    async def buffered():
        while session.buffer.snapshot() != b"replaylive":
            await asyncio.sleep(0.001)
    await asyncio.wait_for(buffered(), 2)
    assert old.frames == []
    release.set()
    assert await attaching
    async def sent():
        while len(old.frames) != 3:
            await asyncio.sleep(0.001)
    await asyncio.wait_for(sent(), 2)
    assert old.frames == ["meta", b"replay", b"live"]
    replacement = Socket()
    assert await session.attach(replacement, initial_text="resume")
    assert replacement.frames == ["resume", b"replaylive"]
    assert old.closed == 4409
    await session.write(old, b"stale")
    session.resize(old, cols=5, rows=6)
    session.detach(old)
    assert session._ws is replacement
    assert bridge.written == []
    assert bridge.resized == []
    await session.write(replacement, b"current")
    session.resize(replacement, cols=80, rows=24)
    assert bridge.written == [b"current"]
    assert bridge.resized == [(80, 24)]
    await session.close()


@pytest.mark.asyncio
async def test_concurrent_attach_claims_before_slow_old_close():
    session = PtySession("k", Bridge(), buffer_cap=1024, read_timeout=0.01)
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowClose(Socket):
        async def close(self, code):
            entered.set()
            await release.wait()
            await super().close(code)

    old, first, winner = SlowClose(), Socket(), Socket()
    await session.attach(old)
    one = asyncio.create_task(session.attach(first, initial_text="first"))
    await entered.wait()
    assert await session.attach(winner, initial_text="winner")
    release.set()
    assert not await one
    assert session._ws is winner
    assert first.closed == 4409
    assert first.frames == []
    assert winner.frames == ["winner", b""]
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["metadata", "replay"])
async def test_failed_initial_delivery_has_finite_retention(failure):
    reg = PtySessionRegistry(ttl=30, max_sessions=1, buffer_cap=32, read_timeout=0.01)
    bridge = Bridge()
    session, _ = await reg.attach_or_spawn("k", spawn=lambda: bridge)
    session.buffer.append(b"buffer")

    class BrokenSocket(Socket):
        async def send_text(self, text):
            if failure == "metadata":
                raise RuntimeError("lost metadata")
            await super().send_text(text)

        async def send_bytes(self, data):
            raise RuntimeError("lost replay")

    assert not await session.attach(BrokenSocket(), initial_text="meta")
    assert not session.attached and session.last_detached_at is not None
    await reg.reap_idle(now=session.last_detached_at + 31)
    assert bridge.closed
    assert not reg._sessions
    await reg.close_all()


@pytest.mark.asyncio
async def test_concurrent_spawn_reserves_key_capacity_and_shutdown():
    reg = PtySessionRegistry(ttl=30, max_sessions=1, buffer_cap=32, read_timeout=0.01)
    entered, release = threading.Event(), threading.Event()
    bridge = Bridge()
    calls = []

    def spawn():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return bridge

    one = asyncio.create_task(reg.attach_or_spawn("k", spawn=spawn))
    assert await asyncio.to_thread(entered.wait, 2)
    two = asyncio.create_task(reg.attach_or_spawn("k", spawn=spawn))
    release.set()
    (first, made), (second, made_again) = await asyncio.gather(one, two)
    assert first is second and made and not made_again
    assert len(calls) == 1
    await first.attach(Socket())
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("other", spawn=spawn)
    await reg.close_all()
    assert bridge.closed

    # Shutdown must wait for an admitted off-thread spawn and close its child.
    reg = PtySessionRegistry(ttl=30, max_sessions=1, buffer_cap=32, read_timeout=0.01)
    bridge = Bridge()
    entered.clear()
    release.clear()
    spawning = asyncio.create_task(reg.attach_or_spawn("k", spawn=spawn))
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(reg.close_all())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(spawning, closing)
    assert bridge.closed and not reg._sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("spawn_fails", [False, True])
async def test_cancelled_spawn_retains_capacity_until_cleanup(spawn_fails):
    import gc

    reg = PtySessionRegistry(ttl=30, max_sessions=1, buffer_cap=32, read_timeout=0.01)
    entered, release = threading.Event(), threading.Event()
    bridge, replacement = Bridge(), Bridge()
    events, unhandled = [], []
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    def spawn():
        entered.set()
        assert release.wait(5)
        events.append("spawn finished")
        if spawn_fails:
            raise OSError("fork failed after caller disconnected")
        return bridge

    def next_spawn():
        events.append("next spawn")
        assert spawn_fails or bridge.closed, "capacity released before abandoned child was closed"
        return replacement

    spawning = asyncio.create_task(reg.attach_or_spawn("k", spawn=spawn))
    next_request = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        spawning.cancel()
        await asyncio.sleep(0)
        spawning.cancel()
        # Request cancellation must finish without waiting for fork/exec, even
        # under a level-triggered ASGI cancellation scope.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(spawning, 2)
        next_request = asyncio.create_task(reg.attach_or_spawn("k", spawn=next_spawn))
        await asyncio.sleep(0)
        release.set()
        session, made = await asyncio.wait_for(next_request, 3)
        assert made and session.bridge is replacement
        assert events == ["spawn finished", "next spawn"]
        assert spawn_fails or bridge.closed
        assert session.last_detached_at is not None
        await reg.reap_idle(now=session.last_detached_at + 31)
        assert replacement.closed and not reg._sessions
        # A late spawn failure has no waiting caller, but must still be consumed.
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled
    finally:
        release.set()
        await asyncio.gather(spawning, *([next_request] if next_request else []), return_exceptions=True)
        await reg.close_all()
        bridge.close()
        replacement.close()
        loop.set_exception_handler(original_handler)


@pytest.mark.asyncio
async def test_spawn_failure_reaches_caller_and_releases_capacity():
    reg = PtySessionRegistry(ttl=30, max_sessions=1, buffer_cap=32, read_timeout=0.01)
    failure = OSError("fork failed with caller still connected")
    bridge = Bridge()

    def spawn():
        raise failure

    try:
        with pytest.raises(OSError) as caught:
            await reg.attach_or_spawn("k", spawn=spawn)
        assert caught.value is failure
        session, made = await asyncio.wait_for(
            reg.attach_or_spawn("k", spawn=lambda: bridge), 2,
        )
        assert made and session.bridge is bridge
    finally:
        await reg.close_all()
    assert bridge.closed


@pytest.mark.asyncio
async def test_supersede_cancels_inflight_input_before_replacement_writes():
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockedBridge(Bridge):
        async def write(self, data):
            if data == b"old":
                entered.set()
                await release.wait()
            return await super().write(data)

    bridge = BlockedBridge()
    session = PtySession("k", bridge, buffer_cap=32, read_timeout=0.01)
    old, new = Socket(), Socket()
    await session.attach(old)
    writing = asyncio.create_task(session.write(old, b"old"))
    await entered.wait()
    await session.attach(new)
    release.set()
    assert not await writing
    assert await session.write(new, b"new")
    assert bridge.written == [b"new"]
    await session.close()


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_superseded_input_cancellation_wins_completed_readiness(monkeypatch):
    import errno
    import os
    import pty
    import select
    import tty
    from hermes_cli.pty_bridge import PtyBridge

    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    bridge = PtyBridge.__new__(PtyBridge)
    bridge._fd, bridge._closed = master, False
    session = PtySession("k", bridge, buffer_cap=32, read_timeout=0.01)
    old, new = Socket(), Socket()
    entered, callbacks, writes = asyncio.Event(), [], []
    loop = asyncio.get_running_loop()
    real_write = os.write

    def write(fd, data):
        if fd == master:
            if not entered.is_set():
                raise BlockingIOError(errno.EAGAIN, "buffer full")
            writes.append(bytes(data))
        return real_write(fd, data)

    def add_writer(fd, callback):
        assert fd == master
        # Control selector delivery, not _wait_writable: a real PTY's capacity
        # changes asynchronously and filling it cannot pin this same-tick race.
        callbacks.append(callback)
        entered.set()

    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(loop, "add_writer", add_writer)
    writing = None
    try:
        await session.attach(old)
        writing = asyncio.create_task(session.write(old, b"STALE"))
        await asyncio.wait_for(entered.wait(), 2)
        callbacks[0]()
        # Ready then cancel before the waiting task resumes: wait_for on 3.11
        # used to return its completed Future and swallow this cancellation.
        await session.attach(new)
        result = await asyncio.wait_for(writing, 2)
        assert await session.write(new, b"CURRENT")
        assert writes == [b"CURRENT"]
        assert not result and session.alive
        readable, _, _ = await asyncio.to_thread(select.select, [slave], [], [], 2)
        assert readable and os.read(slave, 1024) == b"CURRENT"
    finally:
        if writing is not None:
            writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)
        os.close(master)
        os.close(slave)


@pytest.mark.asyncio
async def test_cancelled_close_still_reaps_child():
    bridge = Bridge()
    session = PtySession("k", bridge, buffer_cap=32, read_timeout=0.01)
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowClose(Socket):
        async def close(self, code):
            entered.set()
            await release.wait()

    await session.attach(SlowClose())
    closing = asyncio.create_task(session.close())
    await entered.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await session.close()
    assert bridge.closed
