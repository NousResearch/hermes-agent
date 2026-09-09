import asyncio
import time

import pytest

from hermes_cli.pty_session import RingBuffer


def test_ringbuffer_keeps_everything_under_capacity():
    rb = RingBuffer(10)
    rb.append(b"abc")
    rb.append(b"def")
    assert rb.snapshot() == b"abcdef"
    assert rb.truncated is False


def test_ringbuffer_drops_oldest_over_capacity():
    rb = RingBuffer(4)
    rb.append(b"abcdef")          # 6 bytes into a 4-byte buffer
    assert rb.snapshot() == b"cdef"
    assert rb.truncated is True




class FakeBridge:
    """Implements the bridge contract PtySession depends on."""

    def __init__(self, chunks, *, write_result=True):
        self._chunks = list(chunks)   # bytes; b"" = idle tick; None = EOF
        self.written = bytearray()
        self.write_result = write_result
        self.closed = False
        self.resized = None

    def read(self, timeout):
        if not self._chunks:
            return b""                # idle
        return self._chunks.pop(0)

    async def write(self, data):
        if self.write_result:
            self.written.extend(data)
        return self.write_result

    def resize(self, cols, rows):
        self.resized = (cols, rows)

    def close(self):
        self.closed = True


class FakeWS:
    def __init__(self):
        self.sent = []               # list of ("bytes"|"text", payload)
        self.close_code = None

    async def send_bytes(self, data):
        self.sent.append(("bytes", bytes(data)))

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def close(self, code=1000, reason=""):
        self.close_code = code


@pytest.mark.asyncio
async def test_attach_replays_buffer_then_streams_live():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"hello ", b"world", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)                      # drain consumes "hello world"
    ws = FakeWS()
    await s.attach(ws)
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"hello world"
    await s.close()


@pytest.mark.asyncio
async def test_reattach_can_force_complete_tui_redraw_after_replay():
    """A fresh terminal cannot reconstruct a differential ANSI tail alone."""
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"partial differential frame", b""])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)

    ws = FakeWS()
    assert await s.attach(ws, force_redraw=True) is True

    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"partial differential frame"
    assert bytes(bridge.written) == b"\x0c"
    await s.close()


@pytest.mark.asyncio
async def test_failed_redraw_marks_session_dead_for_replacement():
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b""], write_result=False)
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()

    assert await s.attach(ws, force_redraw=True) is False
    assert s.alive is False
    await s.close()


@pytest.mark.asyncio
async def test_session_serializes_input_across_socket_tasks():
    from hermes_cli.pty_session import PtySession

    class OrderedBridge(FakeBridge):
        def __init__(self):
            super().__init__([b""])
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def write(self, data):
            if not self.written:
                self.first_started.set()
                await self.release_first.wait()
            self.written.extend(data)
            return True

    bridge = OrderedBridge()
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)

    first = asyncio.create_task(s.write(ws, b"first"))
    await bridge.first_started.wait()
    second = asyncio.create_task(s.write(ws, b"second"))
    await asyncio.sleep(0)
    assert bytes(bridge.written) == b""

    bridge.release_first.set()
    assert await first is True
    assert await second is True
    assert bytes(bridge.written) == b"firstsecond"
    await s.close()


@pytest.mark.asyncio
async def test_superseded_failed_write_does_not_kill_replacement_session():
    from hermes_cli.pty_session import PtySession

    class SupersededBridge(FakeBridge):
        def __init__(self):
            super().__init__([b""])
            self.old_write_started = asyncio.Event()
            self.release_old_write = asyncio.Event()
            self.calls = 0

        async def write(self, data):
            self.calls += 1
            if self.calls == 1:
                self.old_write_started.set()
                await self.release_old_write.wait()
                return False
            self.written.extend(data)
            return True

    bridge = SupersededBridge()
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    old_ws = FakeWS()
    new_ws = FakeWS()
    await s.attach(old_ws)

    old_write = asyncio.create_task(s.write(old_ws, b"old input"))
    await bridge.old_write_started.wait()
    new_attach = asyncio.create_task(s.attach(new_ws, force_redraw=True))
    for _ in range(10):
        if s._ws is new_ws:
            break
        await asyncio.sleep(0)
    assert s._ws is new_ws

    bridge.release_old_write.set()
    assert await old_write is False
    assert await new_attach is True
    assert s.alive is True
    assert await s.write(new_ws, b"new input") is True
    assert bytes(bridge.written) == b"\x0cnew input"
    await s.close()


@pytest.mark.asyncio
async def test_detach_keeps_draining_into_buffer():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"one", b"", b"two"])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    assert s.attached is False
    assert s.last_detached_at is not None
    await asyncio.sleep(0.05)                      # "two" drains while detached
    ws2 = FakeWS()
    await s.attach(ws2)
    replay = b"".join(p for kind, p in ws2.sent if kind == "bytes")
    assert replay == b"onetwo"
    await s.close()


@pytest.mark.asyncio
async def test_eof_marks_dead_and_closes_socket_4410():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"bye", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await asyncio.sleep(0.05)                      # drain hits None (EOF)
    assert s.alive is False
    assert ws.close_code == 4410
    await s.close()


from hermes_cli.pty_session import PtySessionRegistry, RegistryFull


def make_registry(ttl=1800.0, max_sessions=16):
    return PtySessionRegistry(ttl=ttl, max_sessions=max_sessions,
                              buffer_cap=1024, read_timeout=0.01)


@pytest.mark.asyncio
async def test_same_key_reattaches_same_session():
    reg = make_registry()
    b1 = FakeBridge([b"", b"", b""])
    s1, created1 = await reg.attach_or_spawn("tok", spawn=lambda: b1)
    s2, created2 = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created1 is True and created2 is False
    assert s1 is s2
    assert s2.bridge is b1                     # second spawn callable was NOT used
    await reg.close_all()




@pytest.mark.asyncio
async def test_new_key_at_capacity_raises_when_none_reapable():
    reg = make_registry(max_sessions=1)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("a", spawn=lambda: b)
    await s.attach(FakeWS())                    # attached → not reapable
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("b", spawn=lambda: FakeBridge([]))
    await reg.close_all()


@pytest.mark.asyncio
async def test_reaper_loop_invokes_reap(monkeypatch):
    from hermes_cli.pty_session import run_reaper
    reg = make_registry()
    calls = {"n": 0}

    async def fake_reap(now=None):
        calls["n"] += 1

    monkeypatch.setattr(reg, "reap_idle", fake_reap)
    task = asyncio.create_task(run_reaper(reg, interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert calls["n"] >= 2


def _register_idle_session(reg, key):
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b""])
    s = PtySession(key, bridge, buffer_cap=1024, read_timeout=0.01)
    await_start = s.start()
    return bridge, s, await_start


@pytest.mark.asyncio
async def test_concurrent_reap_idle_is_idempotent():
    """reap_idle is reached from attach_or_spawn and the run_reaper loop; both
    may doom the same keys, so one reap can pop a key the other already took
    while awaiting its close(). The second pop must skip, not raise."""
    reg = make_registry(ttl=60.0)
    bridges = []
    sessions = []
    for i in range(2):
        bridge, s, start = _register_idle_session(reg, "k%d" % i)
        await start
        s.detach(None)                        # unattached, last_detached_at set
        reg._sessions[s.key] = s
        bridges.append(bridge)
        sessions.append(s)

    entered, release = asyncio.Event(), asyncio.Event()
    original_close = sessions[0].close

    async def gated_close():
        entered.set()
        await release.wait()
        await original_close()

    sessions[0].close = gated_close

    far_future = time.monotonic() + 10_000    # both idle past ttl → doomed
    first = asyncio.create_task(reg.reap_idle(now=far_future))
    await entered.wait()                      # k0 popped; first reap parked in close()
    await reg.reap_idle(now=far_future)       # second reap hits the taken k0

    release.set()
    await first
    assert not reg._sessions
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_close_all_survives_key_popped_by_concurrent_reap():
    """close_all snapshots keys, then awaits each close(); a reap that runs
    during that await can remove a later key from the snapshot."""
    reg = make_registry(ttl=60.0)
    bridges = []
    sessions = []
    for i in range(2):
        bridge, s, start = _register_idle_session(reg, "k%d" % i)
        await start
        s.detach(None)
        reg._sessions[s.key] = s
        bridges.append(bridge)
        sessions.append(s)

    entered, release = asyncio.Event(), asyncio.Event()
    original_close = sessions[0].close

    async def gated_close():
        entered.set()
        await release.wait()
        await original_close()

    sessions[0].close = gated_close

    closer = asyncio.create_task(reg.close_all())
    await entered.wait()                      # close_all popped k0, parked in close()
    await reg.reap_idle(now=time.monotonic() + 10_000)   # pops k1 meanwhile
    release.set()
    await closer                              # k1 of the snapshot is already gone

    assert not reg._sessions
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_reap_idle_closes_doomed_and_keeps_attached():
    reg = make_registry(ttl=60.0)
    live_bridge = FakeBridge([b""])
    s_live, _ = await reg.attach_or_spawn("live", spawn=lambda: live_bridge)
    await s_live.attach(FakeWS())             # attached → survives the reap

    dead_bridge = FakeBridge([b""])
    s_dead, _ = await reg.attach_or_spawn("dead", spawn=lambda: dead_bridge)
    s_dead.alive = False                      # dead remnant → reaped

    idle_bridge = FakeBridge([b""])
    s_idle, _ = await reg.attach_or_spawn("idle", spawn=lambda: idle_bridge)
    s_idle.detach(None)                       # idle past ttl → reaped

    await reg.reap_idle(now=time.monotonic() + 10_000)
    assert list(reg._sessions) == ["live"]
    assert dead_bridge.closed and idle_bridge.closed and not live_bridge.closed
    await reg.close_all()
