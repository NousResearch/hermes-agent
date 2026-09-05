import asyncio
import json
import queue
import threading
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


def test_ringbuffer_snapshot_from_rejects_future_offset():
    rb = RingBuffer(10)
    rb.append(b"abc")
    assert rb.snapshot_from(4) is None




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


class QueueBridge(FakeBridge):
    """Thread-safe bridge whose output can be injected during an async test."""

    def __init__(self):
        super().__init__([])
        self._queue = queue.Queue()
        self.read_started = threading.Event()

    def push(self, data):
        self._queue.put(data)

    def read(self, timeout):
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""
        self.read_started.set()
        return chunk


class BlockingFirstBinaryWS(FakeWS):
    def __init__(self):
        super().__init__()
        self.first_binary_started = asyncio.Event()
        self.release_first_binary = asyncio.Event()
        self.live_sent = asyncio.Event()
        self._binary_count = 0

    async def send_bytes(self, data):
        self._binary_count += 1
        if self._binary_count == 1:
            self.first_binary_started.set()
            await self.release_first_binary.wait()
        await super().send_bytes(data)
        if data == b"live":
            self.live_sent.set()


class FailingControlWS(FakeWS):
    async def send_text(self, text):
        raise RuntimeError("disconnected during attach")


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
    control = json.loads(next(p for kind, p in ws.sent if kind == "text"))
    assert control == {
        "type": "pty.replay",
        "epoch": s.epoch,
        "start_offset": 0,
        "replay_end_offset": len(b"hello world"),
        "reset": True,
        "reason": "initial",
    }
    await s.close()


@pytest.mark.asyncio
async def test_attach_serializes_chunked_replay_before_live_output():
    """A live frame cannot cut between chunks of a large replay."""
    from hermes_cli.pty_session import PtySession

    replay = b"r" * (16384 + 97)
    bridge = QueueBridge()
    s = PtySession("k", bridge, buffer_cap=len(replay) + 1024, read_timeout=0.01)
    s.buffer.append(replay)
    await s.start()
    ws = BlockingFirstBinaryWS()

    attach_task = asyncio.create_task(s.attach(ws))
    await asyncio.wait_for(ws.first_binary_started.wait(), timeout=1)

    # The drain thread reads this while attach() is blocked in the first
    # replay frame. It must then wait on the session send lock.
    bridge.push(b"live")
    assert await asyncio.to_thread(bridge.read_started.wait, 1)
    ws.release_first_binary.set()

    await asyncio.wait_for(attach_task, timeout=1)
    await asyncio.wait_for(ws.live_sent.wait(), timeout=1)
    binary = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert binary == replay + b"live"
    await s.close()


@pytest.mark.asyncio
async def test_attach_send_failure_leaves_session_detached_and_reapable():
    from hermes_cli.pty_session import PtySession

    s = PtySession("k", FakeBridge([]), buffer_cap=1024, read_timeout=0.01)
    ws = FailingControlWS()
    with pytest.raises(RuntimeError, match="disconnected during attach"):
        await s.attach(ws)
    assert s.attached is False
    assert s.last_detached_at is not None
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


@pytest.mark.asyncio
async def test_attach_incremental_replay_from_offset():
    """Reconnect with a prior byte offset sends only the delta, not the full buffer.

    The connection-lifecycle architecture (suspend on hide / resume on show)
    relies on this: when a tab returns, the client reconnects with the offset
    it had before suspend, and the server sends only bytes appended after that
    offset — not the entire 1MB buffer.
    """
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"hello ", b"world", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)  # drain consumes "hello world"

    # First attach: full replay
    ws1 = FakeWS()
    await s.attach(ws1, client_offset=None)
    replay1 = b"".join(p for kind, p in ws1.sent if kind == "bytes")
    assert replay1 == b"hello world"

    # Reconnect with offset = len("hello world") — should send nothing (zero delta)
    ws2 = FakeWS()
    await s.attach(
        ws2,
        client_offset=len(b"hello world"),
        client_epoch=s.epoch,
    )
    replay2 = b"".join(p for kind, p in ws2.sent if kind == "bytes")
    assert replay2 == b""  # no sentinel, no full replay — just return
    control = json.loads(next(p for kind, p in ws2.sent if kind == "text"))
    assert control["reset"] is False
    assert control["reason"] == "resume"

    await s.close()


@pytest.mark.asyncio
async def test_attach_rolled_out_offset_resets_with_retained_snapshot():
    """A gap is explicit and never silently discards retained bytes."""
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"a" * 2000, None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)  # drain consumes 2000 bytes (cap=1024, so oldest evicted)

    # Reconnect with offset=100 (rolled out — earliest available is 2000-1024=976)
    ws = FakeWS()
    await s.attach(ws, client_offset=100, client_epoch=s.epoch)
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    control = json.loads(next(p for kind, p in ws.sent if kind == "text"))
    assert control["reset"] is True
    assert control["reason"] == "offset_rolled_out"
    assert control["start_offset"] == 976
    assert replay == b"a" * 1024

    await s.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_offset", "client_epoch", "reason"),
    [
        (2001, "current", "offset_ahead"),
        (2000, "different-epoch", "epoch_mismatch"),
    ],
)
async def test_attach_invalid_cursor_resets_with_retained_snapshot(
    client_offset, client_epoch, reason
):
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"retained", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)

    if client_epoch == "current":
        client_epoch = s.epoch
    ws = FakeWS()
    await s.attach(ws, client_offset=client_offset, client_epoch=client_epoch)
    control = json.loads(next(p for kind, p in ws.sent if kind == "text"))
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert control["reset"] is True
    assert control["reason"] == reason
    assert replay == b"retained"

    await s.close()
