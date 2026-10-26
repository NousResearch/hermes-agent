import asyncio
import os
import time

import pytest

from hermes_cli.pty_session import RingBuffer
from tests.waiting import async_wait_until


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
        self.dead = False

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

    def is_alive(self):
        return not self.dead


class FakeWS:
    def __init__(self):
        self.sent = []               # list of ("bytes"|"text", payload)
        self.close_code = None
        self.close_reason = None

    async def send_bytes(self, data):
        self.sent.append(("bytes", bytes(data)))

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def close(self, code=1000, reason=""):
        self.close_code = code
        self.close_reason = reason


class FailingWS(FakeWS):
    """A socket whose client already dropped: every send raises, optionally after a hold."""

    def __init__(self, *, hold=False):
        super().__init__()
        self.send_started = asyncio.Event()
        self.release = asyncio.Event()
        self._hold = hold

    async def send_bytes(self, data):
        self.send_started.set()
        if self._hold:
            await self.release.wait()
        raise RuntimeError("socket closed")


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
async def test_failed_replay_detaches_session_so_reaper_reclaims_it():
    """A client dropping mid-replay must not pin the PTY as attached forever (#110849)."""
    from hermes_cli.pty_session import PtySessionRegistry

    reg = PtySessionRegistry(ttl=0.0, max_sessions=2, buffer_cap=1024, read_timeout=0.01)
    bridge = FakeBridge([b""])
    s, _ = await reg.attach_or_spawn("k", spawn=lambda: bridge)
    s.buffer.append(b"replay")

    assert await s.attach(FailingWS()) is False

    assert s.attached is False
    assert s.last_detached_at is not None
    await reg.reap_idle(now=time.monotonic() + 1.0)
    assert "k" not in reg._sessions
    assert bridge.closed


@pytest.mark.asyncio
async def test_drain_send_failure_detaches_current_socket_but_not_a_replacement():
    from hermes_cli.pty_session import PtySession

    s = PtySession("k", FakeBridge([b"live"]), buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FailingWS()
    await s.attach(ws)
    await asyncio.wait_for(ws.send_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert s.attached is False
    assert s.last_detached_at is not None
    await s.close()

    # The failure of a socket superseded mid-send must leave the new viewer attached.
    s = PtySession("k", FakeBridge([b"live"]), buffer_cap=1024, read_timeout=0.01)
    await s.start()
    stale = FailingWS(hold=True)
    await s.attach(stale)
    await asyncio.wait_for(stale.send_started.wait(), timeout=1)
    replacement = FakeWS()
    assert await s.attach(replacement) is True
    stale.release.set()
    await asyncio.sleep(0.05)
    assert s.attached is True
    assert s._ws is replacement
    assert s.last_detached_at is None
    await s.close()


@pytest.mark.asyncio
async def test_viewer_that_stops_reading_live_output_is_told_to_reconnect(monkeypatch):
    """The send limit drops a stalled viewer, not its session. 1011 makes the
    dashboard chat tab offer a new session; 1001 is the code both it and the
    Webapp terminal redial on, and the same key then reattaches the same PTY."""
    from hermes_cli import pty_session
    from hermes_cli.pty_session import PtySessionRegistry

    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.1)
    registry = PtySessionRegistry(ttl=60.0, max_sessions=2, buffer_cap=1024, read_timeout=0.01)
    bridge = FakeBridge([])
    session, _ = await registry.attach_or_spawn("k", spawn=lambda: bridge)
    stalled = FailingWS(hold=True)  # stopped reading: its send never completes
    assert await session.attach(stalled)  # empty history: no replay frame to send
    bridge._chunks.append(b"live output")
    await async_wait_until(lambda: stalled.close_code is not None, timeout=15)
    assert stalled.close_code == 1001
    assert session.alive and not session.attached
    again, created = await registry.attach_or_spawn("k", spawn=lambda: FakeBridge([]))
    viewer = FakeWS()
    assert again is session and not created
    assert await again.attach(viewer)
    assert viewer.sent == [("bytes", b"live output")]
    await registry.close_all()


class SlowLinkWS(FakeWS):
    """Carries every frame in ``seconds``: a viewer still reading, over a slow link."""

    def __init__(self, seconds):
        super().__init__()
        self._seconds = seconds

    async def send_bytes(self, data):
        await asyncio.sleep(self._seconds)  # health: allow HX013 -- the link's transfer time is the behaviour under test
        await super().send_bytes(data)


@pytest.mark.asyncio
async def test_a_full_replay_over_a_slow_link_is_not_a_stalled_viewer(monkeypatch):
    """The stall limit covers one live frame. A reattach replay of up to the 1 MiB buffer gets time
    to cross a slow link too, or every redial replays, times out and redials again."""
    from hermes_cli import pty_session
    from hermes_cli.pty_session import PtySession

    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.05)
    history = b"x" * (1024 * 1024)
    session = PtySession("k", FakeBridge([]), buffer_cap=len(history), read_timeout=0.01)
    session.buffer.append(history)
    viewer = SlowLinkWS(0.5)  # ten stall limits, far inside a full replay's budget

    assert await session.attach(viewer)
    assert viewer.sent == [("bytes", history)] and viewer.close_code is None
    assert session._ws is viewer

    # A viewer that stopped reading still gets dropped, within the limit plus its replay's share.
    small = PtySession("k2", FakeBridge([]), buffer_cap=1024, read_timeout=0.01)
    small.buffer.append(b"tail")
    stalled = FailingWS(hold=True)  # the replay never completes
    try:
        assert await small.attach(stalled) is False
        assert small._ws is None
    finally:
        stalled.release.set()


class NarrowLinkWS(FakeWS):
    """Carries a frame of up to ``capacity`` bytes within the deadline and never finishes a larger
    one: a viewer still reading, over a link too slow for the whole replay."""

    def __init__(self, capacity):
        super().__init__()
        self._capacity = capacity

    async def send_bytes(self, data):
        if len(data) > self._capacity:
            await asyncio.Event().wait()
        await super().send_bytes(data)


@pytest.mark.asyncio
async def test_redials_over_a_link_too_slow_for_the_whole_replay_reach_a_replay_it_carries(monkeypatch):
    """A replay that misses its deadline is halved for the session's next attach, so a viewer on a
    link of any speed gets the newest output it can carry after a few redials instead of timing out
    on every one. A client that dropped is not a slow link and keeps the whole replay."""
    from hermes_cli import pty_session
    from hermes_cli.pty_session import PtySession

    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pty_session, "_REPLAY_MIN_BYTES_PER_SECOND", 1024 ** 3)
    monkeypatch.setattr(pty_session, "_REPLAY_FLOOR_BYTES", 1024, raising=False)
    history = bytes(range(256)) * 256
    session = PtySession("k", FakeBridge([]), buffer_cap=len(history), read_timeout=0.01)
    session.buffer.append(history)

    assert await session.attach(FailingWS()) is False
    whole = FakeWS()
    assert await session.attach(whole)
    assert whole.sent == [("bytes", history)]

    attached = []
    for _redial in range(12):
        viewer = NarrowLinkWS(4096)
        attached.append(await session.attach(viewer))
        if attached[-1]:
            break
    assert attached[-1], attached
    replay = viewer.sent[0][1]
    assert 0 < len(replay) <= 4096 and history.endswith(replay)
    assert session._ws is viewer and viewer.close_code is None


@pytest.mark.asyncio
async def test_a_shrunk_replay_keeps_a_floor(monkeypatch):
    """Replays that keep timing out shrink the next one only down to a floor, so a reattach still
    shows the newest output instead of nothing."""
    from hermes_cli import pty_session
    from hermes_cli.pty_session import PtySession

    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pty_session, "_REPLAY_MIN_BYTES_PER_SECOND", 1024 ** 3)
    monkeypatch.setattr(pty_session, "_REPLAY_FLOOR_BYTES", 1024, raising=False)
    history = bytes(range(256)) * 256
    session = PtySession("k", FakeBridge([]), buffer_cap=len(history), read_timeout=0.01)
    session.buffer.append(history)

    for _stall in range(20):
        await session.attach(NarrowLinkWS(0))  # never finishes a frame of output

    viewer = FakeWS()
    assert await session.attach(viewer)
    assert viewer.sent == [("bytes", history[-1024:])]


@pytest.mark.asyncio
async def test_a_replay_size_that_timed_out_is_not_replayed_again(monkeypatch):
    """A replay send returns once the socket's buffers take the frame, so a quick send does not show
    that the link carries more. After a replay times out, the session never replays that size again:
    a viewer that keeps redialing over the same slow link gets the replay it carries every time,
    instead of timing out on every second redial."""
    from hermes_cli import pty_session
    from hermes_cli.pty_session import PtySession

    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pty_session, "_REPLAY_MIN_BYTES_PER_SECOND", 1024 ** 3)
    monkeypatch.setattr(pty_session, "_REPLAY_FLOOR_BYTES", 1024)
    history = bytes(range(256)) * 256
    session = PtySession("k", FakeBridge([]), buffer_cap=len(history), read_timeout=0.01)
    session.buffer.append(history)

    attached, replays = [], []
    for _redial in range(6):
        # The buffers take half the history at once; the whole of it never crosses in time.
        viewer = NarrowLinkWS(len(history) // 2)
        attached.append(await session.attach(viewer))
        replays.append(viewer.sent[0][1] if viewer.sent else b"")

    assert attached == [False] + [True] * 5
    assert replays[1:] == [history[len(history) // 2:]] * 5


@pytest.mark.asyncio
async def test_a_reattach_that_replays_part_of_the_history_says_output_was_trimmed(tmp_path, monkeypatch):
    """The host terminal's metadata frame tells the browser ``truncated: true`` whenever the replay
    after it leaves out older output, including output the session still holds but a shrunk replay
    omits, so the "output was trimmed" notice shows. A replay of everything says ``false``."""
    import json
    import secrets
    from types import SimpleNamespace

    from hermes_cli import pty_session
    from hermes_cli.web_host_terminal import META_PREFIX
    from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry, host_terminal

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pty_session, "_VIEWER_SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pty_session, "_REPLAY_MIN_BYTES_PER_SECOND", 1024 ** 3)
    history = bytes(range(256)) * 256
    registry = HostTerminalRegistry()
    token = secrets.token_urlsafe(32)
    session = pty_session.PtySession(token, FakeBridge([]), buffer_cap=1024 * 1024, read_timeout=0.01)
    session.buffer.append(history)
    registry._sessions[token] = session
    registry.owners[token] = HostOwner(("local", "owner"), home.resolve(), "sh", str(tmp_path))
    app = SimpleNamespace(state=SimpleNamespace(host_terminals=registry))

    class Viewer(NarrowLinkWS):
        """A browser tab reattaching to the shell, then closing."""

        def __init__(self, capacity):
            super().__init__(capacity)
            self.app = app
            self.query_params = {"attach": token}
            self._hermes_auth_identity = {"user_id": "owner", "provider": "local"}

        async def receive(self):
            return {"type": "websocket.disconnect"}

    def metadata(viewer):
        kind, text = viewer.sent[0]
        assert kind == "text" and text.startswith(META_PREFIX)
        return json.loads(text.removeprefix(META_PREFIX))

    try:
        whole = Viewer(len(history))
        await host_terminal(whole)
        assert whole.sent[1] == ("bytes", history)
        assert metadata(whole)["truncated"] is False

        stalled = Viewer(0)
        await host_terminal(stalled)
        assert stalled.close_code == 1001 and metadata(stalled)["truncated"] is False

        partial = Viewer(len(history))
        await host_terminal(partial)
        kind, replay = partial.sent[1]
        assert kind == "bytes" and len(replay) < len(history) and history.endswith(replay)
        assert metadata(partial)["truncated"] is True
    finally:
        await registry.close_all()


@pytest.mark.asyncio
async def test_a_shell_reaped_while_its_redial_is_validated_closes_the_redial_as_an_exit(tmp_path, monkeypatch):
    """A redial whose shell exited, and which the reaper takes while ``resolve`` checks the
    profile off-loop, is still told the shell exited, so the tab closes instead of offering a new
    shell."""
    import secrets
    import threading
    from types import SimpleNamespace

    from hermes_cli import pty_session, web_host_terminal_sessions
    from hermes_cli.web_host_terminal_sessions import HostOwner, HostTerminalRegistry, TerminalExited

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    identity = ("local", "owner")
    registry = HostTerminalRegistry()
    token = secrets.token_urlsafe(32)
    bridge = FakeBridge([])
    bridge.dead = True  # the shell exited while no tab was attached
    registry._sessions[token] = pty_session.PtySession(token, bridge, buffer_cap=1024, read_timeout=0.01)
    registry.owners[token] = HostOwner(identity, home.resolve(), "sh", str(tmp_path))

    validating, release = threading.Event(), threading.Event()
    request_home = web_host_terminal_sessions._request_home

    def held_request_home(profile):
        validating.set()
        release.wait(5)
        return request_home(profile)

    monkeypatch.setattr(web_host_terminal_sessions, "_request_home", held_request_home)
    redial = SimpleNamespace(query_params={"attach": token})
    try:
        resolving = asyncio.ensure_future(registry.resolve(token, redial, identity))
        assert await asyncio.to_thread(validating.wait, 5)
        await registry.reap_idle()
        assert token not in registry._sessions and token in registry.exits
        release.set()
        with pytest.raises(TerminalExited):
            await resolving
    finally:
        release.set()
        await registry.close_all()


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
async def test_concurrent_attach_on_one_token_forks_one_pty():
    """Two connections racing one attach token must share ONE registered PTY.

    The get-or-spawn decision spans awaits (reap, the spawn thread, start()), so
    both racing callers used to see "no session" and fork their own: the token
    then mapped to whichever registered last, the other tab's live session fell
    out of the registry (never reaped) and a reattach landed on the wrong
    terminal (#115304).
    """
    from hermes_cli.pty_session import WS_CLOSE_SUPERSEDED

    reg = make_registry()
    spawned = []

    def spawn():
        bridge = FakeBridge([b"", b""])
        spawned.append(bridge)
        return bridge

    (s1, created1), (s2, created2) = await asyncio.gather(
        reg.attach_or_spawn("tok", spawn=spawn),
        reg.attach_or_spawn("tok", spawn=spawn),
    )

    assert len(spawned) == 1                    # one token, one PTY
    assert (s1, created1) == (s2, True)
    assert created2 is False
    assert s1.bridge is spawned[0]
    assert list(reg._sessions.values()) == [s1]  # every handed-out session is tracked

    # Whichever socket attached last owns the terminal; the loser is superseded
    # by contract, so no viewer is left writing into an untracked PTY.
    ws_a, ws_b = FakeWS(), FakeWS()
    await s1.attach(ws_a)
    await s2.attach(ws_b)
    assert reg._sessions["tok"] is s1
    assert s1._ws is ws_b and ws_b.close_code is None
    assert ws_a.close_code == WS_CLOSE_SUPERSEDED
    await reg.close_all()


async def _two_idle_sessions_first_close_gated(reg):
    """Two detached (idle) sessions; k0's close() parks until ``release`` is set."""
    from hermes_cli.pty_session import PtySession
    bridges = []
    for i in range(2):
        bridge = FakeBridge([b""])
        s = PtySession(f'k{i:d}', bridge, buffer_cap=1024, read_timeout=0.01)
        await s.start()
        s.detach(None)                        # unattached, last_detached_at set
        reg._sessions[s.key] = s
        bridges.append(bridge)

    entered, release = asyncio.Event(), asyncio.Event()
    k0 = reg._sessions["k0"]
    original_close = k0.close

    async def gated_close():
        entered.set()
        await release.wait()
        await original_close()

    k0.close = gated_close
    return bridges, entered, release


@pytest.mark.asyncio
async def test_concurrent_reap_idle_is_idempotent():
    """reap_idle is reached from attach_or_spawn and the run_reaper loop; both
    may doom the same keys, so one reap can pop a key the other already took
    while awaiting its close(). The second pop must skip, not raise."""
    reg = make_registry(ttl=60.0)
    bridges, entered, release = await _two_idle_sessions_first_close_gated(reg)

    far_future = time.monotonic() + 10_000    # both idle past ttl → doomed
    first = asyncio.create_task(reg.reap_idle(now=far_future))
    await entered.wait()                      # k0 popped; first reap parked in close()
    await reg.reap_idle(now=far_future)       # second reap takes k1

    release.set()
    await first                               # first reap reaches the taken k1
    assert not reg._sessions
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_a_closed_registry_refuses_spawns_until_reopened():
    """``close_all`` (a lifespan shutting down) refuses new terminals, so none outlives it; the
    process-wide registry takes spawns again once a new lifespan reopens it (the gateway falling back
    from the Webapp's listener to its private one in the same process)."""
    reg = make_registry()
    await reg.close_all()
    with pytest.raises(RegistryFull):
        await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))

    reg.reopen()
    session, created = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([b""]))
    assert created and session.alive
    await reg.close_all()


@pytest.mark.asyncio
async def test_close_all_survives_key_popped_by_concurrent_reap():
    """close_all snapshots keys, then awaits each close(); a reap that runs
    during that await can remove a later key from the snapshot."""
    reg = make_registry(ttl=60.0)
    bridges, entered, release = await _two_idle_sessions_first_close_gated(reg)

    closer = asyncio.create_task(reg.close_all())
    await entered.wait()                      # close_all popped k0, parked in close()
    await reg.reap_idle(now=time.monotonic() + 10_000)   # pops k1 meanwhile
    release.set()
    await closer                              # k1 of the snapshot is already gone

    assert not reg._sessions
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_close_all_does_not_wait_on_one_slow_close_before_the_next():
    """Each close() can wait out its helpers' SIGHUP grace, and the backend's teardown runs under
    a SIGKILL budget, so sessions close concurrently rather than one after another."""
    reg = make_registry(ttl=60.0)
    bridges, entered, release = await _two_idle_sessions_first_close_gated(reg)

    closer = asyncio.create_task(reg.close_all())
    await entered.wait()                      # k0's close() is parked
    for _ in range(50):
        if bridges[1].closed:
            break
        await asyncio.sleep(0.01)
    assert bridges[1].closed                  # k1 closed while k0 still waits
    release.set()
    await closer
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_close_all_waits_for_closes_already_running_in_the_background():
    """A dead remnant or evicted session leaves the registry before its close() finishes; the
    backend's teardown must still wait for it, or its helpers outlive the backend."""
    reg = make_registry(ttl=60.0)
    bridges, entered, release = await _two_idle_sessions_first_close_gated(reg)
    reg._close_in_background(reg._sessions.pop("k0"))
    await entered.wait()                      # k0 is closing outside the registry

    closer = asyncio.create_task(reg.close_all())
    await asyncio.sleep(0.05)
    assert not closer.done()                  # still waiting on k0
    release.set()
    await closer
    assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_close_other_sessions_removes_old_profile_session():
    from hermes_cli.pty_session import WS_CLOSE_SUPERSEDED, PtySession

    reg = make_registry()
    old_bridge = FakeBridge([b""])
    current_bridge = FakeBridge([b""])
    old = PtySession("token\0alpha\0session-a", old_bridge, buffer_cap=1024, read_timeout=0.01)
    current = PtySession("token\0beta\0session-b", current_bridge, buffer_cap=1024, read_timeout=0.01)
    await old.start()
    await current.start()
    # A sibling tab (same attach token, other profile) is still viewing the old PTY.
    old_ws = FakeWS()
    assert await old.attach(old_ws)
    reg._sessions[old.key] = old
    reg._sessions[current.key] = current

    await reg.close_other_sessions("token", keep_key=current.key)

    assert old_bridge.closed
    assert old.key not in reg._sessions
    assert reg._sessions[current.key] is current
    # The displaced viewer gets the documented supersede code rather than going silent.
    assert old_ws.close_code == WS_CLOSE_SUPERSEDED
    await reg.close_all()
@pytest.mark.asyncio
async def test_close_orphaned_sessions_closes_pty_stranded_by_token_rotation():
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    stranded_bridge = FakeBridge([b""])
    current_bridge = FakeBridge([b""])
    stranded = PtySession("old-token\0\0session-a", stranded_bridge, buffer_cap=1024, read_timeout=0.01)
    current = PtySession("new-token\0\0session-a", current_bridge, buffer_cap=1024, read_timeout=0.01)
    await stranded.start()
    await current.start()
    stranded.detach(None)                 # New chat rotated the token and walked away
    reg._sessions[stranded.key] = stranded
    reg._sessions[current.key] = current

    await reg.close_orphaned_sessions("session-a", keep_key=current.key)

    assert stranded_bridge.closed
    assert stranded.key not in reg._sessions
    assert reg._sessions[current.key] is current
    await reg.close_all()


@pytest.mark.asyncio
async def test_close_orphaned_sessions_keeps_a_pty_someone_is_viewing():
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    viewed_bridge = FakeBridge([b""])
    current_bridge = FakeBridge([b""])
    viewed = PtySession("old-token\0\0session-a", viewed_bridge, buffer_cap=1024, read_timeout=0.01)
    current = PtySession("new-token\0\0session-a", current_bridge, buffer_cap=1024, read_timeout=0.01)
    await viewed.start()
    await current.start()
    assert await viewed.attach(FakeWS())  # a second tab is still watching that terminal
    reg._sessions[viewed.key] = viewed
    reg._sessions[current.key] = current

    await reg.close_orphaned_sessions("session-a", keep_key=current.key)

    assert not viewed_bridge.closed
    assert reg._sessions[viewed.key] is viewed
    assert reg._sessions[current.key] is current
    await reg.close_all()


@pytest.mark.asyncio
async def test_close_orphaned_sessions_without_resume_target_touches_nothing():
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    other_bridge = FakeBridge([b""])
    other = PtySession("old-token\0\0", other_bridge, buffer_cap=1024, read_timeout=0.01)
    await other.start()
    other.detach(None)
    reg._sessions[other.key] = other

    await reg.close_orphaned_sessions(None, keep_key="new-token\0\0")

    assert not other_bridge.closed
    assert reg._sessions[other.key] is other
    await reg.close_all()


@pytest.mark.parametrize(
    "key,expected",
    [
        ("token\0profile\0session-a", ("profile", "session-a")),
        ("token\0\0session-b", ("", "session-b")),
        ("token\0profile\0", ("profile", "")),
        ("token", ("", "")),
    ],
)
def test_key_segments_read_profile_and_resume(key, expected):
    from hermes_cli.pty_session import _key_segments

    assert _key_segments(key) == expected


@pytest.mark.asyncio
async def test_close_orphaned_sessions_leaves_the_same_session_id_in_another_profile():
    """Session ids are per profile store: alpha's ``session-a`` is not beta's."""
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    other_bridge = FakeBridge([b""])
    current_bridge = FakeBridge([b""])
    other = PtySession("old-token\0beta\0session-a", other_bridge, buffer_cap=1024, read_timeout=0.01)
    current = PtySession("new-token\0alpha\0session-a", current_bridge, buffer_cap=1024, read_timeout=0.01)
    await other.start()
    await current.start()
    other.detach(None)                    # beta's tab is mid-reconnect, not stranded
    reg._sessions[other.key] = other
    reg._sessions[current.key] = current

    await reg.close_orphaned_sessions("session-a", keep_key=current.key)

    assert not other_bridge.closed
    assert reg._sessions[other.key] is other
    await reg.close_all()


class PidBridge(FakeBridge):
    """A bridge that reports its leader pid, like a real PTY does."""

    def __init__(self, chunks, pid):
        super().__init__(chunks)
        self.pid = pid


@pytest.fixture
def process_tree(monkeypatch):
    """Fake ancestry: ``{pid: {ancestor pids}}``. The dashboard PTY leader (node) is pid 4242;
    the tui_gateway child that actually holds the session lease is 4243 underneath it."""
    import hermes_cli.pty_session as mod

    tree = {4243: {4242, 1}, 5152: {5151, 1}}
    monkeypatch.setattr(mod, "_process_ancestors", lambda pid: tree.get(pid, set()))
    return tree


@pytest.mark.asyncio
async def test_close_orphaned_sessions_finds_the_lease_holder_without_a_resume_key(process_tree):
    """The reported repro starts a *fresh* chat in the default profile, so chat A's key is
    the bare token — no profile, no resume segment. Only the process holding the session
    lease identifies it, and that process is the PTY leader's CHILD, not the leader."""
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    fresh_a = PtySession("old-token", PidBridge([b""], 4242), buffer_cap=1024, read_timeout=0.01)
    unrelated = PtySession("other-token", PidBridge([b""], 5151), buffer_cap=1024, read_timeout=0.01)
    current = PtySession("new-token\0\0session-a", FakeBridge([b""]), buffer_cap=1024, read_timeout=0.01)
    for session in (fresh_a, unrelated, current):
        await session.start()
        session.detach(None)
        reg._sessions[session.key] = session

    await reg.close_orphaned_sessions("session-a", keep_key=current.key, holder_pid=4243)

    assert fresh_a.bridge.closed and fresh_a.key not in reg._sessions
    # A detached terminal for a different session is somebody's reconnect
    # window, not this resume's business.
    assert not unrelated.bridge.closed and reg._sessions[unrelated.key] is unrelated
    assert reg._sessions[current.key] is current


@pytest.mark.asyncio
async def test_close_orphaned_sessions_leaves_a_holder_outside_every_pty(process_tree):
    """A lease held by a plain terminal (`hermes` in a shell, pid 9000, no PTY ancestor) is a
    real other window: nothing is closed and the single-writer refusal stands."""
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    detached = PtySession("old-token", PidBridge([b""], 4242), buffer_cap=1024, read_timeout=0.01)
    await detached.start()
    detached.detach(None)
    reg._sessions[detached.key] = detached

    await reg.close_orphaned_sessions("session-a", keep_key="new-token\0\0session-a", holder_pid=9000)

    assert not detached.bridge.closed and reg._sessions[detached.key] is detached
    await reg.close_all()


@pytest.mark.platforms("posix")
def test_hosts_pid_sees_the_lease_holding_grandchild_of_a_real_pty():
    """Live topology: ptyprocess forks the leader; the leader's child is what claims the lease."""
    from hermes_cli.pty_bridge import PtyBridge
    from hermes_cli.pty_session import PtySession

    bridge = PtyBridge.spawn(
        ["sh", "-c", "python3 -c 'import os,time;print(\"CHILD\",os.getpid(),flush=True);time.sleep(20)'; true"]
    )
    try:
        out, deadline = b"", time.monotonic() + 10
        while b"CHILD" not in out and time.monotonic() < deadline:
            out += bridge.read(timeout=0.2) or b""
        grandchild = int(out.split(b"CHILD", 1)[1].split()[0])
        session = PtySession("tok", bridge, buffer_cap=8, read_timeout=0.01)
        assert grandchild != bridge.pid
        assert session.hosts_pid(grandchild)
        assert session.hosts_pid(bridge.pid)
        assert not session.hosts_pid(os.getpid())
    finally:
        bridge.close()


@pytest.mark.asyncio
async def test_reap_reaps_dead_process_even_when_attached():
    # Child killed externally (OOM killer / cgroup SIGKILL) while a grandchild
    # holds the PTY slave, so the drain never sees EOF: the session must still
    # be reaped, otherwise its bridge and registry slot leak forever (#76759).
    reg = make_registry(ttl=3600.0)
    b = FakeBridge([b"", b"", b""])
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    await s.attach(FakeWS())
    b.dead = True
    await reg.reap_idle()
    assert "tok" not in reg._sessions
    assert b.closed is True


    await reg.close_all()


@pytest.mark.asyncio
async def test_close_orphaned_sessions_leaves_a_lease_holding_pty_that_is_viewed(process_tree):
    from hermes_cli.pty_session import PtySession

    reg = make_registry()
    watched = PtySession("old-token", PidBridge([b""], 4242), buffer_cap=1024, read_timeout=0.01)
    current = PtySession("new-token\0\0session-a", FakeBridge([b""]), buffer_cap=1024, read_timeout=0.01)
    await watched.start()
    await current.start()
    assert await watched.attach(FakeWS())
    reg._sessions[watched.key] = watched
    reg._sessions[current.key] = current

    await reg.close_orphaned_sessions("session-a", keep_key=current.key, holder_pid=4243)

    assert not watched.bridge.closed and reg._sessions[watched.key] is watched
    await reg.close_all()


def test_hosts_pid_tolerates_a_bridge_without_one():
    from hermes_cli.pty_session import PtySession

    session = PtySession("tok", FakeBridge([b""]), buffer_cap=8, read_timeout=0.01)
    assert session.hosts_pid(None) is False
    assert session.hosts_pid(123) is False
    assert PidBridge([b""], 7).pid == 7


@pytest.mark.asyncio
async def test_reap_keeps_live_attached_session():
    # A session with a live child and an attached viewer is never reaped,
    # even past the detached TTL.
    reg = make_registry(ttl=1.0)
    b = FakeBridge([b"", b""])
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    await s.attach(FakeWS())
    await reg.reap_idle(now=time.monotonic() + 10)
    assert "tok" in reg._sessions
    assert b.closed is False
    await reg.close_all()


