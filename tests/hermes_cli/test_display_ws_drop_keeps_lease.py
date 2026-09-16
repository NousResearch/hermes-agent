"""A dropped viewer link (1006: lid closed, Wi-Fi) must NOT hand the screen back to the agent — the
human may be mid-login on it. Only a clean close (1000/1001) releases."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from hermes_cli.web_routers import display
from tools.bot_desktop import lease


class _Ws:
    """Just enough of a Starlette WebSocket: one disconnect message with the given close code."""

    def __init__(self, close_code: int):
        self._code = close_code
        self.closed = False

    async def accept(self):
        pass

    async def receive(self):
        await asyncio.sleep(0.05)
        return {"type": "websocket.disconnect", "code": self._code}

    async def send_bytes(self, data):
        pass

    async def close(self, code=1000, reason=""):
        self.closed = True


async def _bridge_once(close_code: int, home: str) -> lease.Lease:
    sock_dir = os.path.join(home, "bot-desktop")
    os.makedirs(sock_dir, exist_ok=True)
    sock = os.path.join(sock_dir, "rfb.sock")

    async def _xvnc(reader, writer):  # a silent framebuffer
        await asyncio.sleep(1)
        writer.close()

    server = await asyncio.start_unix_server(_xvnc, path=sock)
    try:
        info = {"hermes_home": home, "viewer_id": "desk-1"}
        await display._bridge(_Ws(close_code), info)
    finally:
        server.close()
    return lease.get(profile_key=home)


# 1005 (no status code) is what noVNC's code-less socket.close() AND some proxies produce on a drop, so
# the server keeps the lease; the Desktop sends an explicit 1000 when the pane is closed on purpose.
@pytest.mark.parametrize(("close_code", "human_keeps_control"), [(1006, True), (1005, True), (1000, False)])
def test_only_a_clean_viewer_close_hands_the_screen_back(monkeypatch, close_code, human_keeps_control):
    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        after = asyncio.run(_bridge_once(close_code, home))
    lease._reset_for_tests()
    assert (after.holder == lease.HUMAN) is human_keeps_control, after


def test_ex_holder_who_handed_back_is_not_evicted_by_a_later_takeover():
    """desk-1 holds, hands back to the agent, then desk-2 takes over: desk-1 is a plain watcher again
    and must stay connected; only a takeover WHILE desk-1 held (or believed it held) kicks it."""
    held = {"ever": False}
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-1"), "desk-1") is False
    assert display._should_evict(held, lease.Lease(holder=lease.AGENT), "desk-1") is False
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-2"), "desk-1") is False

    held = {"ever": False}
    display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-1"), "desk-1")
    assert display._should_evict(held, lease.Lease(holder=lease.HUMAN, viewer_id="desk-2"), "desk-1") is True


class _OpenWs(_Ws):
    """Stays open until ``finish`` is set, then reports a clean close."""

    def __init__(self):
        super().__init__(1000)
        self.finish = asyncio.Event()

    async def receive(self):
        await self.finish.wait()
        return {"type": "websocket.disconnect", "code": self._code}


def test_a_takeover_made_by_another_process_stops_input_within_the_refresh_interval(monkeypatch):
    """The bridge caches the input decision instead of reading lease.json per message; a takeover
    written by ANOTHER process (no in-process listener fires) must still be seen quickly."""
    from tools.bot_desktop import rfb_filter
    captured = {}

    class _Filter(rfb_filter.RfbClientFilter):
        def __init__(self, allow_input):
            super().__init__(allow_input)
            captured["allow"] = allow_input
    monkeypatch.setattr(rfb_filter, "RfbClientFilter", _Filter)

    async def _run(home: str) -> float:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)
        server = await asyncio.start_unix_server(lambda r, w: None, path=os.path.join(sock_dir, "rfb.sock"))
        ws = _OpenWs()
        task = asyncio.create_task(display._bridge(ws, {"hermes_home": home, "viewer_id": "desk-1"}))
        try:
            while "allow" not in captured:
                await asyncio.sleep(0.01)
            assert captured["allow"]() is True
            # Another process takes over: the file changes, no listener in this process is told.
            lease._write(lease._path(home), lease.Lease(holder=lease.HUMAN, viewer_id="desk-2", epoch=2))
            t0 = asyncio.get_running_loop().time()
            while captured["allow"]() and asyncio.get_running_loop().time() - t0 < 2.0:
                await asyncio.sleep(0.02)
            return asyncio.get_running_loop().time() - t0
        finally:
            ws.finish.set()
            await task
            server.close()

    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        elapsed = asyncio.run(_run(home))
    lease._reset_for_tests()
    assert elapsed < 0.5, elapsed


def test_no_bridge_task_or_socket_outlives_the_bridge():
    """The pumps were cancelled but never awaited and the Xvnc writer closed without wait_closed, so
    a returned _bridge still had tasks (and a half-closed socket) in flight on the loop; a viewer
    reconnecting in a tight loop accumulated them. Everything the bridge started is finished, and
    Xvnc has seen EOF, when it returns."""
    async def _run(home: str) -> set:
        sock_dir = os.path.join(home, "bot-desktop")
        os.makedirs(sock_dir, exist_ok=True)
        xvnc_saw_eof = asyncio.Event()

        async def _xvnc(reader, writer):
            await reader.read()  # until the bridge closes its end
            xvnc_saw_eof.set()

        server = await asyncio.start_unix_server(_xvnc, path=os.path.join(sock_dir, "rfb.sock"))
        try:
            await display._bridge(_Ws(1000), {"hermes_home": home, "viewer_id": "desk-1"})
            leftover = {t for t in asyncio.all_tasks()
                        if not t.done() and t.get_coro().__qualname__.startswith("_bridge.")}
            await asyncio.wait_for(xvnc_saw_eof.wait(), 2)
            return leftover
        finally:
            server.close()

    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        leftover = asyncio.run(_run(home))
    assert leftover == set(), [t.get_coro() for t in leftover]


_HANDSHAKE = b"RFB 003.008\n" + b"\x01" + b"\x01"  # Security None, shared ClientInit
_KEY_A = b"\x04\x01\x00\x00\x00\x00\x00\x61"  # KeyEvent 'a' down
_FBUR = b"\x03\x00" + b"\x00" * 8


class _SeqWs(_Ws):
    """Deliver each message, then report an abnormal disconnect without a timing delay."""

    def __init__(self, messages):
        super().__init__(1006)
        self._messages = iter(messages)

    async def receive(self):
        message = next(self._messages, None)
        if message is None:
            return {"type": "websocket.disconnect", "code": self._code}
        return {"type": "websocket.receive", "bytes": message}


async def _bridge_with_capture(home: str, viewer_id: str, messages) -> bytes:
    """Capture the complete client stream through EOF, and join the fake server handler."""
    sock_dir = os.path.join(home, "bot-desktop")
    os.makedirs(sock_dir, exist_ok=True)
    sock = os.path.join(sock_dir, "rfb.sock")
    before = asyncio.all_tasks()
    connected = asyncio.get_running_loop().create_future()

    async def _capture(reader, writer):
        try:
            return await reader.read()  # EOF proves all bytes were captured, including forbidden input.
        finally:
            writer.close()
            await writer.wait_closed()

    def _accept(reader, writer):
        connected.set_result(asyncio.create_task(_capture(reader, writer)))

    ws = _SeqWs(messages)
    server = await asyncio.start_unix_server(_accept, path=sock)
    try:
        # Bounds only diagnose a hung bridge/handler; elapsed time never decides success.
        await asyncio.wait_for(
            display._bridge(ws, {"hermes_home": home, "viewer_id": viewer_id}), 5
        )
        capture = await asyncio.wait_for(asyncio.shield(connected), 5)
        received = await asyncio.wait_for(asyncio.shield(capture), 5)
        assert ws.closed
        assert not (asyncio.all_tasks() - before), "bridge or capture task survived cleanup"
        return received
    finally:
        server.close()
        await server.wait_closed()
        # Also clean up on an assertion, timeout or unexpected bridge exception.
        pending = asyncio.all_tasks() - before
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        os.unlink(sock)


@pytest.mark.asyncio
async def test_reconnect_after_abnormal_drop_restores_same_viewer_rfb_input():
    """Sequential bridges reuse the retained lease after 1006, without another acquire.

    The holder sends keys again; a later foreign viewer remains view-only. This exercises the
    real bridge, lease and RFB filter with a WebSocket double and a Unix-socket byte sink.
    """
    async def _run(home):
        original = lease.acquire("desk-1", profile_key=home)
        first = await _bridge_with_capture(home, "desk-1", [_HANDSHAKE, _KEY_A])
        assert first == _HANDSHAKE + _KEY_A, "initial holder input was denied"
        assert lease.get(profile_key=home) == original

        again = await _bridge_with_capture(home, "desk-1", [_HANDSHAKE, _KEY_A, _FBUR])
        assert again == _HANDSHAKE + _KEY_A + _FBUR, "reconnected holder input was denied"
        assert lease.get(profile_key=home) == original

        foreign = await _bridge_with_capture(home, "desk-2", [_HANDSHAKE, _KEY_A, _FBUR])
        assert foreign == _HANDSHAKE + _FBUR, "foreign viewer input passed or viewing was blocked"
        assert lease.get(profile_key=home) == original

    lease._reset_for_tests()
    try:
        with tempfile.TemporaryDirectory() as home:
            await _run(home)
    finally:
        lease._reset_for_tests()
