"""End-to-end input gating through ``display._bridge``: the WebSocket-to-Xvnc path runs every client
byte through :class:`RfbClientFilter`, so a watcher (no input lease) can watch but cannot type. These
tests stand up a real Unix socket as Xvnc, drive the bridge with a fake WebSocket, and assert on the
bytes that actually reach Xvnc."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from hermes_cli.web_routers import display
from tests.tools.test_bot_desktop_rfb_filter_limits import fence
from tools.bot_desktop import lease

pytestmark = pytest.mark.platforms("posix")  # the bridge and fake Xvnc need Unix sockets

_HANDSHAKE = b"RFB 003.008\n\x01\x01"
_KEY_EVENT = b"\x04\x00\x61\x00\x00\x00\x00\x00"  # KeyEvent, key 'a'


class _SendingWs:
    """Just enough of a Starlette WebSocket: replays binary frames, then disconnects."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.closed = False

    async def accept(self):
        pass

    async def receive(self):
        if self._chunks:
            return {"type": "websocket.receive", "bytes": self._chunks.pop(0)}
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_bytes(self, data):
        pass

    async def close(self, code=1000, reason=""):
        self.closed = True


async def _bridge_capture(home: str, ws: _SendingWs, viewer_id: str = "desk-1") -> bytes:
    """Run the real bridge against a fake Xvnc that records every byte it receives."""
    sock_dir = os.path.join(home, "bot-desktop")
    os.makedirs(sock_dir, exist_ok=True)
    received = bytearray()
    eof = asyncio.Event()

    async def _xvnc(reader, writer):
        while True:
            data = await reader.read(65536)
            if not data:
                eof.set()
                return
            received.extend(data)

    server = await asyncio.start_unix_server(_xvnc, path=os.path.join(sock_dir, "rfb.sock"))
    try:
        await display._bridge(ws, {"hermes_home": home, "viewer_id": viewer_id})
        await asyncio.wait_for(eof.wait(), timeout=2)
        return bytes(received)
    finally:
        server.close()


@pytest.mark.parametrize("holds_lease", [False, True])
def test_ws_input_gate_reaches_xvnc(holds_lease):
    """A Request-flagged fence followed by a KeyEvent, in a single WebSocket frame: the watcher gets
    only the fence through to Xvnc; the holder gets both. Runs the production ``_bridge`` path, not
    the filter in isolation."""
    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        if holds_lease:
            lease.acquire("desk-1", profile_key=home)
        try:
            client_bytes = _HANDSHAKE + fence(flags=0x80000007) + _KEY_EVENT
            got = asyncio.run(_bridge_capture(home, _SendingWs([client_bytes])))
        finally:
            lease._reset_for_tests()
    expected = _HANDSHAKE + fence(flags=0x80000007) + (_KEY_EVENT if holds_lease else b"")
    assert got == expected


@pytest.mark.parametrize("holds_lease", [False, True])
def test_ws_unframeable_message_closes_the_socket(holds_lease):
    """An unframeable client message (QEMU sub-type 1, audio) makes the filter raise;
    the bridge must close that viewer's WebSocket instead of forwarding blind, and
    nothing past it may reach Xvnc. Holder and watcher alike: a misframe is not an
    input path."""
    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        if holds_lease:
            lease.acquire("desk-1", profile_key=home)
        try:
            ws = _SendingWs([_HANDSHAKE, bytes([255, 1, 0, 0]) + _KEY_EVENT])
            got = asyncio.run(_bridge_capture(home, ws))
        finally:
            lease._reset_for_tests()
    assert ws.closed is True
    assert got == _HANDSHAKE
