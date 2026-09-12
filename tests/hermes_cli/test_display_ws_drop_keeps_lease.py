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


@pytest.mark.parametrize(("close_code", "human_keeps_control"), [(1006, True), (1000, False)])
def test_only_a_clean_viewer_close_hands_the_screen_back(monkeypatch, close_code, human_keeps_control):
    lease._reset_for_tests()
    with tempfile.TemporaryDirectory() as home:
        lease.acquire("desk-1", profile_key=home)
        after = asyncio.run(_bridge_once(close_code, home))
    lease._reset_for_tests()
    assert (after.holder == lease.HUMAN) is human_keeps_control, after
