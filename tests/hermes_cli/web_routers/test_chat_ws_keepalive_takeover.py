"""A takeover that cancels a keep-alive /api/pty write is not an input stall.

The new viewer's attach cancels the old socket's in-flight write, which then returns False. The
keep-alive writer read that as a stalled child: it logged "recycling terminal session" and sent
the superseded socket a second close, 1013, which only the attach's 4409 kept from landing
(PR #93508 review).
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from hermes_cli.pty_session import PtySession, WS_CLOSE_SUPERSEDED
from hermes_cli.web_routers.chat_ws import _pump_keepalive_input


class BackpressuredBridge:
    """The child is not reading its input: every write waits until it is cancelled."""

    def __init__(self):
        self.waiting = asyncio.Event()

    async def write(self, data, **_options):
        self.waiting.set()
        await asyncio.Event().wait()


class Viewer:
    """Sends ``frames``, then reads until closed; a second close raises, like Starlette's."""

    def __init__(self, *frames):
        self.closed = []
        self.inbound = asyncio.Queue()
        for frame in frames:
            self.inbound.put_nowait({"type": "websocket.receive", "bytes": frame})

    async def receive(self):
        return await self.inbound.get()

    async def close(self, code=1000, reason=None):
        if self.closed:
            raise RuntimeError("Cannot call send once a close message has been sent.")
        self.closed.append(code)
        self.inbound.put_nowait({"type": "websocket.disconnect"})


@pytest.mark.platforms("any")  # the bridge is a stand-in, so the takeover runs the same on every host
@pytest.mark.asyncio
async def test_takeover_during_a_backpressured_write_is_not_reported_as_a_stall(caplog):
    bridge = BackpressuredBridge()
    session = PtySession("chat", bridge, buffer_cap=32, read_timeout=0.01)
    old, new = Viewer(b"a long paste"), Viewer()
    assert await session.attach(old)
    pump = asyncio.create_task(_pump_keepalive_input(old, session))
    await bridge.waiting.wait()

    with caplog.at_level(logging.WARNING):
        assert await session.attach(new)
        async with asyncio.timeout(3):
            await pump

    assert old.closed == [WS_CLOSE_SUPERSEDED]
    assert "stalled" not in caplog.text
    assert session.alive and session._ws is new
