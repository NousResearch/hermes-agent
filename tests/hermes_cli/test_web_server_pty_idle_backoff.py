"""Regression: dashboard PTY pump must back off when the terminal is idle."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from hermes_cli.web_server import _legacy_pump


class _FakeBridge:
    def __init__(self, reads):
        self._reads = list(reads)
        self.closed = False

    def read(self, timeout):
        if self._reads:
            return self._reads.pop(0)
        return None

    def write(self, data):
        pass

    def resize(self, cols, rows):
        pass

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_legacy_pump_sleeps_on_idle_pty():
    """An empty PTY read must not spin with ``asyncio.sleep(0)``."""
    sleeps: list[float] = []
    first_idle_seen = asyncio.Event()
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        first_idle_seen.set()
        # Yield with the real sleep so we don't recurse through the patch.
        await real_sleep(0)

    bridge = _FakeBridge([b"", b"", None])

    class _FakeWebSocket:
        def __init__(self):
            self.sent: list[bytes] = []

        async def send_bytes(self, data):
            self.sent.append(data)

        async def receive(self):
            await first_idle_seen.wait()
            return {"type": "websocket.disconnect"}

        async def close(self):
            pass

    ws = _FakeWebSocket()

    with patch("hermes_cli.web_server.asyncio.sleep", side_effect=fake_sleep):
        # _legacy_pump is typed against starlette WebSocket; our fake is an
        # intentional behavioral shim.  # type: ignore[invalid-argument-type]
        await _legacy_pump(ws, bridge)

    assert bridge.closed
    assert sleeps, "pty pump did not call asyncio.sleep on idle ticks"
    assert all(d > 0 for d in sleeps), (
        f"idle pump used zero or negative sleeps: {sleeps}"
    )
