"""Buzz WebSocket dial races IPv6/IPv4 (#114265): ``happy_eyeballs_delay`` reaches
``loop.create_connection`` through ``websockets.connect`` — see tests/gateway/test_ws_dial_happy_eyeballs.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("websockets")

EXPECTED_DELAY = 0.25


@pytest.mark.asyncio
async def test_buzz_websocket_loop_dial_races_ipv6_ipv4(monkeypatch):
    import websockets

    from plugins.platforms.buzz.adapter import BuzzAdapter

    adapter = BuzzAdapter.__new__(BuzzAdapter)
    adapter.relay_url = "https://relay.example.com/"
    adapter._ws_ready = None
    adapter._mark_connected = MagicMock()
    recorded: dict = {}

    def fake_connect(*_args, **kwargs):
        recorded.update(kwargs)
        raise asyncio.CancelledError  # first dial ends the loop: only the kwargs matter here

    monkeypatch.setattr(websockets, "connect", fake_connect)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await adapter._websocket_loop()
    assert recorded["happy_eyeballs_delay"] == EXPECTED_DELAY
