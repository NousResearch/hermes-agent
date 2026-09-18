"""Cold-start WebSocket dials race IPv6/IPv4 (#114265).

``websockets.connect`` forwards unknown kwargs to ``loop.create_connection``, whose
``happy_eyeballs_delay`` defaults to ``None`` (serial walk over getaddrinfo results). On a
network with an advertised-but-blackholed IPv6 route every AAAA record burns the full
connect timeout before IPv4 answers — the same stall class the bootstrap racer closes for
sync connects. Each Hermes ``websockets.connect`` call site must pass the RFC 8305 delay.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("websockets")

EXPECTED_DELAY = 0.25


class _Recorder:
    def __init__(self):
        self.kwargs = None

    async def __call__(self, *_args, **kwargs):
        self.kwargs = kwargs
        raise ConnectionRefusedError("recorded")


@pytest.mark.asyncio
async def test_relay_ws_transport_dial_races_ipv6_ipv4(monkeypatch):
    import gateway.relay.ws_transport as mod

    rec = _Recorder()
    monkeypatch.setattr(mod.websockets, "connect", rec)
    t = mod.WebSocketRelayTransport("ws://unused", "discord", "bot", reconnect=False)
    with pytest.raises(ConnectionRefusedError):
        await t._dial_and_start()
    assert rec.kwargs["happy_eyeballs_delay"] == EXPECTED_DELAY


@pytest.mark.asyncio
async def test_yuanbao_dial_races_ipv6_ipv4():
    from gateway.platforms.yuanbao import ConnectionManager, YuanbaoAdapter

    adapter = MagicMock(spec=YuanbaoAdapter)
    adapter.name = "yuanbao"
    adapter._bot_id = "b"
    adapter._ws_url = "wss://test.example.com/ws"
    cm = ConnectionManager(adapter)
    rec = _Recorder()
    with patch("gateway.platforms.yuanbao.websockets.connect", rec), pytest.raises(ConnectionRefusedError):
        await cm._dial({"bot_id": "b", "token": "t"})
    assert rec.kwargs["happy_eyeballs_delay"] == EXPECTED_DELAY

