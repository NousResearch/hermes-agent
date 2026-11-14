"""The CLI/TUI gateway client never keepalive-pings a loopback owner (#154910).

The owner answers pings on the same event loop that runs the turn, so a long turn that keeps the
loop busy for 20 s left the client's ping unanswered; the client then closed every socket with
``1011 keepalive ping timeout`` and reported the turn outcome as unknown while it kept running.
A dead loopback peer sends a real FIN/RST, so the ping buys nothing there (the dashboard server
drops it on loopback for the same reason). A remote owner keeps the websockets default ping.
"""

import pytest

from hermes_cli import gateway_client


async def _dial_kwargs(monkeypatch, url: str) -> dict:
    captured: dict = {}

    def _connect(*args, **kwargs):
        captured.update(kwargs)
        raise OSError("not dialing in this test")

    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", url)
    monkeypatch.setattr("websockets.asyncio.client.connect", _connect)
    with pytest.raises(gateway_client.GatewayClientError):
        async with gateway_client.connect_gateway():
            pass
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "ws://127.0.0.1:9119/api/ws", "ws://127.8.0.1:9119/api/ws", "ws://localhost:9119/api/ws",
    "ws://LOCALHOST.:9119/api/ws", "ws://[::1]:9119/api/ws",
])
async def test_loopback_owner_is_not_keepalive_pinged(monkeypatch, url):
    kwargs = await _dial_kwargs(monkeypatch, url)
    assert kwargs["ping_interval"] is None
    assert kwargs["ping_timeout"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["wss://gateway.example.com/api/ws", "ws://192.168.1.20:9119/api/ws"])
async def test_remote_owner_keeps_the_default_keepalive(monkeypatch, url):
    kwargs = await _dial_kwargs(monkeypatch, url)
    assert "ping_interval" not in kwargs
    assert "ping_timeout" not in kwargs
