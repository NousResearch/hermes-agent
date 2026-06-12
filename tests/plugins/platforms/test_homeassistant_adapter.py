"""Tests for the Home Assistant platform adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.homeassistant.adapter import HomeAssistantAdapter


@pytest.mark.asyncio
async def test_connect_websocket_cleans_up_session_when_ws_connect_fails(monkeypatch):
    import plugins.platforms.homeassistant.adapter as ha_module

    class FailingSession:
        def ws_connect(self, *args, **kwargs):
            raise RuntimeError("dial failed")

    monkeypatch.setattr(
        ha_module,
        "aiohttp",
        SimpleNamespace(
            ClientSession=lambda **kwargs: FailingSession(),
            ClientTimeout=lambda **kwargs: object(),
        ),
    )
    adapter = HomeAssistantAdapter(
        PlatformConfig(
            enabled=True,
            token="token-1",
            extra={"url": "http://homeassistant.local:8123"},
        )
    )
    adapter._cleanup_ws = AsyncMock()

    with pytest.raises(RuntimeError, match="dial failed"):
        await adapter._ws_connect()

    adapter._cleanup_ws.assert_awaited_once()
