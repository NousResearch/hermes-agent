"""Discord typing indicator configuration tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.discord import DiscordAdapter
from gateway.platforms.helpers import ThreadParticipationTracker

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="e2e-test-token")
    with patch.object(ThreadParticipationTracker, "_load", return_value=set()):
        adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(http=SimpleNamespace(request=AsyncMock()))
    return adapter


async def test_discord_typing_disabled_by_env(adapter, monkeypatch):
    """DISCORD_TYPING=false suppresses Discord typing REST calls."""
    monkeypatch.setenv("DISCORD_TYPING", "false")

    await adapter.send_typing("12345")
    await asyncio.sleep(0)

    adapter._client.http.request.assert_not_awaited()
    assert adapter._typing_tasks == {}


async def test_discord_typing_enabled_by_default(adapter, monkeypatch):
    """Typing remains enabled unless DISCORD_TYPING is explicitly false/0/no."""
    monkeypatch.delenv("DISCORD_TYPING", raising=False)

    await adapter.send_typing("12345")
    await asyncio.sleep(0)

    adapter._client.http.request.assert_awaited_once()
    await adapter.stop_typing("12345")
