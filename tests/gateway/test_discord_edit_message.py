"""Tests for Discord message edit routing and mention safety."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import adapter as discord_platform


@pytest.mark.asyncio
async def test_edit_message_uses_thread_metadata_and_suppresses_mentions(monkeypatch):
    monkeypatch.setattr(
        discord_platform,
        "_build_allowed_mentions",
        lambda: SimpleNamespace(everyone=False, roles=False, users=True, replied_user=True),
    )
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    edited: dict = {}

    async def _edit(**kwargs):
        edited.update(kwargs)

    message = SimpleNamespace(edit=AsyncMock(side_effect=_edit))
    parent = SimpleNamespace(id=123, fetch_message=AsyncMock())
    thread = SimpleNamespace(id=456, fetch_message=AsyncMock(return_value=message))

    def _get_channel(channel_id):
        if channel_id == 456:
            return thread
        if channel_id == 123:
            return parent
        return None

    adapter._client = SimpleNamespace(
        get_channel=_get_channel,
        fetch_channel=AsyncMock(return_value=None),
    )

    result = await adapter.edit_message(
        "123",
        "789",
        "@everyone progress update",
        metadata={"thread_id": "456"},
    )

    assert result.success is True
    thread.fetch_message.assert_awaited_once_with(789)
    parent.fetch_message.assert_not_called()
    assert edited["content"] == "@everyone progress update"
    assert "allowed_mentions" in edited
    assert edited["allowed_mentions"].everyone is False
    assert edited["allowed_mentions"].roles is False


@pytest.mark.asyncio
async def test_edit_message_marks_missing_message_as_permanent_failure():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)

    channel = SimpleNamespace(fetch_message=AsyncMock(side_effect=RuntimeError("404 Not Found")))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=None),
    )

    result = await adapter.edit_message("123", "789", "content")

    assert result.success is False
    assert result.retryable is False
