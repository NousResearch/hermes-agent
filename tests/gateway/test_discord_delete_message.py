"""Tests for DiscordAdapter.delete_message cleanup support."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.discord.adapter import DiscordAdapter


class _FakeClient:
    def __init__(self, channel=None, fetched_channel=None):
        self.channel = channel
        self.fetched_channel = fetched_channel
        self.fetch_channel = AsyncMock(return_value=fetched_channel)

    def get_channel(self, channel_id):
        self.last_get_channel_id = channel_id
        return self.channel


def _adapter(client):
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._client = client
    return adapter


@pytest.mark.asyncio
async def test_delete_message_overrides_base_adapter():
    assert DiscordAdapter.delete_message is not BasePlatformAdapter.delete_message


@pytest.mark.asyncio
async def test_delete_message_deletes_message_from_cached_channel():
    message = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter = _adapter(_FakeClient(channel=channel))

    assert await adapter.delete_message("123", "456") is True

    channel.fetch_message.assert_awaited_once_with(456)
    message.delete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_message_fetches_channel_when_not_cached():
    message = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    client = _FakeClient(channel=None, fetched_channel=channel)
    adapter = _adapter(client)

    assert await adapter.delete_message("123", "456") is True

    client.fetch_channel.assert_awaited_once_with(123)
    channel.fetch_message.assert_awaited_once_with(456)
    message.delete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_message_returns_false_when_not_connected():
    adapter = _adapter(None)

    assert await adapter.delete_message("123", "456") is False


@pytest.mark.asyncio
async def test_delete_message_treats_unknown_message_as_cleaned_up():
    channel = SimpleNamespace(
        fetch_message=AsyncMock(
            side_effect=Exception("404 Not Found (error code: 10008): Unknown Message")
        )
    )
    adapter = _adapter(_FakeClient(channel=channel))

    assert await adapter.delete_message("123", "456") is True


@pytest.mark.asyncio
async def test_delete_message_returns_false_on_delete_failure():
    message = SimpleNamespace(delete=AsyncMock(side_effect=Exception("Missing Permissions")))
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=message))
    adapter = _adapter(_FakeClient(channel=channel))

    assert await adapter.delete_message("123", "456") is False
