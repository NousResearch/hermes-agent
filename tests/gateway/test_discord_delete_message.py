"""Tests for Discord adapter delete_message — enables cleanup_progress.

The gateway's ``cleanup_progress`` feature gates on
``type(adapter).delete_message is not BasePlatformAdapter.delete_message``.
Without an override, cleanup is silently disabled and progress bubbles
remain in the channel forever.  These tests guard against accidental removal.
"""
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter  # noqa: E402


class TestDeleteMessageOverride:
    """Guard: delete_message must be overridden so cleanup_progress works."""

    def test_delete_message_is_overridden(self):
        """The gateway cleanup_progress gate checks
        ``type(adapter).delete_message is BasePlatformAdapter.delete_message``.
        If DiscordAdapter doesn't override, cleanup is silently disabled."""
        assert DiscordAdapter.delete_message is not BasePlatformAdapter.delete_message


class TestDeleteMessage:
    @pytest.mark.asyncio
    async def test_delete_message_deletes_via_channel_fetch(self):
        """delete_message fetches the channel, fetches the message, and deletes it."""
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

        msg = SimpleNamespace(delete=AsyncMock())
        channel = SimpleNamespace(
            fetch_message=AsyncMock(return_value=msg),
        )
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=channel),
        )

        result = await adapter.delete_message("123456", "789012")

        assert result is True
        msg.delete.assert_awaited_once()
        channel.fetch_message.assert_awaited_once_with(789012)

    @pytest.mark.asyncio
    async def test_delete_message_falls_back_to_fetch_channel(self):
        """When get_channel returns None, falls back to fetch_channel."""
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

        msg = SimpleNamespace(delete=AsyncMock())
        channel = SimpleNamespace(
            fetch_message=AsyncMock(return_value=msg),
        )
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=None),
            fetch_channel=AsyncMock(return_value=channel),
        )

        result = await adapter.delete_message("123456", "789012")

        assert result is True
        adapter._client.fetch_channel.assert_awaited_once_with(123456)
        msg.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_message_returns_false_when_channel_not_found(self):
        """Both get_channel and fetch_channel return None → False."""
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=None),
            fetch_channel=AsyncMock(return_value=None),
        )

        result = await adapter.delete_message("123456", "789012")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_returns_false_when_no_client(self):
        """No Discord client connected → False."""
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        adapter._client = None

        result = await adapter.delete_message("123456", "789012")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_swallows_exception(self):
        """If msg.delete() raises, return False instead of propagating."""
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

        msg = SimpleNamespace(delete=AsyncMock(side_effect=RuntimeError("boom")))
        channel = SimpleNamespace(
            fetch_message=AsyncMock(return_value=msg),
        )
        adapter._client = SimpleNamespace(
            get_channel=MagicMock(return_value=channel),
        )

        result = await adapter.delete_message("123456", "789012")
        assert result is False
