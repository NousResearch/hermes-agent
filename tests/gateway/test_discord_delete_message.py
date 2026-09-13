"""Tests for DiscordAdapter.delete_message.

The gateway's ``display.cleanup_progress`` feature (bf843adf0) deletes
tool-progress / status bubbles after the final response lands, but only on
adapters that override ``BasePlatformAdapter.delete_message``.  These tests
cover the Discord override: successful deletion, the fetch_channel fallback,
and graceful ``False`` returns on failure (cleanup callers fall back to
leaving the message in place).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

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


def _adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


@pytest.mark.asyncio
async def test_delete_message_deletes_and_returns_true():
    adapter = _adapter()

    msg = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    assert await adapter.delete_message("555", "1234") is True
    channel.fetch_message.assert_awaited_once_with(1234)
    msg.delete.assert_awaited_once()
    adapter._client.fetch_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_message_falls_back_to_fetch_channel():
    adapter = _adapter()

    msg = SimpleNamespace(delete=AsyncMock())
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: None,
        fetch_channel=AsyncMock(return_value=channel),
    )

    assert await adapter.delete_message("555", "1234") is True
    adapter._client.fetch_channel.assert_awaited_once_with(555)
    msg.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_message_returns_false_when_not_connected():
    adapter = _adapter()
    adapter._client = None

    assert await adapter.delete_message("555", "1234") is False


@pytest.mark.asyncio
async def test_delete_message_returns_false_on_api_error():
    adapter = _adapter()

    # e.g. 404 Unknown Message (already deleted) or 403 Missing Permissions.
    channel = SimpleNamespace(
        fetch_message=AsyncMock(
            side_effect=RuntimeError("404 Not Found (error code: 10008): Unknown Message")
        ),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    assert await adapter.delete_message("555", "1234") is False


@pytest.mark.asyncio
async def test_delete_message_returns_false_when_delete_raises():
    adapter = _adapter()

    msg = SimpleNamespace(
        delete=AsyncMock(
            side_effect=RuntimeError("403 Forbidden (error code: 50013): Missing Permissions")
        ),
    )
    channel = SimpleNamespace(fetch_message=AsyncMock(return_value=msg))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    assert await adapter.delete_message("555", "1234") is False
