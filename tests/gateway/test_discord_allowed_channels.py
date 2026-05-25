"""Regression guard for #14920: wildcard "*" in Discord channel config lists.

Setting ``allowed_channels: "*"``, ``free_response_channels: "*"``, or
``ignored_channels: "*"`` in config (or their ``DISCORD_*_CHANNELS`` env var
equivalents) must behave as a wildcard — i.e. the bot responds in every
channel (or is silenced in every channel, for the ignored list). Previously
the literal string "*" was placed into a set and compared against numeric
channel IDs via set-intersection, which always produced an empty set and
caused every message to be silently dropped (for ``allowed_channels``) or
every ``free_response`` / ``ignored`` check to fail open.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys
import unittest

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.MessageType = SimpleNamespace(default=0, reply=1)
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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _channel_is_allowed(channel_id: str, allowed_channels_raw: str) -> bool:
    """Replicate the channel-allow-list check from discord.py on_message."""
    if not allowed_channels_raw:
        return True
    allowed_channels = {ch.strip() for ch in allowed_channels_raw.split(",") if ch.strip()}
    if "*" in allowed_channels:
        return True
    return bool({channel_id} & allowed_channels)


def _channel_is_ignored(channel_id: str, ignored_channels_raw: str) -> bool:
    """Replicate the ignored-channel check from discord.py on_message."""
    ignored_channels = {
        ch.strip() for ch in ignored_channels_raw.split(",") if ch.strip()
    }
    return "*" in ignored_channels or bool({channel_id} & ignored_channels)


def _channel_is_free_response(channel_id: str, free_channels_raw: str) -> bool:
    """Replicate the free-response-channel check from discord.py on_message."""
    free_channels = {
        ch.strip() for ch in free_channels_raw.split(",") if ch.strip()
    }
    return "*" in free_channels or bool({channel_id} & free_channels)


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", type("DMChannel", (), {}), raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", type("Thread", (), {}), raising=False)
    monkeypatch.setattr(discord_platform.discord, "MessageType", SimpleNamespace(default=0, reply=1), raising=False)
    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_AUTO_THREAD",
    ):
        monkeypatch.delenv(var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str):
    author = SimpleNamespace(id=42, display_name="TestUser", name="TestUser", bot=False)
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=[],
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=discord_platform.discord.MessageType.default,
    )


class TestDiscordAllowedChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_ALLOWED_CHANNELS."""

    def test_wildcard_allows_any_channel(self):
        """'*' should allow messages from any channel ID."""
        self.assertTrue(_channel_is_allowed("1234567890", "*"))

    def test_wildcard_in_list_allows_any_channel(self):
        """'*' mixed with other entries still allows any channel."""
        self.assertTrue(_channel_is_allowed("9999999999", "111,*,222"))

    def test_exact_match_allowed(self):
        """Channel ID present in the explicit list is allowed."""
        self.assertTrue(_channel_is_allowed("1234567890", "1234567890,9876543210"))

    def test_non_matching_channel_blocked(self):
        """Channel ID absent from the explicit list is blocked."""
        self.assertFalse(_channel_is_allowed("5555555555", "1234567890,9876543210"))

    def test_empty_allowlist_allows_all(self):
        """Empty DISCORD_ALLOWED_CHANNELS means no restriction."""
        self.assertTrue(_channel_is_allowed("1234567890", ""))

    def test_whitespace_only_entry_ignored(self):
        """Entries that are only whitespace are stripped and ignored."""
        self.assertFalse(_channel_is_allowed("1234567890", "  ,  "))


def test_discord_allowed_channels_can_come_from_config_extra(adapter):
    adapter.config.extra["allowed_channels"] = ["123", "456"]

    assert adapter._discord_allowed_channels() == {"123", "456"}


def test_discord_allowed_channels_env_overrides_config_extra(adapter, monkeypatch):
    adapter.config.extra["allowed_channels"] = ["123"]
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "456")

    assert adapter._discord_allowed_channels() == {"456"}


def test_discord_allowed_channels_config_wildcard(adapter):
    adapter.config.extra["allowed_channels"] = "*"

    assert adapter._discord_allowed_channels() == {"*"}


@pytest.mark.asyncio
async def test_discord_allowed_channels_config_blocks_unlisted_channel(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["allowed_channels"] = ["123"]

    message = make_message(channel=FakeTextChannel(channel_id=999), content="blocked")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_allowed_channels_config_allows_listed_channel(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["allowed_channels"] = ["123"]

    message = make_message(channel=FakeTextChannel(channel_id=123), content="allowed")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


class TestDiscordIgnoredChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_IGNORED_CHANNELS."""

    def test_wildcard_silences_every_channel(self):
        """'*' in ignored_channels silences the bot everywhere."""
        self.assertTrue(_channel_is_ignored("1234567890", "*"))

    def test_empty_ignored_list_silences_nothing(self):
        self.assertFalse(_channel_is_ignored("1234567890", ""))

    def test_exact_match_is_ignored(self):
        self.assertTrue(_channel_is_ignored("111", "111,222"))

    def test_non_match_not_ignored(self):
        self.assertFalse(_channel_is_ignored("333", "111,222"))


class TestDiscordFreeResponseChannelsWildcard(unittest.TestCase):
    """Wildcard and channel-list behaviour for DISCORD_FREE_RESPONSE_CHANNELS."""

    def test_wildcard_makes_every_channel_free_response(self):
        """'*' in free_response_channels exempts every channel from mention-required."""
        self.assertTrue(_channel_is_free_response("1234567890", "*"))

    def test_wildcard_in_list_applies_everywhere(self):
        self.assertTrue(_channel_is_free_response("9999999999", "111,*,222"))

    def test_exact_match_is_free_response(self):
        self.assertTrue(_channel_is_free_response("111", "111,222"))

    def test_non_match_not_free_response(self):
        self.assertFalse(_channel_is_free_response("333", "111,222"))

    def test_empty_list_no_free_response(self):
        self.assertFalse(_channel_is_free_response("111", ""))
