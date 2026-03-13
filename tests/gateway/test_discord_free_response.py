"""Tests for Discord free-response defaults and mention gating."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

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
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int = 1, name: str = "general", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.topic = None


class FakeForumChannel:
    def __init__(self, channel_id: int = 1, name: str = "support-forum", guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.type = 15
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int = 1, name: str = "thread", parent=None, guild_name: str = "Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None):
    author = SimpleNamespace(id=42, display_name="Jezza", name="Jezza")
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
    )


@pytest.mark.asyncio
async def test_discord_defaults_to_require_mention(adapter, monkeypatch):
    """Default behavior: require @mention in server channels."""
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=123), content="hello from channel")

    await adapter._handle_message(message)

    # Should be ignored — no mention, require_mention defaults to true
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_free_response_in_server_channels(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=123), content="hello from channel")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello from channel"
    assert event.source.chat_id == "123"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_discord_free_response_in_threads(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    thread = FakeThread(channel_id=456, name="Ghost reader skill")
    message = make_message(channel=thread, content="hello from thread")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello from thread"
    assert event.source.chat_id == "456"
    assert event.source.thread_id == "456"
    assert event.source.chat_type == "thread"


@pytest.mark.asyncio
async def test_discord_forum_threads_are_handled_as_threads(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    forum = FakeForumChannel(channel_id=222, name="support-forum")
    thread = FakeThread(channel_id=456, name="Can Hermes reply here?", parent=forum)
    message = make_message(channel=thread, content="hello from forum post")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello from forum post"
    assert event.source.chat_id == "456"
    assert event.source.thread_id == "456"
    assert event.source.chat_type == "thread"
    assert event.source.chat_name == "Hermes Server / support-forum / Can Hermes reply here?"


@pytest.mark.asyncio
async def test_discord_can_still_require_mentions_when_enabled(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeTextChannel(channel_id=789), content="ignored without mention")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_free_response_channel_overrides_mention_requirement(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "789,999")

    message = make_message(channel=FakeTextChannel(channel_id=789), content="allowed without mention")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "allowed without mention"


@pytest.mark.asyncio
async def test_discord_forum_parent_in_free_response_list_allows_forum_thread(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "222")

    forum = FakeForumChannel(channel_id=222, name="support-forum")
    thread = FakeThread(channel_id=333, name="Forum topic", parent=forum)
    message = make_message(channel=thread, content="allowed from forum thread")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "allowed from forum thread"
    assert event.source.chat_id == "333"


@pytest.mark.asyncio
async def test_discord_accepts_and_strips_bot_mentions_when_required(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"<@{bot_user.id}> hello with mention",
        mentions=[bot_user],
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello with mention"


@pytest.mark.asyncio
async def test_discord_dms_ignore_mention_requirement(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    message = make_message(channel=FakeDMChannel(channel_id=654), content="dm without mention")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "dm without mention"
    assert event.source.chat_type == "dm"
