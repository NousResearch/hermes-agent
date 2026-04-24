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
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    return adapter


def make_message(*, channel, content: str, mentions=None, msg_type=None):
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
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
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
async def test_discord_free_response_channel_can_come_from_config_extra(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    adapter.config.extra["free_response_channels"] = ["789", "999"]

    message = make_message(channel=FakeTextChannel(channel_id=789), content="allowed from config")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "allowed from config"


def test_discord_free_response_channels_bare_int(adapter, monkeypatch):
    # YAML `discord.free_response_channels: 1491973769726791812` (single bare
    # integer) is loaded as an int and previously fell through the
    # isinstance(str) branch in _discord_free_response_channels, silently
    # returning an empty set.  Scalar → str coercion makes single-channel
    # config work without having to quote the ID in YAML.
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    adapter.config.extra["free_response_channels"] = 1491973769726791812

    assert adapter._discord_free_response_channels() == {"1491973769726791812"}


def test_discord_free_response_channels_int_list(adapter, monkeypatch):
    # YAML list form with bare numeric entries — each element should be coerced.
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    adapter.config.extra["free_response_channels"] = [1491973769726791812, 99999]

    assert adapter._discord_free_response_channels() == {"1491973769726791812", "99999"}


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


@pytest.mark.asyncio
async def test_discord_auto_thread_enabled_by_default(adapter, monkeypatch):
    """Auto-threading should be enabled by default (DISCORD_AUTO_THREAD defaults to 'true')."""
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    # Patch _auto_create_thread to return a fake thread
    fake_thread = FakeThread(channel_id=999, name="auto-thread")
    adapter._auto_create_thread = AsyncMock(return_value=fake_thread)

    message = make_message(channel=FakeTextChannel(channel_id=123), content="hello")

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "999"


@pytest.mark.asyncio
async def test_discord_reply_message_skips_auto_thread(adapter, monkeypatch):
    """Quote-replies should stay in-channel instead of trying to create a thread."""
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "123")

    adapter._auto_create_thread = AsyncMock()

    message = make_message(
        channel=FakeTextChannel(channel_id=123),
        content="reply without mention",
        msg_type=discord_platform.discord.MessageType.reply,
    )

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "reply without mention"
    assert event.source.chat_id == "123"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_discord_auto_thread_can_be_disabled(adapter, monkeypatch):
    """Setting auto_thread to false skips thread creation."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    adapter._auto_create_thread = AsyncMock()

    message = make_message(channel=FakeTextChannel(channel_id=123), content="hello")

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_discord_bot_thread_skips_mention_requirement(adapter, monkeypatch):
    """Messages in a thread the bot has participated in should not require @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    # Simulate bot having previously participated in thread 456
    adapter._threads.mark("456")

    thread = FakeThread(channel_id=456, name="existing thread")
    message = make_message(channel=thread, content="follow-up without mention")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "follow-up without mention"
    assert event.source.chat_type == "thread"


@pytest.mark.asyncio
async def test_discord_unknown_thread_still_requires_mention(adapter, monkeypatch):
    """Messages in a thread the bot hasn't participated in should still require @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    # Bot has NOT participated in thread 789
    thread = FakeThread(channel_id=789, name="some thread")
    message = make_message(channel=thread, content="hello from unknown thread")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_auto_thread_tracks_participation(adapter, monkeypatch):
    """Auto-created threads should be tracked for future mention-free replies."""
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    fake_thread = FakeThread(channel_id=555, name="auto-thread")
    adapter._auto_create_thread = AsyncMock(return_value=fake_thread)

    message = make_message(channel=FakeTextChannel(channel_id=123), content="start a thread")

    await adapter._handle_message(message)

    assert "555" in adapter._threads


@pytest.mark.asyncio
async def test_discord_thread_participation_tracked_on_dispatch(adapter, monkeypatch):
    """When the bot processes a message in a thread, it tracks participation."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    thread = FakeThread(channel_id=777, name="manually created thread")
    message = make_message(channel=thread, content="hello in thread")

    await adapter._handle_message(message)

    assert "777" in adapter._threads


@pytest.mark.asyncio
async def test_discord_voice_linked_channel_skips_mention_requirement_and_auto_thread(adapter, monkeypatch):
    """Active voice-linked text channels should behave like free-response channels."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)

    adapter._voice_text_channels[111] = 789
    adapter._auto_create_thread = AsyncMock()

    message = make_message(
        channel=FakeTextChannel(channel_id=789),
        content="follow-up from voice text chat",
    )

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "follow-up from voice text chat"
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_discord_free_channel_skips_auto_thread(adapter, monkeypatch):
    """Free-response channels must NOT auto-create threads — bot replies inline.

    Without this, every message in a free-response channel would spin off a
    thread (since the channel bypasses the @mention gate), defeating the
    lightweight-chat purpose of free-response mode.
    """
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "789")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)  # default true

    adapter._auto_create_thread = AsyncMock()

    message = make_message(
        channel=FakeTextChannel(channel_id=789),
        content="free chat message",
    )

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_discord_voice_linked_parent_thread_still_requires_mention(adapter, monkeypatch):
    """Threads under a voice-linked channel should still require @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    adapter._voice_text_channels[111] = 789
    message = make_message(
        channel=FakeThread(channel_id=790, parent=FakeTextChannel(channel_id=789)),
        content="thread reply without mention",
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()
