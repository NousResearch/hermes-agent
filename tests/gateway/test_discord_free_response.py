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

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


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

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)

    # Clear DISCORD_* env vars the test file exercises so tests don't leak
    # process-env state from the contributor's shell into per-test behaviour.
    # Individual tests still monkeypatch.setenv() for their own scenarios.
    for _var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_HISTORY_BACKFILL_LIMIT",
        "DISCORD_ALLOW_BOTS",
    ):
        monkeypatch.delenv(_var, raising=False)

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


def make_history_message(
    *,
    author,
    content: str,
    msg_id: int,
    msg_type=None,
    attachments=None,
):
    return SimpleNamespace(
        id=msg_id,
        author=author,
        content=content,
        attachments=list(attachments or []),
        type=msg_type if msg_type is not None else discord_platform.discord.MessageType.default,
    )


class FakeHistoryChannel(FakeTextChannel):
    def __init__(self, history_messages, **kwargs):
        super().__init__(**kwargs)
        self._history_messages = list(history_messages)

    def history(self, *, limit, before, after=None, oldest_first=None):
        before_id = int(getattr(before, "id", before))
        after_id = int(getattr(after, "id", after)) if after is not None else None
        if oldest_first is None:
            oldest_first = after is not None

        messages = [
            message for message in self._history_messages
            if int(message.id) < before_id
            and (after_id is None or int(message.id) > after_id)
        ]
        messages.sort(key=lambda message: int(message.id), reverse=not oldest_first)

        async def _iter():
            for message in messages[:limit]:
                yield message

        return _iter()


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
async def test_discord_free_response_channel_skips_auto_thread(adapter, monkeypatch):
    """Free-response channels should reply inline, never spawn a new thread.

    Without this, every message in a free-response channel would auto-create
    a fresh thread (since the channel bypasses the @mention gate, every
    message looks like a fresh trigger).  That turns a "lightweight chat"
    channel into a thread-spawning machine — see the docs at
    website/docs/user-guide/messaging/discord.md which already describe
    this as the intended behavior.
    """
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "789")
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)  # default true

    adapter._auto_create_thread = AsyncMock()

    message = make_message(
        channel=FakeTextChannel(channel_id=789),
        content="casual chat in free-response channel",
    )

    await adapter._handle_message(message)

    adapter._auto_create_thread.assert_not_awaited()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "casual chat in free-response channel"
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


@pytest.mark.asyncio
async def test_discord_thread_default_keeps_responding_after_participation(adapter, monkeypatch):
    """Default behavior: once the bot is in a thread, it auto-responds without @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("DISCORD_THREAD_REQUIRE_MENTION", raising=False)

    thread = FakeThread(channel_id=456, name="follow-up")
    adapter._threads.mark("456")  # bot has previously participated

    message = make_message(channel=thread, content="follow-up without mention")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_thread_require_mention_gates_followups(adapter, monkeypatch):
    """When thread_require_mention=true, even bot-participated threads need @mention."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    thread = FakeThread(channel_id=456, name="multi-bot thread")
    adapter._threads.mark("456")  # bot has previously participated

    message = make_message(channel=thread, content="ambient chatter — not for me")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_thread_require_mention_still_responds_when_mentioned(adapter, monkeypatch):
    """thread_require_mention=true still lets explicit @mentions through in threads."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_THREAD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)

    thread = FakeThread(channel_id=456, name="multi-bot thread")
    adapter._threads.mark("456")
    bot_user = adapter._client.user

    message = make_message(
        channel=thread,
        content=f"<@{bot_user.id}> hey, this one's for you",
        mentions=[bot_user],
    )
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_discord_thread_require_mention_via_config_extra(adapter, monkeypatch):
    """thread_require_mention can also be set via config.extra (yaml)."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_THREAD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    adapter.config.extra["thread_require_mention"] = True

    thread = FakeThread(channel_id=456, name="multi-bot thread")
    adapter._threads.mark("456")

    message = make_message(channel=thread, content="ambient — should be ignored")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()



@pytest.mark.asyncio
async def test_fetch_channel_context_stops_at_self_message_and_reverses_to_chronological_order(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["history_backfill_limit"] = 10

    other_bot = SimpleNamespace(id=55, display_name="Gemini", name="Gemini", bot=True)
    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)
    old_human = SimpleNamespace(id=57, display_name="Bob", name="Bob", bot=False)

    channel = FakeHistoryChannel(
        [
            make_history_message(author=human, content="latest human note", msg_id=4),
            make_history_message(author=other_bot, content="latest bot note", msg_id=3),
            make_history_message(author=adapter._client.user, content="our prior response", msg_id=2),
            make_history_message(author=old_human, content="older than boundary", msg_id=1),
        ],
        channel_id=123,
    )

    result = await adapter._fetch_channel_context(channel, before=make_message(channel=channel, content="trigger"))

    assert result == (
        "[Recent channel messages]\n"
        "[Gemini [bot]] latest bot note\n"
        "[Alice] latest human note"
    )


@pytest.mark.asyncio
async def test_fetch_channel_context_skips_other_bots_when_allow_bots_none(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["history_backfill_limit"] = 10

    other_bot = SimpleNamespace(id=55, display_name="Gemini", name="Gemini", bot=True)
    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)

    channel = FakeHistoryChannel(
        [
            make_history_message(author=human, content="human note", msg_id=3),
            make_history_message(author=other_bot, content="bot note", msg_id=2),
        ],
        channel_id=123,
    )

    result = await adapter._fetch_channel_context(channel, before=make_message(channel=channel, content="trigger"))

    assert result == "[Recent channel messages]\n[Alice] human note"


@pytest.mark.asyncio
async def test_fetch_channel_context_uses_cache_to_narrow_window(adapter, monkeypatch):
    """When _last_self_message_id is cached, the fetch passes after= to skip old messages."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["history_backfill_limit"] = 50

    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)

    # Record the after= arg passed to history()
    recorded_after = {}

    class CacheTrackingChannel(FakeHistoryChannel):
        def history(self, *, limit, before, after=None, oldest_first=None):
            recorded_after["value"] = after
            return super().history(
                limit=limit,
                before=before,
                after=after,
                oldest_first=oldest_first,
            )

    channel = CacheTrackingChannel(
        [make_history_message(author=human, content="hello", msg_id=200)],
        channel_id=777,
    )

    # Seed the cache — bot's last message in this channel was ID 100
    adapter._last_self_message_id["777"] = "100"

    trigger = make_message(channel=channel, content="trigger")
    trigger.id = 300  # trigger is newer than cache

    result = await adapter._fetch_channel_context(channel, before=trigger)

    assert result == "[Recent channel messages]\n[Alice] hello"
    # Verify cache was used: after= should be set (not None)
    assert recorded_after["value"] is not None


@pytest.mark.asyncio
async def test_fetch_channel_context_cache_uses_latest_window_when_after_set(adapter, monkeypatch):
    """Regression: discord.py defaults oldest_first=True when after= is provided.

    The hot cache path passes both after= and before=. We still want the latest
    messages before the trigger, not the earliest messages after our prior
    response, otherwise tool traces can crowd out the final answer.
    """
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["history_backfill_limit"] = 3

    codex = SimpleNamespace(id=56, display_name="Codex", name="Codex", bot=True)
    human = SimpleNamespace(id=57, display_name="Alice", name="Alice", bot=False)

    channel = FakeHistoryChannel(
        [
            make_history_message(author=codex, content="old tool trace 1", msg_id=101),
            make_history_message(author=codex, content="old tool trace 2", msg_id=102),
            make_history_message(author=codex, content="old tool trace 3", msg_id=103),
            make_history_message(author=codex, content="final analysis", msg_id=104),
            make_history_message(author=human, content="latest follow-up", msg_id=105),
        ],
        channel_id=777,
    )
    adapter._last_self_message_id["777"] = "100"

    trigger = make_message(channel=channel, content="trigger")
    trigger.id = 200

    result = await adapter._fetch_channel_context(channel, before=trigger)

    assert "[Codex [bot]] final analysis" in result
    assert "[Alice] latest follow-up" in result
    assert "old tool trace 1" not in result
    assert "old tool trace 2" not in result


@pytest.mark.asyncio
async def test_fetch_channel_context_ignores_stale_cache(adapter, monkeypatch):
    """If cached ID is >= trigger ID (stale/future), fall back to cold-start scan."""
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["history_backfill_limit"] = 50

    human = SimpleNamespace(id=56, display_name="Alice", name="Alice", bot=False)

    recorded_after = {}

    class CacheTrackingChannel(FakeHistoryChannel):
        def history(self, *, limit, before, after=None, oldest_first=None):
            recorded_after["value"] = after
            return super().history(
                limit=limit,
                before=before,
                after=after,
                oldest_first=oldest_first,
            )

    channel = CacheTrackingChannel(
        [make_history_message(author=human, content="hello", msg_id=50)],
        channel_id=777,
    )

    # Cache has a NEWER ID than the trigger — stale/invalid
    adapter._last_self_message_id["777"] = "500"

    trigger = make_message(channel=channel, content="trigger")
    trigger.id = 300

    result = await adapter._fetch_channel_context(channel, before=trigger)

    assert result == "[Recent channel messages]\n[Alice] hello"
    # Cache should have been ignored — after= should be None
    assert recorded_after["value"] is None


@pytest.mark.asyncio
async def test_discord_shared_channel_backfill_prepends_context(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["group_sessions_per_user"] = False
    adapter.config.extra["history_backfill"] = True
    adapter._fetch_channel_context = AsyncMock(return_value="[Recent channel messages]\n[Alice] context")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"<@{bot_user.id}> hello with mention",
        mentions=[bot_user],
    )

    await adapter._handle_message(message)

    adapter._fetch_channel_context.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello with mention"
    assert event.channel_context == "[Recent channel messages]\n[Alice] context"


@pytest.mark.asyncio
async def test_discord_per_user_channel_backfills_too(adapter, monkeypatch):
    """Per-user sessions also benefit from backfill: Alice's session is missing
    other-channel-participants' context and her own pre-mention messages."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["group_sessions_per_user"] = True
    adapter.config.extra["history_backfill"] = True
    adapter._fetch_channel_context = AsyncMock(return_value="[Recent channel messages]\n[Alice] context")

    bot_user = adapter._client.user
    message = make_message(
        channel=FakeTextChannel(channel_id=321),
        content=f"<@{bot_user.id}> hello with mention",
        mentions=[bot_user],
    )

    await adapter._handle_message(message)

    adapter._fetch_channel_context.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello with mention"
    assert event.channel_context == "[Recent channel messages]\n[Alice] context"


@pytest.mark.asyncio
async def test_discord_dm_does_not_backfill(adapter, monkeypatch):
    """DMs skip backfill — every DM triggers the bot, so there's no mention gap."""
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    adapter.config.extra["history_backfill"] = True
    adapter._fetch_channel_context = AsyncMock(return_value="[Recent channel messages]\n[Alice] context")

    bot_user = adapter._client.user
    dm_channel = SimpleNamespace(
        id=999,
        name=None,
        guild=None,
        topic=None,
    )
    # Make isinstance(channel, discord.DMChannel) return True
    monkeypatch.setattr(
        discord_platform.discord, "DMChannel", type(dm_channel), raising=False,
    )

    message = make_message(
        channel=dm_channel,
        content="hello in DM",
        mentions=[],
    )

    await adapter._handle_message(message)

    adapter._fetch_channel_context.assert_not_awaited()
    if adapter.handle_message.await_args is not None:
        event = adapter.handle_message.await_args.args[0]
        assert event.channel_context is None


