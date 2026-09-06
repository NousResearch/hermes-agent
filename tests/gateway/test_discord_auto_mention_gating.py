"""Tests for Discord auto mention gating driven by how many bots share a room.

Rule under test: a room with two or more bots requires an explicit @mention
before this bot answers, and replies in such a room open with an @mention of
the addressee.  A room where this bot is the only bot needs no mention in
either direction.
"""

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

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


SELF_BOT_ID = 999


class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"
        self.guild = None


def _member(member_id: int, *, bot: bool, name: str = "member"):
    return SimpleNamespace(id=member_id, bot=bot, display_name=name, name=name)


class FakeGuild:
    def __init__(self, members):
        self.name = "AI Agents"
        self.members = list(members)


class FakeTextChannel:
    """Guild text channel where every member can view the channel."""

    def __init__(self, channel_id: int = 10, *, members, hidden_bot_ids=()):
        self.id = channel_id
        self.name = "general"
        self.guild = FakeGuild(members)
        self.topic = None
        self._hidden_bot_ids = set(hidden_bot_ids)

    def permissions_for(self, member):
        return SimpleNamespace(view_channel=member.id not in self._hidden_bot_ids)

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


class FakeThread:
    """Thread that inherits its parent's permission model (no permissions_for)."""

    def __init__(self, channel_id: int = 20, *, parent):
        self.id = channel_id
        self.name = "thread"
        self.parent = parent
        self.parent_id = parent.id
        self.guild = parent.guild
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


def _make_adapter(monkeypatch, *, extra=None):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_BOT_EXCHANGE_LIMIT",
        "DISCORD_BOT_EXCHANGE_WINDOW",
    ):
        monkeypatch.delenv(var, raising=False)

    # auto_thread is orthogonal to mention gating and needs a real channel.create_thread()
    # (a FakeTextChannel has none). Default it off so these tests exercise only the gate;
    # a test that cares about threading can override it via extra.
    config = PlatformConfig(
        enabled=True, token="fake-token", extra={"auto_thread": False, **(extra or {})},
    )
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=_member(SELF_BOT_ID, bot=True, name="HermesMac"))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def _make_message(*, channel, content, mentions=None, author=None):
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author or _member(42, bot=False, name="van"),
        type=discord_platform.discord.MessageType.default,
    )


def _solo_channel():
    return FakeTextChannel(members=[
        _member(SELF_BOT_ID, bot=True, name="HermesMac"),
        _member(42, bot=False, name="van"),
    ])


def _shared_channel():
    return FakeTextChannel(members=[
        _member(SELF_BOT_ID, bot=True, name="HermesMac"),
        _member(555, bot=True, name="HitoBot"),
        _member(42, bot=False, name="van"),
    ])


# --- bot counting ------------------------------------------------------------

def test_bot_count_counts_only_bots_that_can_see_the_channel(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    channel = FakeTextChannel(
        members=[
            _member(SELF_BOT_ID, bot=True, name="HermesMac"),
            _member(555, bot=True, name="HitoBot"),
            _member(42, bot=False, name="van"),
        ],
        hidden_bot_ids={555},
    )
    assert adapter._discord_bot_count_for_channel(channel) == 1


def test_bot_count_for_thread_falls_back_to_parent_permissions(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    thread = FakeThread(parent=_shared_channel())
    assert adapter._discord_bot_count_for_channel(thread) == 2


def test_bot_count_is_unknown_without_member_cache(monkeypatch):
    """No Server Members intent → empty member list → cannot tell."""
    adapter = _make_adapter(monkeypatch)
    channel = FakeTextChannel(members=[])
    assert adapter._discord_bot_count_for_channel(channel) is None


# --- channel gating ---------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_mode_answers_without_mention_when_sole_bot(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    message = _make_message(channel=_solo_channel(), content="no mention needed here")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_mode_stays_silent_without_mention_when_two_bots(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    message = _make_message(channel=_shared_channel(), content="ambient chatter")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_mode_answers_when_mentioned_in_shared_room(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    message = _make_message(
        channel=_shared_channel(),
        content=f"<@{SELF_BOT_ID}> this one is for you",
        mentions=[adapter._client.user],
    )

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_mode_requires_mention_when_member_cache_missing(monkeypatch):
    """Unknown bot count must not turn into a free-for-all."""
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    message = _make_message(channel=FakeTextChannel(members=[]), content="who is there?")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_true_still_requires_mention_even_when_sole_bot(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": True})
    message = _make_message(channel=_solo_channel(), content="no mention")

    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_false_answers_even_in_shared_room(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": False})
    message = _make_message(channel=_shared_channel(), content="anything goes")

    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


def test_auto_mode_can_be_set_via_env(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "auto")
    assert adapter._discord_mention_mode() == "auto"
    assert adapter._discord_room_requires_mention(_solo_channel()) is False
    assert adapter._discord_room_requires_mention(_shared_channel()) is True


# --- thread gating ----------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_thread_gating_keeps_answering_in_solo_thread(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        extra={"require_mention": "auto", "thread_require_mention": "auto"},
    )
    thread = FakeThread(channel_id=77, parent=_solo_channel())
    adapter._threads.mark("77")

    message = _make_message(channel=thread, content="follow-up, no mention")
    await adapter._handle_message(message)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_thread_gating_requires_mention_in_shared_thread(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        extra={"require_mention": "auto", "thread_require_mention": "auto"},
    )
    thread = FakeThread(channel_id=78, parent=_shared_channel())
    adapter._threads.mark("78")

    message = _make_message(channel=thread, content="follow-up, no mention")
    await adapter._handle_message(message)

    adapter.handle_message.assert_not_awaited()


# --- outbound addressing ----------------------------------------------------

def test_prefix_mention_only_in_shared_rooms(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    assert adapter._discord_should_prefix_mention(_shared_channel()) is True
    assert adapter._discord_should_prefix_mention(_solo_channel()) is False
    assert adapter._discord_should_prefix_mention(FakeDMChannel()) is False


@pytest.mark.asyncio
async def test_send_prefixes_mention_in_shared_room(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    channel = _shared_channel()
    sent = []

    async def fake_send(*, content, reference=None):
        sent.append(content)
        return SimpleNamespace(id=1)

    channel.send = AsyncMock(side_effect=fake_send)
    adapter._client = SimpleNamespace(
        user=_member(SELF_BOT_ID, bot=True, name="HermesMac"),
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("10", "đã xong", metadata={"discord_mention_user_id": "555"})

    assert result.success is True
    assert sent[0].startswith("<@555>")


@pytest.mark.asyncio
async def test_send_omits_mention_when_sole_bot(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})
    channel = _solo_channel()
    sent = []

    async def fake_send(*, content, reference=None):
        sent.append(content)
        return SimpleNamespace(id=2)

    channel.send = AsyncMock(side_effect=fake_send)
    adapter._client = SimpleNamespace(
        user=_member(SELF_BOT_ID, bot=True, name="HermesMac"),
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("10", "đã xong", metadata={"discord_mention_user_id": "42"})

    assert result.success is True
    assert sent[0] == "đã xong"


# --- bot-to-bot loop guard --------------------------------------------------

def test_bot_exchange_guard_trips_after_limit(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_LIMIT", "3")
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_WINDOW", "120")

    assert [adapter._discord_bot_exchange_allowed("10") for _ in range(4)] == [
        True, True, True, False,
    ]


def test_bot_exchange_guard_is_per_channel(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_LIMIT", "1")

    assert adapter._discord_bot_exchange_allowed("10") is True
    assert adapter._discord_bot_exchange_allowed("10") is False
    assert adapter._discord_bot_exchange_allowed("11") is True


def test_bot_exchange_guard_recovers_after_window(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_LIMIT", "1")
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_WINDOW", "10")

    clock = {"now": 1000.0}
    monkeypatch.setattr(discord_platform.time, "monotonic", lambda: clock["now"])

    assert adapter._discord_bot_exchange_allowed("10") is True
    assert adapter._discord_bot_exchange_allowed("10") is False
    clock["now"] += 11
    assert adapter._discord_bot_exchange_allowed("10") is True


def test_bot_exchange_guard_disabled_by_zero_limit(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setenv("DISCORD_BOT_EXCHANGE_LIMIT", "0")
    assert adapter._discord_bot_exchange_allowed("10") is False


# --- agent-facing etiquette -------------------------------------------------

def test_etiquette_text_reflects_room_shape(monkeypatch):
    adapter = _make_adapter(monkeypatch, extra={"require_mention": "auto"})

    shared = adapter._discord_room_etiquette(_shared_channel())
    assert shared is not None and "2 bots" in shared and "@mention" in shared

    solo = adapter._discord_room_etiquette(_solo_channel())
    assert solo is not None and "only bot" in solo

    assert adapter._discord_room_etiquette(FakeDMChannel()) is None
