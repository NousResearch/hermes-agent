"""Tests for Discord missed-message startup backfill."""

import os
import sys
from datetime import datetime, timezone
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


class FakeReaction:
    def __init__(self, emoji, *, me=False, users=None):
        self.emoji = emoji
        self.me = me
        self._users = list(users or [])

    async def users(self):
        for user in self._users:
            yield user


class FakeChannel:
    def __init__(self, channel_id=123, history_messages=None, parent_id=None):
        self.id = channel_id
        self.parent_id = parent_id
        self.name = "wiki-inbox"
        self.guild = SimpleNamespace(name="emo")
        self.topic = None
        self._history_messages = list(history_messages or [])

    def history(self, **kwargs):
        async def _gen():
            for message in self._history_messages:
                yield message

        return _gen()


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999), get_channel=lambda _id: None)
    adapter._handle_message = AsyncMock()
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    return adapter


def make_message(*, message_id=1, author_id=42, content="please ingest", reactions=None, channel=None):
    channel = channel or FakeChannel()
    return SimpleNamespace(
        id=message_id,
        content=content,
        reactions=list(reactions or []),
        author=SimpleNamespace(id=author_id, bot=False, display_name="Emo", name="emo"),
        channel=channel,
        created_at=datetime.now(timezone.utc),
        attachments=[],
        mentions=[],
        reference=None,
        type=None,
    )


@pytest.mark.asyncio
async def test_backfills_message_with_only_own_success_reaction(adapter):
    message = make_message(reactions=[FakeReaction("✅", me=True)])

    assert await adapter._should_backfill_discord_message(message) is True


@pytest.mark.asyncio
async def test_should_not_backfill_message_with_non_down_bot_response(adapter):
    bot_reply = SimpleNamespace(
        id=2,
        content="Done — captured it.",
        author=SimpleNamespace(id=999, bot=True),
        reference=SimpleNamespace(message_id=1),
        created_at=datetime.now(timezone.utc),
    )
    channel = FakeChannel(history_messages=[bot_reply])
    message = make_message(message_id=1, channel=channel)

    assert await adapter._should_backfill_discord_message(message) is False


@pytest.mark.asyncio
async def test_parent_channel_unreferenced_bot_message_does_not_suppress_backfill(adapter):
    unrelated_bot_post = SimpleNamespace(
        id=2,
        content="Done — captured a different item.",
        author=SimpleNamespace(id=999, bot=True),
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    channel = FakeChannel(history_messages=[unrelated_bot_post])
    message = make_message(message_id=1, channel=channel)

    assert await adapter._should_backfill_discord_message(message) is True


@pytest.mark.asyncio
async def test_thread_unreferenced_bot_message_suppresses_backfill(adapter):
    bot_post = SimpleNamespace(
        id=2,
        content="Done — captured it.",
        author=SimpleNamespace(id=999, bot=True),
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    thread = FakeChannel(channel_id=456, parent_id=123, history_messages=[bot_post])
    message = make_message(message_id=1, channel=thread)

    assert await adapter._should_backfill_discord_message(message) is False


@pytest.mark.asyncio
async def test_backfills_when_only_down_notice_exists(adapter):
    down_notice = SimpleNamespace(
        id=2,
        content="The agent is down right now.",
        author=SimpleNamespace(id=999, bot=True),
        reference=SimpleNamespace(message_id=1),
        created_at=datetime.now(timezone.utc),
    )
    channel = FakeChannel(history_messages=[down_notice])
    message = make_message(message_id=1, channel=channel)

    assert await adapter._should_backfill_discord_message(message) is True


@pytest.mark.asyncio
async def test_run_backfill_dispatches_unaddressed_messages(adapter, monkeypatch):
    message = make_message(message_id=1)

    async def fake_candidates(_channels):
        yield message

    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "123")
    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", fake_candidates)
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_missed_message_backfill_max_dispatches", lambda: 10)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    await adapter._run_missed_message_backfill()

    adapter._handle_message.assert_awaited_once_with(message)


def test_missed_message_backfill_config_bridge(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "DISCORD_MISSED_MESSAGE_BACKFILL",
        "DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS",
        "DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS",
        "DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT",
        "DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES",
    ):
        monkeypatch.delenv(key, raising=False)

    (tmp_path / "config.yaml").write_text(
        "platforms:\n"
        "  discord:\n"
        "    enabled: true\n"
        "discord:\n"
        "  missed_message_backfill:\n"
        "    enabled: true\n"
        "    channels: ['1501971993405292796']\n"
        "    window_seconds: 3600\n"
        "    limit: 25\n"
        "    max_dispatches: 3\n"
    )

    load_gateway_config()

    assert os.environ["DISCORD_MISSED_MESSAGE_BACKFILL"] == "true"
    assert os.environ["DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS"] == "1501971993405292796"
    assert os.environ["DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS"] == "3600"
    assert os.environ["DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT"] == "25"
    assert os.environ["DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES"] == "3"


@pytest.mark.asyncio
async def test_persistent_responded_record_suppresses_backfill(adapter):
    message = make_message(message_id=77)
    adapter._record_discord_message_seen(message, status="responded")
    adapter._record_discord_response(
        reply_to="77",
        result=SimpleNamespace(success=True, message_id="9001"),
        content="Done — captured it.",
    )

    assert await adapter._should_backfill_discord_message(message) is False


def test_down_notice_response_does_not_mark_message_complete(adapter):
    adapter._record_discord_response(
        reply_to="88",
        result=SimpleNamespace(success=True, message_id="9002"),
        content="The agent is down right now.",
    )

    assert adapter._discord_message_is_persistently_complete("88") is False


@pytest.mark.asyncio
async def test_iter_candidates_includes_active_and_archived_threads(adapter):
    active_msg = make_message(message_id=201, channel=FakeChannel(channel_id=2010))
    archived_msg = make_message(message_id=202, channel=FakeChannel(channel_id=2020))
    active_thread = FakeChannel(channel_id=2010, history_messages=[active_msg])
    archived_thread = FakeChannel(channel_id=2020, history_messages=[archived_msg])

    class ParentChannel(FakeChannel):
        threads = [active_thread]

        def archived_threads(self, **kwargs):
            async def _gen():
                yield archived_thread
            return _gen()

    parent = ParentChannel(channel_id=123, history_messages=[])
    adapter._client.get_channel = lambda _id: parent

    got = []
    async for msg in adapter._iter_missed_message_backfill_candidates({"123"}):
        got.append(msg.id)

    assert got == [201, 202]
