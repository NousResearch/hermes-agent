"""Tests for Discord missed-message startup backfill."""

import datetime as dt
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome


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

import discord  # noqa: E402
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
    bot_user = SimpleNamespace(id=999, bot=True, display_name="Hermes", name="hermes")
    adapter._client = SimpleNamespace(user=bot_user, get_channel=lambda _id: None)
    adapter._ready_event.set()
    adapter._handle_message = AsyncMock()
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    return adapter


def make_message(*, message_id=1, author_id=42, content="please ingest", reactions=None, channel=None, mentions=None):
    channel = channel or FakeChannel()
    return SimpleNamespace(
        id=message_id,
        content=content,
        reactions=list(reactions or []),
        author=SimpleNamespace(id=author_id, bot=False, display_name="Emo", name="emo"),
        channel=channel,
        guild=getattr(channel, "guild", None),
        created_at=datetime.now(timezone.utc),
        attachments=[],
        mentions=list(mentions or []),
        reference=None,
        type=discord.MessageType.default,
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
async def test_generic_unavailable_response_counts_as_completed(adapter):
    bot_reply = SimpleNamespace(
        id=2,
        content="That package is unavailable on this platform.",
        author=SimpleNamespace(id=999, bot=True),
        reference=SimpleNamespace(message_id=1),
        created_at=datetime.now(timezone.utc),
    )
    channel = FakeChannel(history_messages=[bot_reply])
    message = make_message(message_id=1, channel=channel)

    assert await adapter._should_backfill_discord_message(message) is False


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

    adapter._handle_message.assert_awaited_once_with(message, role_authorized=False)


@pytest.mark.asyncio
async def test_run_backfill_counts_only_messages_that_reach_dispatch(adapter, monkeypatch):
    dropped = make_message(message_id=1)
    accepted = make_message(message_id=2)

    async def fake_candidates(_channels):
        yield dropped
        yield accepted

    async def fake_dispatch(message):
        return message is accepted

    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", fake_candidates)
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=True))
    dispatch = AsyncMock(side_effect=fake_dispatch)
    monkeypatch.setattr(adapter, "_dispatch_recovered_message", dispatch)
    monkeypatch.setattr(adapter, "_missed_message_backfill_max_dispatches", lambda: 1)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})

    await adapter._run_missed_message_backfill()

    assert dispatch.await_count == 2


@pytest.mark.asyncio
async def test_recovered_mention_reuses_live_auth_and_mention_gates(adapter, monkeypatch):
    bot_user = adapter._client.user
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    denied = make_message(
        message_id=1,
        author_id=41,
        content=f"<@{bot_user.id}> denied",
        mentions=[bot_user],
    )
    allowed = make_message(
        message_id=2,
        content=f"<@{bot_user.id}> allowed",
        mentions=[bot_user],
    )

    monkeypatch.setattr(
        adapter,
        "_is_allowed_user",
        lambda user_id, *_a, **_kw: user_id == str(allowed.author.id),
    )

    assert await adapter._dispatch_recovered_message(denied) is False
    assert await adapter._dispatch_recovered_message(allowed) is True
    adapter._handle_message.assert_awaited_once_with(allowed, role_authorized=False)


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


def test_default_config_exposes_missed_message_backfill_settings():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["discord"]["missed_message_backfill"] == {
        "enabled": False,
        "channels": "",
        "window_seconds": 21600,
        "limit": 100,
        "max_dispatches": 10,
    }


def test_default_recovery_scope_includes_allowed_and_free_response_channels(adapter, monkeypatch):
    monkeypatch.delenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", raising=False)
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "100,200")
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "200,300")

    assert adapter._missed_message_backfill_channels() == {"100", "200", "300"}


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


def test_recovery_ledger_prunes_expired_rows(adapter):
    old = (datetime.now(timezone.utc) - dt.timedelta(days=31)).isoformat()

    def insert_old_rows(conn):
        conn.execute(
            "INSERT INTO discord_messages "
            "(message_id, status, updated_at) VALUES ('old-message', 'responded', ?)",
            (old,),
        )
        conn.execute(
            "INSERT INTO discord_recovery_scans "
            "(scan_id, started_at, completed_at, status, channels, window_seconds, limit_count) "
            "VALUES ('old-scan', ?, ?, 'success', '[]', 3600, 10)",
            (old, old),
        )

    adapter._with_discord_recovery_db(insert_old_rows)
    adapter._with_discord_recovery_db(lambda _conn: None)

    def count_old(conn):
        messages = conn.execute(
            "SELECT COUNT(*) FROM discord_messages WHERE message_id='old-message'"
        ).fetchone()[0]
        scans = conn.execute(
            "SELECT COUNT(*) FROM discord_recovery_scans WHERE scan_id='old-scan'"
        ).fetchone()[0]
        return messages, scans

    assert adapter._with_discord_recovery_db(count_old) == (0, 0)


def test_empty_successful_turn_is_not_persistently_complete(adapter):
    message = make_message(message_id=89)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )
    adapter._record_discord_processing_start(event, emoji_ack=False)
    adapter._record_discord_processing_complete(event, outcome=ProcessingOutcome.SUCCESS)

    assert adapter._discord_message_is_persistently_complete("89") is False


def test_disabled_recovery_does_not_create_hot_path_ledger(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "false")
    message = make_message(message_id=90)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )

    adapter._record_discord_processing_start(event, emoji_ack=False)
    adapter._record_discord_processing_complete(event, ProcessingOutcome.SUCCESS)
    adapter._record_discord_response(
        reply_to="90",
        result=SimpleNamespace(success=True, message_id="9003"),
        content="Done",
    )

    db_path = adapter._discord_recovery_db_path()
    assert not db_path.exists()


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


@pytest.mark.asyncio
async def test_iter_candidates_applies_one_global_scan_limit(adapter, monkeypatch):
    first = FakeChannel(
        channel_id=123,
        history_messages=[make_message(message_id=1), make_message(message_id=2)],
    )
    second = FakeChannel(
        channel_id=456,
        history_messages=[make_message(message_id=3), make_message(message_id=4)],
    )
    adapter._client.get_channel = lambda channel_id: {123: first, 456: second}[channel_id]
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 3)

    got = []
    async for msg in adapter._iter_missed_message_backfill_candidates({"123", "456"}):
        got.append(msg.id)

    assert got == [1, 2, 3]


@pytest.mark.asyncio
async def test_iter_candidates_keeps_latest_messages_when_window_exceeds_limit(adapter, monkeypatch):
    class RealisticChannel(FakeChannel):
        def history(self, **kwargs):
            async def _gen():
                messages = list(self._history_messages)
                if not kwargs["oldest_first"]:
                    messages.reverse()
                for message in messages[:kwargs["limit"]]:
                    yield message

            return _gen()

    channel = RealisticChannel(
        channel_id=123,
        history_messages=[
            make_message(message_id=1),
            make_message(message_id=2),
            make_message(message_id=3),
            make_message(message_id=4),
        ],
    )
    adapter._client.get_channel = lambda _channel_id: channel
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 3)

    got = []
    async for msg in adapter._iter_missed_message_backfill_candidates({"123"}):
        got.append(msg.id)

    assert got == [2, 3, 4]
