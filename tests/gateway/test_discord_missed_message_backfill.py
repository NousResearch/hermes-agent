"""Tests for Discord missed-message startup backfill."""

import asyncio
import datetime as dt
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
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
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
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
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    _apply_yaml_config,
)


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
        self.guild = SimpleNamespace(id=777, name="emo")
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
    adapter._handle_message = AsyncMock(return_value=True)
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


def make_bot_message(*, message_id=1, content="please ingest", channel=None, mentions=None):
    message = make_message(
        message_id=message_id,
        content=content,
        channel=channel,
        mentions=mentions,
    )
    message.author.bot = True
    return message


@pytest.mark.asyncio
async def test_configured_bot_sender_is_left_for_shared_ingress_policy(adapter, monkeypatch):
    bot_user = adapter._client.user
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    message = make_bot_message(
        message_id=98,
        content=f"<@{bot_user.id}> run this",
        mentions=[bot_user],
    )

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
async def test_thread_unreferenced_bot_message_does_not_mask_request(adapter):
    bot_post = SimpleNamespace(
        id=2,
        content="Done — captured a different request.",
        author=SimpleNamespace(id=999, bot=True),
        reference=None,
        created_at=datetime.now(timezone.utc),
    )
    thread = FakeChannel(channel_id=456, parent_id=123, history_messages=[bot_post])
    message = make_message(message_id=1, channel=thread)

    assert await adapter._should_backfill_discord_message(message) is True


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
    bot_user = adapter._client.user
    message = make_message(
        message_id=1,
        content=f"<@{bot_user.id}> please ingest",
        mentions=[bot_user],
    )

    async def fake_candidates(_channels):
        yield message

    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS", "123")
    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", fake_candidates)
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_missed_message_backfill_max_dispatches", lambda: 10)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    await adapter._run_missed_message_backfill()

    adapter._handle_message.assert_awaited_once_with(
        message,
        role_authorized=False,
        recovered=True,
    )


@pytest.mark.asyncio
async def test_repeated_ready_coalesces_instead_of_cancelling_active_recovery(adapter):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_recovery():
        started.set()
        await release.wait()

    first = asyncio.create_task(slow_recovery())
    adapter._missed_message_backfill_task = first
    await started.wait()

    second = adapter._ensure_missed_message_backfill_task()

    assert second is first
    assert first.cancelled() is False
    release.set()
    await first


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
    adapter._handle_message.assert_awaited_once_with(
        allowed,
        role_authorized=False,
        recovered=True,
    )


def test_default_config_exposes_missed_message_backfill_settings():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["discord"]["missed_message_backfill"] == {
        "enabled": False,
        "channels": "",
        "window_seconds": 21600,
        "limit": 100,
        "max_dispatches": 10,
    }


def test_missed_message_backfill_config_stays_per_adapter():
    first_extra = _apply_yaml_config(
        {},
        {
            "missed_message_backfill": {
                "enabled": True,
                "channels": ["111"],
                "window_seconds": 60,
                "limit": 5,
                "max_dispatches": 2,
            }
        },
    )
    second_extra = _apply_yaml_config(
        {},
        {
            "missed_message_backfill": {
                "enabled": False,
                "channels": ["222"],
                "window_seconds": 120,
                "limit": 6,
                "max_dispatches": 3,
            }
        },
    )

    first = DiscordAdapter(PlatformConfig(enabled=True, token="one", extra=first_extra or {}))
    second = DiscordAdapter(PlatformConfig(enabled=True, token="two", extra=second_extra or {}))

    assert first._missed_message_backfill_enabled() is True
    assert first._missed_message_backfill_channels() == {"111"}
    assert first._missed_message_backfill_window_seconds() == 60
    assert first._missed_message_backfill_limit() == 5
    assert first._missed_message_backfill_max_dispatches() == 2
    assert second._missed_message_backfill_enabled() is False
    assert second._missed_message_backfill_channels() == {"222"}
    assert second._missed_message_backfill_window_seconds() == 120
    assert second._missed_message_backfill_limit() == 6
    assert second._missed_message_backfill_max_dispatches() == 3


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
    adapter._discord_recovery_store._initialized = False
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


@pytest.mark.asyncio
async def test_send_offloads_final_delivery_ledger_write(adapter, monkeypatch):
    channel = FakeChannel(channel_id=123)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=9011))
    channel.fetch_message = AsyncMock()
    adapter._client.get_channel = lambda _channel_id: channel

    def slow_record(**_kwargs):
        import time
        time.sleep(0.1)

    monkeypatch.setattr(adapter, "_record_discord_response", slow_record)
    sending = asyncio.create_task(
        adapter.send(
            "123",
            "done",
            reply_to="104",
            metadata={"notify": True},
        )
    )
    await asyncio.sleep(0.01)

    assert sending.done() is False
    assert (await sending).success is True


def test_final_delivery_remains_complete_after_processing_hook(adapter):
    message = make_message(message_id=91)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )

    adapter._record_discord_processing_start(event, emoji_ack=False)
    adapter._record_discord_response(
        reply_to="91",
        result=SimpleNamespace(success=True, message_id="9004"),
        content="Done",
        final=True,
    )
    adapter._record_discord_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert adapter._discord_message_is_persistently_complete("91") is True


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


