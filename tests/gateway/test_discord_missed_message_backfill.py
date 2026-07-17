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
async def test_backfills_message_with_only_own_success_reaction(adapter):
    message = make_message(reactions=[FakeReaction("✅", me=True)])

    assert await adapter._should_backfill_discord_message(message) is True


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
async def test_recovery_aborts_when_durable_ledger_is_unavailable(adapter, monkeypatch):
    dispatch = AsyncMock()
    monkeypatch.setattr(adapter, "_dispatch_recovered_message", dispatch)
    monkeypatch.setattr(
        adapter,
        "_with_discord_recovery_db_async",
        AsyncMock(return_value=False),
    )

    await adapter._run_missed_message_backfill()

    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_releases_dedup_claim_when_dispatch_is_cancelled(adapter, monkeypatch):
    message = make_message(message_id=97)
    started = asyncio.Event()

    async def cancelled_dispatch(_message):
        adapter._dedup.is_duplicate(str(message.id))
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_dispatch_recovered_message", cancelled_dispatch)
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", candidates)
    task = asyncio.create_task(adapter._run_missed_message_backfill())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter._dedup.contains(str(message.id)) is False


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
async def test_recovery_task_joins_gateway_startup_restore(adapter, monkeypatch):
    release = asyncio.Event()

    async def recovery():
        await release.wait()

    runner = SimpleNamespace(
        _startup_restore_in_progress=True,
        _startup_restore_tasks=[],
    )
    adapter.gateway_runner = runner
    monkeypatch.setattr(adapter, "_run_missed_message_backfill", recovery)

    task = adapter._ensure_missed_message_backfill_task()

    assert runner._startup_restore_tasks == [task]
    release.set()
    await task


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


@pytest.mark.asyncio
async def test_recovery_does_not_treat_unmentioned_message_as_dispatched(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    adapter.config.extra["free_response_channels"] = ""
    adapter.handle_message = AsyncMock()
    message = make_message(message_id=95, content="not addressed")

    assert await adapter._dispatch_recovered_message(message) is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_messages_bypass_live_text_debounce(adapter, monkeypatch):
    bot_user = adapter._client.user
    message = make_message(
        message_id=96,
        content=f"<@{bot_user.id}> recover",
        mentions=[bot_user],
    )
    adapter._text_batch_delay_seconds = 0.6
    adapter._handle_message = DiscordAdapter._handle_message.__get__(
        adapter, DiscordAdapter
    )
    adapter.handle_message = AsyncMock()
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    assert await adapter._dispatch_recovered_message(message) is True
    adapter.handle_message.assert_awaited_once()
    assert adapter._pending_text_batches == {}


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

    config = load_gateway_config()
    backfill = config.platforms[Platform.DISCORD].extra[
        "missed_message_backfill"
    ]

    assert backfill == {
        "enabled": True,
        "channels": ["1501971993405292796"],
        "window_seconds": 3600,
        "limit": 25,
        "max_dispatches": 3,
    }


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


def test_recovery_store_pins_profile_home_at_adapter_construction(monkeypatch, tmp_path):
    first_home = tmp_path / "first"
    second_home = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first_home))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="one"))
    monkeypatch.setenv("HERMES_HOME", str(second_home))

    assert adapter._discord_recovery_db_path() == (
        first_home / "gateway" / "discord_message_recovery.db"
    )


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
        final=True,
    )

    assert await adapter._should_backfill_discord_message(message) is False


def test_down_notice_response_does_not_mark_message_complete(adapter):
    adapter._record_discord_response(
        reply_to="88",
        result=SimpleNamespace(success=False, message_id="9002"),
        content="The agent is down right now.",
        final=True,
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


def test_fresh_processing_claim_suppresses_duplicate_recovery(adapter):
    message = make_message(message_id=99)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )
    adapter._record_discord_processing_start(event, emoji_ack=False)

    assert adapter._discord_message_has_active_claim("99") is True


def test_stale_processing_claim_is_recoverable(adapter):
    message = make_message(message_id=100)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )
    adapter._record_discord_processing_start(event, emoji_ack=False)
    stale = (datetime.now(timezone.utc) - dt.timedelta(minutes=11)).isoformat()
    adapter._with_discord_recovery_db(
        lambda conn: conn.execute(
            "UPDATE discord_messages SET updated_at=? WHERE message_id='100'",
            (stale,),
        )
    )

    assert adapter._discord_message_has_active_claim("100") is False


@pytest.mark.asyncio
async def test_processing_hook_offloads_contended_ledger(adapter, monkeypatch):
    message = make_message(message_id=101)
    event = MessageEvent(
        text=message.content,
        message_type=MessageType.TEXT,
        raw_message=message,
        message_id=str(message.id),
    )

    def slow_record(*_args, **_kwargs):
        import time
        time.sleep(0.1)

    monkeypatch.setattr(adapter, "_record_discord_processing_start", slow_record)
    processing = asyncio.create_task(adapter.on_processing_start(event))
    await asyncio.sleep(0.01)

    assert processing.done() is False
    await processing


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


def test_preview_delivery_does_not_mark_message_complete(adapter):
    adapter._record_discord_response(
        reply_to="92",
        result=SimpleNamespace(success=True, message_id="9005"),
        content="partial",
        final=False,
    )

    assert adapter._discord_message_is_persistently_complete("92") is False


def test_successful_final_delivery_clears_prior_outage_state(adapter):
    adapter._record_discord_response(
        reply_to="93",
        result=SimpleNamespace(success=False, message_id="9006"),
        content="Hermes is offline",
        final=True,
    )
    assert adapter._discord_message_is_persistently_complete("93") is False

    adapter._record_discord_response(
        reply_to="93",
        result=SimpleNamespace(success=True, message_id="9007"),
        content="Recovered successfully",
        final=True,
    )

    assert adapter._discord_message_is_persistently_complete("93") is True


@pytest.mark.asyncio
async def test_send_uses_notify_metadata_as_final_delivery_signal(adapter):
    channel = FakeChannel(channel_id=123)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=9008))
    channel.fetch_message = AsyncMock()
    adapter._client.get_channel = lambda _channel_id: channel

    preview = await adapter.send(
        "123",
        "partial",
        reply_to="94",
        metadata={"expect_edits": True},
    )
    assert preview.success is True
    assert adapter._discord_message_is_persistently_complete("94") is False

    final = await adapter.send(
        "123",
        "complete",
        reply_to="94",
        metadata={"notify": True},
    )
    assert final.success is True
    assert adapter._discord_message_is_persistently_complete("94") is True


@pytest.mark.asyncio
async def test_final_stream_edit_marks_original_request_complete(adapter):
    channel = FakeChannel(channel_id=123)
    message = SimpleNamespace(edit=AsyncMock())
    channel.fetch_message = AsyncMock(return_value=message)
    adapter._client.get_channel = lambda _channel_id: channel

    result = await adapter.edit_message(
        "123",
        "9009",
        "complete streamed response",
        finalize=True,
        metadata={"reply_to_message_id": "102"},
    )

    assert result.success is True
    assert adapter._discord_message_is_persistently_complete("102") is True


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
        final=True,
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

    assert len(got) == 3
    assert set(got).issubset({1, 2, 3, 4})


@pytest.mark.asyncio
async def test_iter_candidates_round_robins_configured_channels(adapter, monkeypatch):
    first = FakeChannel(
        channel_id=123,
        history_messages=[
            make_message(message_id=1),
            make_message(message_id=2),
            make_message(message_id=3),
        ],
    )
    second = FakeChannel(
        channel_id=456,
        history_messages=[make_message(message_id=4)],
    )
    adapter._client.get_channel = lambda channel_id: {123: first, 456: second}[channel_id]
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 3)

    got = []
    async for message in adapter._iter_missed_message_backfill_candidates({"123", "456"}):
        got.append(message.id)

    assert 4 in got


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


def test_recovery_cursor_round_trip_is_channel_scoped(adapter):
    adapter._advance_discord_recovery_cursor("123", "1001")
    adapter._advance_discord_recovery_cursor("456", "2002")

    assert adapter._discord_recovery_cursor("123") == "1001"
    assert adapter._discord_recovery_cursor("456") == "2002"


@pytest.mark.asyncio
async def test_cursor_does_not_advance_past_incomplete_dispatched_message(adapter, monkeypatch):
    channel = FakeChannel(
        channel_id=123,
        history_messages=[
            make_message(message_id=1),
            make_message(message_id=2),
        ],
    )
    for message in channel._history_messages:
        message.channel = channel
    adapter._client.get_channel = lambda _channel_id: channel
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_dispatch_recovered_message", AsyncMock(side_effect=[True, True]))
    monkeypatch.setattr(adapter, "_missed_message_backfill_max_dispatches", lambda: 10)

    await adapter._run_missed_message_backfill()

    assert adapter._discord_recovery_cursor("123") is None


def test_final_delivery_advances_channel_cursor(adapter):
    message = make_message(message_id=103, channel=FakeChannel(channel_id=123))
    adapter._record_discord_message_seen(message, status="processing")

    adapter._record_discord_response(
        reply_to="103",
        result=SimpleNamespace(success=True, message_id="9010"),
        content="done",
        final=True,
    )

    assert adapter._discord_recovery_cursor("123") == "103"


@pytest.mark.asyncio
async def test_iter_candidates_uses_persisted_channel_cursor(adapter, monkeypatch):
    class CursorChannel(FakeChannel):
        def history(self, **kwargs):
            self.history_kwargs = kwargs

            async def _gen():
                yield make_message(message_id=11, channel=self)

            return _gen()

    channel = CursorChannel(channel_id=123)
    adapter._client.get_channel = lambda _channel_id: channel
    adapter._advance_discord_recovery_cursor("123", "10")
    monkeypatch.setattr(discord, "Object", lambda *, id: SimpleNamespace(id=id))

    got = []
    async for message in adapter._iter_missed_message_backfill_candidates({"123"}):
        got.append(message.id)

    assert got == [11]
    assert getattr(channel.history_kwargs["after"], "id", None) == 10
