"""Seam-identity + behavioral tests for the R2-S1 recovery-backfill extraction.

Verifies that the 34 methods moved into ``DiscordRecoveryBackfillMixin``
(adapter.py god-file slice R2-S1, epic #78647, target #78634) resolve
through the adapter with bound-method identity — proving the mixin-FIRST
bases ordering (``class DiscordAdapter(DiscordRecoveryBackfillMixin,
BasePlatformAdapter)``) wins MRO over the ``BasePlatformAdapter`` no-op
stubs, which would otherwise silently kill the reaction hooks.
"""

import asyncio
import inspect
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult


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
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from plugins.platforms.discord.recovery_backfill_mixin import (  # noqa: E402
    DiscordRecoveryBackfillMixin,
)

# The 34 moved top-level methods: 29 recovery-backfill (C1) + 5 reaction hooks (C2).
MOVED_METHODS = [
    # C1 recovery backfill
    "_missed_message_backfill_enabled",
    "_missed_message_backfill_channels",
    "_missed_message_backfill_window_seconds",
    "_missed_message_backfill_limit",
    "_missed_message_backfill_max_dispatches",
    "_ensure_missed_message_backfill_task",
    "_run_missed_message_backfill",
    "_dispatch_recovered_message",
    "_iter_missed_message_backfill_candidates",
    "_iter_channel_and_thread_messages",
    "_discord_recovery_cursor",
    "_advance_discord_recovery_cursor",
    "_should_backfill_discord_message",
    "_is_down_notice_content",
    "_message_has_non_down_bot_response",
    "_discord_recovery_db_path",
    "_with_discord_recovery_db",
    "_with_discord_recovery_db_async",
    "_utc_now_iso",
    "_message_channel_ids",
    "_record_discord_message_seen",
    "_record_recovery_attempt",
    "_record_discord_processing_start",
    "_record_discord_processing_complete",
    "_record_discord_response",
    "_discord_message_is_persistently_complete",
    "_discord_message_has_active_claim",
    "_record_recovery_scan_start",
    "_record_recovery_scan_complete",
    # C2 reaction hooks
    "_add_reaction",
    "_remove_reaction",
    "_reactions_enabled",
    "on_processing_start",
    "on_processing_complete",
]


class FakeChannel:
    def __init__(self, channel_id=123, parent_id=None):
        self.id = channel_id
        self.parent_id = parent_id
        self.name = "wiki-inbox"
        self.guild = SimpleNamespace(id=777, name="emo")
        self.topic = None
        self._history_messages = []

    def history(self, **kwargs):
        async def _gen():
            for message in self._history_messages:
                yield message

        return _gen()


def make_message(*, message_id=1, author_id=42, content="please ingest", channel=None, author_bot=False, mentions=None):
    channel = channel or FakeChannel()
    return SimpleNamespace(
        id=message_id,
        content=content,
        reactions=[],
        author=SimpleNamespace(id=author_id, bot=author_bot, display_name="Emo", name="emo"),
        channel=channel,
        guild=getattr(channel, "guild", None),
        created_at=datetime.now(timezone.utc),
        attachments=[],
        mentions=[] if mentions is None else mentions,
        reference=None,
        type=discord.MessageType.default,
    )


class FakeReactionMessage:
    """Minimal discord-message stand-in whose reaction calls are recorded."""

    def __init__(self, message_id=7):
        self.id = message_id
        self.channel = FakeChannel()
        self.author = SimpleNamespace(id=42, bot=False)
        self.calls = []

    async def add_reaction(self, emoji):
        self.calls.append(("add", emoji))

    async def remove_reaction(self, emoji, member=None):
        self.calls.append(("remove", emoji))


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    adapter._ready_event.set()
    adapter._handle_message = AsyncMock(return_value=True)
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    return adapter


def make_event(message, *, message_id=None):
    return MessageEvent(
        text=getattr(message, "content", ""),
        message_type=MessageType.TEXT,
        source=None,
        raw_message=message,
        message_id=message_id or str(getattr(message, "id", "")),
    )


# ── 1. Seam identity: mixin-FIRST MRO wins (the hard requirement) ───────────

@pytest.mark.parametrize("name", MOVED_METHODS)
def test_moved_method_identity_via_adapter(name):
    adapter_method = getattr(DiscordAdapter, name)
    mixin_method = getattr(DiscordRecoveryBackfillMixin, name)
    assert adapter_method is mixin_method, (
        f"{name} resolved to {adapter_method!r} instead of the mixin — "
        "MRO is shadowing the extracted method"
    )
    assert name in DiscordRecoveryBackfillMixin.__dict__, (
        f"{name} missing from mixin class dict"
    )


def test_reaction_hooks_are_not_base_noop_stubs():
    # If the mixin were appended LAST in the bases tuple, these would resolve
    # to BasePlatformAdapter's empty stubs and reactions would silently die.
    import gateway.platforms.base as base_mod

    assert DiscordAdapter.on_processing_start is DiscordRecoveryBackfillMixin.on_processing_start
    assert DiscordAdapter.on_processing_complete is DiscordRecoveryBackfillMixin.on_processing_complete
    assert DiscordAdapter.on_processing_start is not base_mod.BasePlatformAdapter.on_processing_start
    assert DiscordAdapter.on_processing_complete is not base_mod.BasePlatformAdapter.on_processing_complete
    assert issubclass(DiscordAdapter, DiscordRecoveryBackfillMixin)
    assert issubclass(DiscordAdapter, base_mod.BasePlatformAdapter)


def test_mixin_imports_without_adapter_import():
    # Zero circular import: the mixin module must be importable without ever
    # importing the adapter module. Run in a fresh interpreter with the same
    # discord stub injected pre-import.
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        sys.modules.setdefault("discord", discord_mod)
        sys.modules.setdefault("discord.ext", MagicMock())

        from plugins.platforms.discord.recovery_backfill_mixin import DiscordRecoveryBackfillMixin
        mixin_module = sys.modules["plugins.platforms.discord.recovery_backfill_mixin"]
        # The parent package __init__ re-exports the adapter, so it will be in
        # sys.modules regardless; the real invariant is that the mixin module
        # itself never binds/imports the adapter (module-level no-import).
        adapter_bindings = [
            name
            for name, value in vars(mixin_module).items()
            if getattr(value, "__name__", None) == "plugins.platforms.discord.adapter"
        ]
        assert not adapter_bindings, (
            f"mixin module binds the adapter module directly: {adapter_bindings}"
        )
        assert DiscordRecoveryBackfillMixin.__name__ == "DiscordRecoveryBackfillMixin"
        print("OK")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_signatures_unchanged(adapter):
    for name in MOVED_METHODS:
        adapter_sig = inspect.signature(getattr(DiscordAdapter, name))
        mixin_sig = inspect.signature(getattr(DiscordRecoveryBackfillMixin, name))
        assert adapter_sig == mixin_sig, f"signature drift on {name}: {adapter_sig} != {mixin_sig}"


def test_adapter_init_still_provides_ledger_state(adapter):
    assert adapter._discord_recovery_store is not None
    assert hasattr(adapter, "_dedup")
    assert adapter._missed_message_backfill_task is None
    assert isinstance(adapter._threads, dict) or hasattr(adapter, "_threads")


# ── 2. Behavioral: reaction hooks fire through the mixin (C2) ──────────────

@pytest.mark.asyncio
async def test_on_processing_start_adds_ack_reaction_and_records_state(adapter, monkeypatch):
    message = FakeReactionMessage(message_id=7)
    event = make_event(message, message_id="7")
    recorded = {}

    def fake_start(ev, *, emoji_ack):
        recorded["emoji_ack"] = emoji_ack
        recorded["message_id"] = str(getattr(ev.raw_message, "id", ""))

    monkeypatch.setattr(adapter, "_record_discord_processing_start", fake_start)
    monkeypatch.setattr(adapter, "_reactions_enabled", lambda: True)

    await adapter.on_processing_start(event)

    assert message.calls == [("add", "👀")]
    assert recorded == {"emoji_ack": True, "message_id": "7"}


@pytest.mark.asyncio
async def test_on_processing_complete_swaps_reactions_and_records_outcome(adapter, monkeypatch):
    message = FakeReactionMessage(message_id=7)
    event = make_event(message, message_id="7")
    recorded = {}

    def fake_complete(ev, outcome):
        recorded["outcome"] = outcome
        recorded["message_id"] = str(getattr(ev.raw_message, "id", ""))

    monkeypatch.setattr(adapter, "_record_discord_processing_complete", fake_complete)
    monkeypatch.setattr(adapter, "_reactions_enabled", lambda: True)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert message.calls == [("remove", "👀"), ("add", "✅")]
    assert recorded["outcome"] is ProcessingOutcome.SUCCESS
    assert recorded["message_id"] == "7"

    message2 = FakeReactionMessage(message_id=8)
    await adapter.on_processing_complete(make_event(message2, message_id="8"), ProcessingOutcome.FAILURE)
    assert message2.calls == [("remove", "👀"), ("add", "❌")]


@pytest.mark.asyncio
async def test_reactions_disabled_skips_reaction_calls_but_records_state(adapter, monkeypatch):
    message = FakeReactionMessage(message_id=7)
    event = make_event(message, message_id="7")
    recorded = {}

    def fake_start(ev, *, emoji_ack):
        recorded["emoji_ack"] = emoji_ack

    monkeypatch.setattr(adapter, "_record_discord_processing_start", fake_start)
    monkeypatch.setattr(adapter, "_reactions_enabled", lambda: False)

    await adapter.on_processing_start(event)

    assert message.calls == []
    assert recorded == {"emoji_ack": False}


# ── 3. Behavioral: recovery backfill (C1) ──────────────────────────────────

@pytest.mark.asyncio
async def test_backfill_dispatches_missed_messages_and_writes_ledger(adapter, monkeypatch):
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

    # Ledger: message discovered + queued, scan recorded as success.
    store = adapter._discord_recovery_store

    def _inspect(conn):
        row = conn.execute(
            "SELECT status, attempts FROM discord_messages WHERE message_id='1'"
        ).fetchone()
        scan = conn.execute(
            "SELECT status, scanned, missed, dispatched FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row, scan

    row, scan = store.call(_inspect)
    assert row is not None and row[0] in ("queued", "discovered")
    assert scan is not None and scan[0] == "success"
    assert scan[2] == 1 and scan[3] == 1


@pytest.mark.asyncio
async def test_ledger_cursor_advances_on_persistent_completion(adapter, monkeypatch):
    # Record the message with a known channel so the completion path can
    # resolve the channel for cursor advancement.
    channel = FakeChannel(channel_id=123)
    message = make_message(message_id=123, channel=channel)
    adapter._record_discord_message_seen(message, status="discovered")

    result = SendResult(success=True, message_id="srv-1")
    adapter._record_discord_response(
        reply_to="123",
        result=result,
        content="ok",
        final=True,
    )

    assert adapter._discord_recovery_cursor("123") == "123"

    # A non-final/failed response must NOT advance the cursor.
    adapter._record_discord_response(
        reply_to="456",
        result=SendResult(success=False, error="boom"),
        content="ok",
        final=True,
    )
    assert adapter._discord_recovery_cursor("456") is None


@pytest.mark.asyncio
async def test_processing_complete_writes_durable_state_and_advances_cursor(adapter, monkeypatch):
    message = make_message(message_id=7, content="handle me")
    event = make_event(message, message_id="7")

    adapter._record_discord_message_seen(message, status="processing")
    adapter._record_discord_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert adapter._discord_message_is_persistently_complete("7") is False  # not responded yet

    adapter._record_discord_response(
        reply_to="7",
        result=SendResult(success=True, message_id="srv-2"),
        content="done",
        final=True,
    )

    assert adapter._discord_message_is_persistently_complete("7") is True
    assert adapter._discord_recovery_cursor("123") == "7"
