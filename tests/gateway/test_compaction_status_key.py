"""Compaction status must own its bubble, and deletes must prune the cache.

Two defects motivated this, both found on 2026-08-16 after a compaction ran
with nothing shown to the user:

1. The compaction status rode on the generic "lifecycle" event_type, which is
   shared by all 32 ``_emit_status`` call sites. The gateway passes event_type
   straight through as the Telegram status_key, and ``send_or_update_status``
   edits ONE bubble per key -- so any other lifecycle status emitted during a
   compaction overwrote "Compacting context" in place. The Claude SDK lane hit
   this hardest because its PreCompact hook fires mid-turn while tool-progress
   statuses are streaming.

2. End-of-turn progress cleanup deletes status bubbles, but never pruned
   ``_status_message_ids``. The next status edited a dead id, costing two
   wasted round-trips (MarkdownV2 + plain-text retry) before the fail-open
   path sent fresh. 261 "Message to edit not found" errors were logged between
   2026-07-21 and 2026-08-16 from exactly this.

Deliberately NOT tested here: durability. Compaction status stays ephemeral
(swept by cleanup) to match Codex and native Hermes, which emit it the same
way. See ``_status_event_is_durable``.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _install_fake_telegram(monkeypatch):
    """Stub python-telegram-bot so TelegramAdapter can be imported."""
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = object
    fake_telegram.InlineKeyboardMarkup = object

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = type("NetworkError", (Exception,), {})
    fake_error.BadRequest = type("BadRequest", (Exception,), {})
    fake_error.TimedOut = type("TimedOut", (Exception,), {})
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup",
        CHANNEL="channel", PRIVATE="private",
    )
    fake_telegram.constants = fake_constants

    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.Application = object
    fake_ext.CommandHandler = object
    fake_ext.CallbackQueryHandler = object
    fake_ext.MessageHandler = object
    fake_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    fake_ext.filters = object

    fake_request = types.ModuleType("telegram.request")
    fake_request.HTTPXRequest = object

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


@pytest.fixture
def adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    a._bot.delete_message = AsyncMock()
    a.send = AsyncMock()
    a.edit_message = AsyncMock()
    return a


def _key():
    from agent.conversation_compression import COMPACTION_STATUS_KEY

    return COMPACTION_STATUS_KEY


class TestDedicatedKey:

    def test_compaction_does_not_share_the_lifecycle_key(self):
        """The whole point: a separate key so lifecycle traffic can't clobber it."""
        assert _key() != "lifecycle"

    @pytest.mark.asyncio
    async def test_a_lifecycle_status_cannot_overwrite_the_compaction_bubble(self, adapter):
        """The reported symptom: compaction notice replaced mid-compaction."""
        from agent.conversation_compression import COMPACTION_STATUS

        adapter.send.side_effect = [
            SendResult(success=True, message_id="100"),
            SendResult(success=True, message_id="200"),
        ]

        await adapter.send_or_update_status("chat-1", _key(), COMPACTION_STATUS)
        await adapter.send_or_update_status("chat-1", "lifecycle", "Running a tool...")

        assert adapter._status_message_ids[("chat-1", _key())] == "100"
        assert adapter._status_message_ids[("chat-1", "lifecycle")] == "200"
        # Two separate bubbles: the lifecycle status never edited the compaction one.
        adapter.edit_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_done_notice_edits_the_compacting_bubble_in_place(self, adapter):
        """Start and done share the key, so completion replaces "Compacting…"."""
        from agent.conversation_compression import (
            COMPACTION_DONE_STATUS,
            COMPACTION_STATUS,
        )

        adapter.send.return_value = SendResult(success=True, message_id="100")
        adapter.edit_message.return_value = SendResult(success=True, message_id="100")

        await adapter.send_or_update_status("chat-1", _key(), COMPACTION_STATUS)
        await adapter.send_or_update_status("chat-1", _key(), COMPACTION_DONE_STATUS)

        adapter.send.assert_awaited_once()
        adapter.edit_message.assert_awaited_once()
        assert adapter._status_message_ids[("chat-1", _key())] == "100"

    def test_the_done_notice_is_emitted_on_the_dedicated_key(self):
        """_emit_compaction_done must not fall back to a bare "compacted"."""
        from agent.conversation_compression import (
            COMPACTION_DONE_STATUS,
            _emit_compaction_done,
        )

        seen = []
        agent = SimpleNamespace(
            status_callback=lambda ev, msg: seen.append((ev, msg))
        )
        _emit_compaction_done(agent)

        assert seen == [(_key(), COMPACTION_DONE_STATUS)]


class TestDeletePrunesTheCache:
    """Defect 2 -- the 261 'Message to edit not found' errors."""

    @pytest.mark.asyncio
    async def test_deleting_a_status_message_prunes_its_cached_id(self, adapter):
        adapter.send.return_value = SendResult(success=True, message_id="100")
        await adapter.send_or_update_status("chat-1", _key(), "Compacting...")
        assert ("chat-1", _key()) in adapter._status_message_ids

        await adapter.delete_message("chat-1", "100")

        assert ("chat-1", _key()) not in adapter._status_message_ids

    @pytest.mark.asyncio
    async def test_the_next_status_sends_fresh_instead_of_editing_a_dead_id(self, adapter):
        """The actual payoff: no wasted edit round-trip after cleanup."""
        adapter.send.side_effect = [
            SendResult(success=True, message_id="100"),
            SendResult(success=True, message_id="101"),
        ]
        await adapter.send_or_update_status("chat-1", _key(), "Compacting...")
        await adapter.delete_message("chat-1", "100")

        await adapter.send_or_update_status("chat-1", _key(), "Compacting again...")

        adapter.edit_message.assert_not_awaited()
        assert adapter._status_message_ids[("chat-1", _key())] == "101"

    @pytest.mark.asyncio
    async def test_a_delete_only_prunes_the_matching_message(self, adapter):
        """Deleting one bubble must not blow away unrelated cached ids."""
        adapter.send.side_effect = [
            SendResult(success=True, message_id="100"),
            SendResult(success=True, message_id="200"),
        ]
        await adapter.send_or_update_status("chat-1", _key(), "Compacting...")
        await adapter.send_or_update_status("chat-1", "lifecycle", "Working...")

        await adapter.delete_message("chat-1", "100")

        assert ("chat-1", _key()) not in adapter._status_message_ids
        assert adapter._status_message_ids[("chat-1", "lifecycle")] == "200"

    @pytest.mark.asyncio
    async def test_a_delete_in_another_chat_does_not_prune_this_one(self, adapter):
        """Message ids are only unique per chat, so the chat must be matched."""
        adapter.send.return_value = SendResult(success=True, message_id="100")
        await adapter.send_or_update_status("chat-1", _key(), "Compacting...")

        await adapter.delete_message("chat-2", "100")

        assert adapter._status_message_ids[("chat-1", _key())] == "100"

    @pytest.mark.asyncio
    async def test_a_failed_delete_keeps_the_cached_id(self, adapter):
        """If the message may still exist, editing it next time is still right."""
        adapter.send.return_value = SendResult(success=True, message_id="100")
        await adapter.send_or_update_status("chat-1", _key(), "Compacting...")

        adapter._bot.delete_message.side_effect = RuntimeError("network")
        ok = await adapter.delete_message("chat-1", "100")

        assert ok is False
        assert adapter._status_message_ids[("chat-1", _key())] == "100"
