"""Tests for TelegramAdapter.send_or_update_status (issue #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message has a permanent edit failure.
  4. Keep the cached bubble on transient edit failures, especially flood-control.
  5. Keep distinct keys independent (no cross-talk).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner, _send_or_update_status_coro


def _install_fake_telegram(monkeypatch):
    """Stub the python-telegram-bot package so TelegramAdapter can be imported."""
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
    # Patch send / edit_message so tests can drive them directly.
    a.send = AsyncMock()
    a.edit_message = AsyncMock()
    return a


@pytest.fixture
def real_edit_adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    a._bot = MagicMock()
    a._bot.edit_message_text = AsyncMock()
    a.send = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_first_call_sends_and_caches_message_id(adapter):
    """First call for a (chat, key) pair must send and remember the id."""
    adapter.send.return_value = SendResult(success=True, message_id="100")

    result = await adapter.send_or_update_status("chat-1", "lifecycle", "starting")

    assert result.success is True
    assert result.message_id == "100"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"


@pytest.mark.asyncio
async def test_second_call_edits_in_place(adapter):
    """Same (chat, key) on the second call must edit, not send."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    await adapter.send_or_update_status("chat-1", "lifecycle", "step 1")
    await adapter.send_or_update_status("chat-1", "lifecycle", "step 2")

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    # Edit was directed at the cached message id.
    args, kwargs = adapter.edit_message.call_args
    assert args[0] == "chat-1"
    assert args[1] == "100"
    assert args[2] == "step 2"


@pytest.mark.asyncio
async def test_edit_failure_falls_back_to_fresh_send(adapter):
    """When edit_message fails the cache is cleared and a new send happens."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]
    adapter.edit_message.return_value = SendResult(
        success=False, error="Bad Request: message to edit not found",
    )

    await adapter.send_or_update_status("chat-1", "lifecycle", "step 1")
    result = await adapter.send_or_update_status("chat-1", "lifecycle", "step 2")

    assert result.success is True
    assert result.message_id == "200"
    assert adapter.send.await_count == 2
    assert adapter.edit_message.await_count == 1
    # Cache now points at the fresh message id.
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "200"


@pytest.mark.asyncio
async def test_edit_flood_control_keeps_cached_bubble_without_fresh_send(adapter):
    """Transient edit flood-control must not append duplicate status bubbles."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="flood_control:38",
    )

    await adapter.send_or_update_status("chat-1", "compacting", "step 1")
    result = await adapter.send_or_update_status("chat-1", "compacting", "step 2")

    assert result.success is False
    assert result.error == "flood_control:38"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    assert adapter._status_message_ids[("chat-1", "compacting")] == "100"


@pytest.mark.asyncio
async def test_retry_failure_after_short_flood_wait_keeps_cached_bubble(
    real_edit_adapter,
    monkeypatch,
):
    """A transient failure after an inline RetryAfter retry must not send fresh."""

    class _RetryAfter(Exception):
        retry_after = 0.01

    real_edit_adapter.send.return_value = SendResult(success=True, message_id="100")
    real_edit_adapter._bot.edit_message_text.side_effect = [
        _RetryAfter("retry after 0.01"),
        _RetryAfter("retry after 0.01"),
        RuntimeError("NetworkError: temporarily unavailable"),
    ]
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    await real_edit_adapter.send_or_update_status(
        "chat-1",
        "compacting",
        "step 1",
    )
    result = await real_edit_adapter.send_or_update_status(
        "chat-1",
        "compacting",
        "step 2",
    )

    assert result.success is False
    assert result.retryable is True
    real_edit_adapter.send.assert_awaited_once()
    assert real_edit_adapter._status_message_ids[("chat-1", "compacting")] == "100"


@pytest.mark.asyncio
async def test_status_updates_for_same_key_are_ordered(adapter):
    """A slow older edit must not finish after and overwrite a newer final status."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    edit_started = asyncio.Event()
    release_edit = asyncio.Event()
    edited_contents = []

    async def _edit(_chat_id, _message_id, content, **_kwargs):
        edited_contents.append(content)
        if content == "heartbeat":
            edit_started.set()
            await release_edit.wait()
        return SendResult(success=True, message_id="100")

    adapter.edit_message.side_effect = _edit

    await adapter.send_or_update_status("chat-1", "compacting", "start")
    heartbeat = asyncio.create_task(
        adapter.send_or_update_status("chat-1", "compacting", "heartbeat")
    )
    await edit_started.wait()
    final = asyncio.create_task(
        adapter.send_or_update_status("chat-1", "compacting", "done")
    )

    release_edit.set()
    await heartbeat
    await final

    assert edited_contents == ["heartbeat", "done"]
    assert adapter._status_message_ids[("chat-1", "compacting")] == "100"


@pytest.mark.asyncio
async def test_older_emission_sequence_started_late_cannot_overwrite_done(adapter):
    """An older heartbeat coroutine that starts after done must be dropped."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    await adapter.send_or_update_status(
        "chat-1", "compacting", "start", sequence=1,
    )
    await adapter.send_or_update_status(
        "chat-1", "compacting", "done", sequence=3,
    )
    stale = await adapter.send_or_update_status(
        "chat-1", "compacting", "heartbeat", sequence=2,
    )

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    assert adapter.edit_message.call_args.args[2] == "done"
    assert stale.raw_response == {"skipped_stale_status_update": True}


@pytest.mark.asyncio
async def test_gateway_sequences_continue_across_status_lifecycles(adapter):
    """A later lifecycle for the same chat/key must not restart below adapter high-water."""
    runner = object.__new__(GatewayRunner)
    runner._status_update_sequences = {}
    runner._status_update_sequence_lock = threading.Lock()
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    # First compaction lifecycle reaches a high sequence.
    for content in ("start-1", "heartbeat-1", "done-1"):
        await adapter.send_or_update_status(
            "chat-1",
            "compacting",
            content,
            sequence=runner._next_status_update_sequence("chat-1", "compacting"),
        )

    # Second lifecycle starts later for the same chat/key. Its sequence must
    # continue above the first lifecycle, not reset to 1 and get skipped.
    second_start = await adapter.send_or_update_status(
        "chat-1",
        "compacting",
        "start-2",
        sequence=runner._next_status_update_sequence("chat-1", "compacting"),
    )

    assert second_start.raw_response != {"skipped_stale_status_update": True}
    assert adapter.send.await_count == 1
    assert adapter.edit_message.await_count == 3
    assert adapter.edit_message.call_args.args[2] == "start-2"


@pytest.mark.asyncio
async def test_status_coro_forwards_emission_sequence():
    """Gateway-assigned emission order must reach the status adapter."""
    seen = {}

    class _Adapter:
        async def send_or_update_status(
            self,
            chat_id,
            status_key,
            content,
            *,
            metadata=None,
            sequence=None,
        ):
            seen.update(
                chat_id=chat_id,
                status_key=status_key,
                content=content,
                metadata=metadata,
                sequence=sequence,
            )
            return SendResult(success=True, message_id="100")

    await _send_or_update_status_coro(
        _Adapter(),
        "chat-1",
        "compacting",
        "done",
        {"thread_id": "7"},
        sequence=42,
    )

    assert seen == {
        "chat_id": "chat-1",
        "status_key": "compacting",
        "content": "done",
        "metadata": {"thread_id": "7"},
        "sequence": 42,
    }


@pytest.mark.asyncio
async def test_distinct_status_keys_do_not_collide(adapter):
    """A different status_key gets its own message; the original isn't touched."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status("chat-1", "lifecycle", "ctx pressure")
    await adapter.send_or_update_status("chat-1", "model-switch", "switched to opus")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[("chat-1", "model-switch")] == "200"


@pytest.mark.asyncio
async def test_distinct_chat_ids_do_not_collide(adapter):
    """Same status_key in different chats must not edit each other's messages."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status("chat-1", "lifecycle", "first")
    await adapter.send_or_update_status("chat-2", "lifecycle", "second")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[("chat-2", "lifecycle")] == "200"
