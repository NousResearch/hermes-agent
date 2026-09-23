"""Tests for TelegramAdapter.send_or_update_status (issue #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
"""

from __future__ import annotations

import asyncio
import gc
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


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
    fake_ext.InlineQueryHandler = object
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
async def test_gateway_status_bubbles_are_owned_by_one_turn(adapter, monkeypatch):
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    pending = []
    monkeypatch.setattr(TurnRunner, "_schedule", lambda self, coro, message: pending.append(coro))
    adapter.send.side_effect = [
        SendResult(success=True, message_id=str(index)) for index in range(10)
    ]
    adapter.edit_message.return_value = SendResult(success=True)

    async def emit(session_key, generation, event_type="lifecycle"):
        ctx = TurnContext(
            source=SimpleNamespace(platform=Platform.TELEGRAM),
            session_key=session_key, run_generation=generation,
            _run_still_current=lambda: True,
            _status_adapter=adapter, _status_chat_id="chat-1",
            user_config={},
        )
        TurnRunner(MagicMock(), ctx)._status_callback_sync(event_type, "Recalling memories")
        assert len(pending) == 1
        await pending.pop()

    await emit("topic-a", 1)
    await emit("topic-a", 1)
    adapter.send.assert_awaited_once()
    assert adapter.edit_message.await_args.args[1] == "0"
    await emit("topic-a", 2)
    assert adapter.send.await_count == 2
    await emit("topic-b", 1)
    assert adapter.send.await_count == 3
    await emit("topic-a", 1, "model-switch")
    assert adapter.send.await_count == 4
    # Incomplete identities retain the legacy event-type contract.
    await emit(None, 1)
    await emit("topic-a", None)
    assert adapter.send.await_count == 5
    assert adapter.edit_message.await_args.args[1] == "4"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["overlap", "bounds"])
async def test_status_bookkeeping_is_serialized_and_bounded(adapter, mode):
    if mode == "bounds":
        adapter._STATUS_MESSAGE_IDS_MAX = 4
        adapter.send.side_effect = [
            SendResult(success=True, message_id=str(index)) for index in range(8)
        ]
        for index in range(8):
            await adapter.send_or_update_status("chat-1", f"turn-{index}", "status")
        assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
        assert adapter._status_message_ids[("chat-1", "turn-7")] == "7"
        gc.collect()
        assert not adapter._status_locks
        return

    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(*args, **kwargs):
        started.set()
        await release.wait()
        return SendResult(success=True, message_id="100")

    adapter.send.side_effect = delayed_send
    adapter.edit_message.return_value = SendResult(success=True)
    tasks = [asyncio.create_task(adapter.send_or_update_status("chat-1", "turn-1", "first"))]
    try:
        await asyncio.wait_for(started.wait(), 5)
        tasks.append(asyncio.create_task(adapter.send_or_update_status("chat-1", "turn-1", "second")))
        await asyncio.sleep(0)
        gc.collect()
        tasks.append(asyncio.create_task(adapter.send_or_update_status("chat-1", "turn-1", "third")))
        await asyncio.sleep(0)
        assert adapter.send.await_count == 1
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert adapter.edit_message.await_count == 2
    assert not adapter._status_locks
    # A failed edit must drop the old id and recover with a new send.
    adapter.edit_message.return_value = SendResult(success=False)
    await adapter.send_or_update_status("chat-1", "turn-1", "recover")
    assert adapter.send.await_count == 2


