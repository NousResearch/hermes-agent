"""Tests for Telegram status updates (issues #30045 and #92210).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
  5. Keep gateway keys stable within a turn and distinct across turns/topics.
  6. Serialize overlapping updates for the same turn.
  7. Bound cached turn-scoped status-message IDs.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import TurnRunner
from gateway.turn_context import TurnContext


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
async def test_status_message_id_cache_is_bounded(adapter):
    adapter._STATUS_MESSAGE_IDS_MAX = 4
    adapter.send.side_effect = [
        SendResult(success=True, message_id=str(index)) for index in range(5)
    ]

    for index in range(5):
        await adapter.send_or_update_status(
            "chat-1", f"turn-{index}:lifecycle", f"status {index}"
        )

    assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
    assert adapter._status_message_ids[("chat-1", "turn-4:lifecycle")] == "4"


@pytest.mark.asyncio
async def test_concurrent_same_turn_statuses_send_once_then_edit(adapter):
    """Overlapping callbacks for one turn must not create duplicate bubbles."""
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def delayed_send(*args, **kwargs):
        send_started.set()
        await release_send.wait()
        return SendResult(success=True, message_id="100")

    adapter.send.side_effect = delayed_send
    first = asyncio.create_task(
        adapter.send_or_update_status("chat-1", "turn-1:lifecycle", "first")
    )
    await send_started.wait()
    second = asyncio.create_task(
        adapter.send_or_update_status("chat-1", "turn-1:lifecycle", "second")
    )
    await asyncio.sleep(0)

    assert adapter.send.await_count == 1
    release_send.set()
    await asyncio.gather(first, second)

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()


def _status_keys_for_turn(
    monkeypatch,
    *,
    session_key: str,
    run_generation: int,
) -> list[str]:
    captured: list[str] = []

    monkeypatch.setattr(
        "gateway.run._prepare_gateway_status_message",
        lambda platform, event_type, message: message,
    )
    monkeypatch.setattr(
        "gateway.run._send_or_update_status_coro",
        lambda adapter, chat_id, status_key, content, metadata: captured.append(
            status_key
        ),
    )
    monkeypatch.setattr(
        "gateway.run.safe_schedule_threadsafe",
        lambda awaitable, loop, **kwargs: MagicMock(),
    )

    ctx = TurnContext(
        source=SimpleNamespace(platform=Platform.TELEGRAM),
        _run_still_current=lambda: True,
        session_key=session_key,
        run_generation=run_generation,
        _status_adapter=object(),
        _status_chat_id="chat-1",
        _loop_for_step=object(),
    )
    runner = TurnRunner(MagicMock(), ctx)
    runner._status_callback_sync("lifecycle", "first status")
    runner._status_callback_sync("lifecycle", "updated status")
    return captured


def test_lifecycle_status_key_is_stable_within_turn_and_unique_across_turns(
    monkeypatch,
) -> None:
    first_turn = _status_keys_for_turn(
        monkeypatch,
        session_key="agent:main:telegram:dm:chat-1:topic-a",
        run_generation=1,
    )
    next_turn = _status_keys_for_turn(
        monkeypatch,
        session_key="agent:main:telegram:dm:chat-1:topic-a",
        run_generation=2,
    )
    other_topic = _status_keys_for_turn(
        monkeypatch,
        session_key="agent:main:telegram:dm:chat-1:topic-b",
        run_generation=1,
    )

    assert first_turn[0] == first_turn[1]
    assert first_turn[0] != next_turn[0]
    assert first_turn[0] != other_topic[0]
