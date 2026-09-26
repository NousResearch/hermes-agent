"""Tests for the Telegram Continue button after an iteration-limit turn."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run_turn import GatewayTurnMixin
from plugins.platforms.telegram.adapter import TelegramAdapter


class _SessionStore:
    def __init__(self, value=None):
        self.values = {}
        if value is not None:
            self.values["telegram_continue_token"] = value

    def get_session_metadata(self, session_key, key, default=None):
        return self.values.get(key, default)

    def set_session_metadata(self, session_key, key, value):
        self.values[key] = value
        return True


def _make_adapter(*, token="abc"):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = SimpleNamespace()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=900))
    adapter._session_store = _SessionStore(token)
    return adapter


def _callback_query(*, token="abc"):
    message = SimpleNamespace(
        chat=SimpleNamespace(id=-1004494376259, type="supergroup", title="Альтрон", is_forum=True),
        chat_id=-1004494376259,
        message_id=700,
        message_thread_id=836,
        is_topic_message=True,
    )
    return SimpleNamespace(
        data=f"hermes:continue:{token}",
        message=message,
        from_user=SimpleNamespace(id=123, first_name="Vlad", full_name="Vlad", is_bot=False),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )


def test_iteration_limit_arms_persisted_token_only_for_telegram():
    store = _SessionStore()
    event = MessageEvent(text="task")
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    runner = SimpleNamespace(
        session_store=store,
        _adapter_for_source=lambda _source: SimpleNamespace(
            _streaming_tts_turn_completed=lambda *_args: False,
        ),
        _should_send_voice_reply=lambda *_args, **_kwargs: False,
    )

    response = asyncio.run(GatewayTurnMixin._hmwa_deliver_turn_response(
        runner, event, source, SimpleNamespace(session_id="sid"), "session", 1,
        {"turn_exit_reason": "max_iterations_reached(2/2)"}, [], "Итог", None, False,
    ))

    assert response == "Итог"
    token = event.metadata["telegram_continue_token"]
    assert token
    assert store.values["telegram_continue_token"] == token

    regular_event = MessageEvent(text="task")
    asyncio.run(GatewayTurnMixin._hmwa_deliver_turn_response(
        runner, regular_event, source, SimpleNamespace(session_id="sid"), "session", 1,
        {"turn_exit_reason": "text_response(stop)"}, [], "Готово", None, False,
    ))
    assert "telegram_continue_token" not in regular_event.metadata


def test_iteration_limit_metadata_renders_continue_button(monkeypatch):
    adapter = _make_adapter(token=None)
    captured = {}

    async def send_message(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(message_id=900)

    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda text, callback_data: {"text": text, "callback_data": callback_data},
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
        lambda rows: rows,
    )

    result = asyncio.run(adapter.send(
        "-1004494376259",
        "Итог после лимита",
        metadata={"thread_id": "836", "telegram_continue_token": "abc123"},
    ))

    assert result.success is True
    assert captured["reply_markup"] == [[
        {"text": "▶️ Продолжить", "callback_data": "hermes:continue:abc123"},
    ]]
    assert captured["message_thread_id"] == 836


def test_continue_callback_resumes_same_topic_and_consumes_token(monkeypatch):
    adapter = _make_adapter(token="abc")
    adapter._message_handler = AsyncMock(return_value="Продолжение выполнено")
    query = _callback_query(token="abc")
    update = SimpleNamespace(callback_query=query)

    monkeypatch.setattr(os, "environ", {**os.environ, "TELEGRAM_ALLOWED_USERS": "*"})

    asyncio.run(adapter._handle_callback_query(update, None))

    query.answer.assert_awaited_once()
    assert "Продолжаю" in query.answer.call_args.kwargs["text"]
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
    adapter._message_handler.assert_awaited_once()
    event = adapter._message_handler.await_args.args[0]
    assert "iteration limit" in event.text.lower()
    assert event.source.chat_id == "-1004494376259"
    assert event.source.thread_id == "836"
    assert adapter._session_store.values["telegram_continue_token"] is None

    # A duplicate delivery of the same callback is acknowledged but cannot start a second turn.
    asyncio.run(adapter._handle_callback_query(update, None))
    assert adapter._message_handler.await_count == 1
    assert "уже" in query.answer.await_args_list[-1].kwargs["text"].lower()


def test_continue_callback_rejects_stale_token_without_running(monkeypatch):
    adapter = _make_adapter(token="fresh-token")
    adapter._message_handler = AsyncMock(return_value="must not run")
    query = _callback_query(token="old-token")
    update = SimpleNamespace(callback_query=query)

    monkeypatch.setattr(os, "environ", {**os.environ, "TELEGRAM_ALLOWED_USERS": "*"})

    asyncio.run(adapter._handle_callback_query(update, None))

    adapter._message_handler.assert_not_awaited()
    query.edit_message_reply_markup.assert_not_awaited()
    assert "устар" in query.answer.call_args.kwargs["text"].lower()
