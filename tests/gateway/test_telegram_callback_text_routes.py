"""Tests for config-driven Telegram callback→text routes (extra.callback_text_routes)."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter

ROUTES = {
    "log:breakfast:default": "Yes, had the usual ✅",
    "log:breakfast:custom": "Had something different ✏️",
}


def _make_adapter(routes=ROUTES):
    extra = {"callback_text_routes": routes} if routes is not None else {}
    config = PlatformConfig(enabled=True, token="test-token", extra=extra)
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _make_callback(data: str):
    chat = SimpleNamespace(
        id=12345,
        type="private",
        title=None,
        full_name="Alice",
        is_forum=False,
    )
    prompt_message = SimpleNamespace(
        chat=chat,
        chat_id=12345,
        text="Did you have your usual breakfast?",
        message_id=777,
        message_thread_id=None,
        is_topic_message=False,
    )
    from_user = SimpleNamespace(id="49334209", first_name="Alice", full_name="Alice")
    query = AsyncMock()
    query.data = data
    query.message = prompt_message
    query.from_user = from_user
    return SimpleNamespace(callback_query=query), query


@pytest.mark.asyncio
async def test_routed_callback_injects_mapped_text_with_full_context(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter()
    update, query = _make_callback("log:breakfast:default")

    await adapter._handle_callback_query(update, None)

    query.answer.assert_awaited_once()
    query.edit_message_text.assert_awaited_once()
    handle_message = cast(AsyncMock, adapter.handle_message)
    handle_message.assert_awaited_once()
    event = handle_message.call_args.args[0]
    assert event.text == "Yes, had the usual ✅"
    assert event.source.chat_id == "12345"
    assert event.source.user_id == "49334209"


@pytest.mark.asyncio
async def test_routed_callback_strips_buttons_and_shows_choice(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter()
    update, query = _make_callback("log:breakfast:custom")

    await adapter._handle_callback_query(update, None)

    kwargs = query.edit_message_text.call_args.kwargs
    assert kwargs["reply_markup"] is None
    assert "Alice" in kwargs["text"]
    assert "Had something different" in kwargs["text"]


@pytest.mark.asyncio
async def test_unknown_callback_data_is_not_routed(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter()
    update, query = _make_callback("log:unknown:choice")

    await adapter._handle_callback_query(update, None)

    handle_message = cast(AsyncMock, adapter.handle_message)
    handle_message.assert_not_awaited()
    query.answer.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_user_is_rejected(monkeypatch):
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=False)
    update, query = _make_callback("log:breakfast:default")

    await adapter._handle_callback_query(update, None)

    handle_message = cast(AsyncMock, adapter.handle_message)
    handle_message.assert_not_awaited()
    answer_kwargs = query.answer.call_args.kwargs
    assert "not authorized" in answer_kwargs.get("text", "")


@pytest.mark.asyncio
async def test_edit_failure_still_routes_text(monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_adapter()
    update, query = _make_callback("log:breakfast:default")
    query.edit_message_text.side_effect = RuntimeError("message too old")

    await adapter._handle_callback_query(update, None)

    handle_message = cast(AsyncMock, adapter.handle_message)
    handle_message.assert_awaited_once()
    event = handle_message.call_args.args[0]
    assert event.text == "Yes, had the usual ✅"


def test_invalid_routes_config_is_ignored():
    adapter = _make_adapter(routes="not-a-mapping")
    assert adapter._callback_text_routes == {}


def test_oversized_and_empty_entries_are_dropped():
    adapter = _make_adapter(
        routes={
            "k" * 65: "over the 64-byte callback_data limit",
            "log:empty": "   ",
            "log:ok": "fine",
            42: "non-string key",
        }
    )
    assert adapter._callback_text_routes == {"log:ok": "fine"}
