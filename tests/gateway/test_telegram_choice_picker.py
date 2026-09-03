"""Telegram finite-choice pickers stay bound to one user and message."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(state):
    adapter = object.__new__(TelegramAdapter)
    adapter._choice_picker_state = {"chat-1": state}
    adapter._is_callback_user_authorized = lambda *_args, **_kwargs: True
    adapter.format_message = lambda text: text
    return adapter


def _query(*, message_id=10, user_id="user-1"):
    return SimpleNamespace(
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        from_user=SimpleNamespace(id=user_id, first_name="Owner"),
        message=SimpleNamespace(
            chat=SimpleNamespace(type="private"),
            chat_id="chat-1",
            message_id=message_id,
            message_thread_id=None,
        ),
    )


def _state(callback):
    return {
        "choices": [{"value": "12", "label": "Product launch"}],
        "expires_at": time.monotonic() + 120,
        "msg_id": 10,
        "on_choice_selected": callback,
        "requester_user_id": "user-1",
        "session_key": "session-1",
    }


@pytest.mark.asyncio
async def test_full_width_choices_render_one_room_per_row(monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_adapter

    monkeypatch.setattr(
        telegram_adapter,
        "InlineKeyboardButton",
        lambda label, callback_data: {"label": label, "callback_data": callback_data},
    )
    monkeypatch.setattr(
        telegram_adapter,
        "InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows),
    )
    adapter = object.__new__(TelegramAdapter)
    adapter._bot = object()
    adapter._choice_picker_state = {}
    adapter._link_preview_kwargs = lambda: {}
    adapter._reply_to_message_id_for_send = lambda *_args, **_kwargs: None
    adapter._reply_to_mode = "none"
    adapter._send_message_with_thread_fallback = AsyncMock(
        return_value=SimpleNamespace(message_id=10)
    )
    adapter._thread_kwargs_for_send = lambda *_args, **_kwargs: {}
    adapter.format_message = lambda text: text

    result = await adapter.send_choice_picker(
        chat_id="chat-1",
        title="Group Chats",
        choices=[
            {"value": "1", "label": "Release", "full_width": True},
            {"value": "2", "label": "Research", "full_width": True},
        ],
        session_key="session-1",
        on_choice_selected=AsyncMock(),
        metadata={"requester_user_id": "user-1"},
    )

    assert result.success is True
    markup = adapter._send_message_with_thread_fallback.await_args.kwargs[
        "reply_markup"
    ]
    assert [len(row) for row in markup.inline_keyboard] == [1, 1]
    assert adapter._choice_picker_state["chat-1"]["requester_user_id"] == "user-1"


@pytest.mark.asyncio
async def test_stale_keyboard_cannot_select_a_newer_picker():
    callback = AsyncMock(return_value="opened")
    adapter = _adapter(_state(callback))
    query = _query(message_id=9)

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat-1")

    callback.assert_not_awaited()
    query.edit_message_text.assert_not_awaited()
    assert "chat-1" in adapter._choice_picker_state
    assert "expired" in query.answer.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_expired_picker_is_removed_without_running_callback():
    callback = AsyncMock(return_value="opened")
    state = _state(callback)
    state["expires_at"] = time.monotonic() - 1
    adapter = _adapter(state)
    query = _query()

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat-1")

    callback.assert_not_awaited()
    assert "chat-1" not in adapter._choice_picker_state


@pytest.mark.asyncio
async def test_picker_rejects_another_authorized_user():
    callback = AsyncMock(return_value="opened")
    adapter = _adapter(_state(callback))
    query = _query(user_id="user-2")

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat-1")

    callback.assert_not_awaited()
    assert "another user" in query.answer.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_current_owner_selection_edits_the_bound_message():
    callback = AsyncMock(return_value="💬 **Product launch**")
    adapter = _adapter(_state(callback))
    query = _query()

    await adapter._handle_choice_picker_callback(query, "cp:0", "chat-1")

    callback.assert_awaited_once_with("chat-1", "12")
    assert query.edit_message_text.await_args.kwargs["text"] == "💬 **Product launch**"
    assert "chat-1" not in adapter._choice_picker_state
