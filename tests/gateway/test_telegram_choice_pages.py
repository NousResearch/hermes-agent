"""Drive the real Telegram adapter/helper against a fake Bot API."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from plugins.platforms.telegram import adapter as module
from plugins.platforms.telegram.choice_picker import cancel_choice_pages, _expire
from tests.gateway.test_telegram_choice_picker import _query


def page(value="older"):
    return ChoicePage(
        "Files",
        [
            {
                "value": value,
                "label": "report.pdf · Writer · 06:41 · 210 KB",
                "full_width": True,
            }
        ],
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        module,
        "InlineKeyboardButton",
        lambda text, callback_data: SimpleNamespace(
            text=text, callback_data=callback_data
        ),
    )
    monkeypatch.setattr(
        module,
        "InlineKeyboardMarkup",
        lambda rows: SimpleNamespace(inline_keyboard=rows),
    )
    a = object.__new__(module.TelegramAdapter)
    a._bot = SimpleNamespace(
        edit_message_reply_markup=AsyncMock(), edit_message_text=AsyncMock()
    )
    a._choice_picker_state = {}
    a._reply_to_mode = "none"
    a._reply_to_message_id_for_send = lambda *args, **kwargs: 7
    a._thread_kwargs_for_send = lambda *args, **kwargs: {}
    a._link_preview_kwargs = lambda: {}
    a.format_message = lambda text: text
    a._is_callback_user_authorized = lambda *args, **kwargs: True
    a._send_message_with_thread_fallback = AsyncMock(
        return_value=SimpleNamespace(message_id=10)
    )
    yield a
    cancel_choice_pages(a)


async def send(adapter, callback, *, reusable=True, choices=None):
    result = await adapter.send_choice_picker(
        "chat-1",
        "Files",
        choices or page().choices,
        "session",
        callback,
        {"choice_pages": reusable, "requester_user_id": "user-1"},
    )
    assert result.success
    return (
        adapter._send_message_with_thread_fallback.await_args
        .kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .callback_data
    )


@pytest.mark.asyncio
async def test_next_back_same_message_and_old_revision_refused(adapter):
    callback = AsyncMock(side_effect=[page("newer"), page("older"), "Cancelled"])
    old = await send(adapter, callback)
    q = _query()
    await adapter._handle_choice_picker_callback(q, old, "chat-1")
    current = (
        q.edit_message_text.await_args
        .kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .callback_data
    )
    assert current != old
    await adapter._handle_choice_picker_callback(_query(), old, "chat-1")
    assert callback.await_count == 1
    await adapter._handle_choice_picker_callback(q, current, "chat-1")
    latest = (
        q.edit_message_text.await_args
        .kwargs["reply_markup"]
        .inline_keyboard[0][0]
        .callback_data
    )
    await adapter._handle_choice_picker_callback(q, latest, "chat-1")
    assert [call.args[1] for call in callback.await_args_list] == [
        "older",
        "newer",
        "older",
    ]
    assert q.edit_message_text.await_args.kwargs["reply_markup"] is None
    assert not adapter._choice_picker_state
    assert adapter._send_message_with_thread_fallback.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reusable", [False, True])
async def test_double_tap_claims_before_ack_and_callback(adapter, reusable):
    entered, release = asyncio.Event(), asyncio.Event()
    q = _query()

    async def callback(*args):
        q.answer.assert_awaited_once()
        entered.set()
        await release.wait()
        return "Sent"

    action = await send(adapter, callback, reusable=reusable)
    first = asyncio.create_task(
        adapter._handle_choice_picker_callback(q, action, "chat-1")
    )
    await entered.wait()
    duplicate = _query()
    await adapter._handle_choice_picker_callback(duplicate, action, "chat-1")
    assert not first.done()
    release.set()
    await first
    assert q.edit_message_text.await_count == 1
    duplicate.edit_message_text.assert_not_awaited()
    assert not adapter._choice_picker_state


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["user", "message", "chat", "thread"])
async def test_original_actor_message_and_topic_scope(adapter, mismatch):
    callback = AsyncMock(return_value=page())
    action = await send(adapter, callback)
    q = _query(
        user_id="other" if mismatch == "user" else "user-1",
        message_id=9 if mismatch == "message" else 10,
    )
    if mismatch == "chat":
        q.message.chat_id = "another-chat"
    if mismatch == "thread":
        q.message.message_thread_id = 44
    await adapter._handle_choice_picker_callback(q, action, "chat-1")
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_expiry_clears_controls_and_gives_feedback(adapter):
    callback = AsyncMock(return_value=page())
    action = await send(adapter, callback)
    adapter._choice_picker_state["chat-1"]["expires_at"] = time.monotonic() - 1
    q = _query()
    await adapter._handle_choice_picker_callback(q, action, "chat-1")
    callback.assert_not_awaited()
    adapter._bot.edit_message_text.assert_awaited_once()
    assert "expired" in q.answer.await_args.kwargs["text"]
    assert not adapter._choice_picker_state


@pytest.mark.asyncio
async def test_progress_runs_once_after_sending_feedback_and_keeps_page(adapter):
    q = _query()

    async def deliver():
        assert q.edit_message_text.await_args.kwargs["text"] == "Sending"
        assert q.edit_message_text.await_args.kwargs["reply_markup"] is None
        return page("sent")

    callback = AsyncMock(return_value=ChoiceProgress("Sending", deliver))
    action = await send(adapter, callback)
    await adapter._handle_choice_picker_callback(q, action, "chat-1")
    assert q.edit_message_text.await_count == 2
    assert adapter._choice_picker_state["chat-1"]["choices"][0]["value"] == "sent"


@pytest.mark.asyncio
async def test_replacement_while_callback_waits_is_not_destroyed(adapter):
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        entered.set()
        await release.wait()
        return page()

    action = await send(adapter, callback)
    q = _query()
    task = asyncio.create_task(
        adapter._handle_choice_picker_callback(q, action, "chat-1")
    )
    await entered.wait()
    adapter._send_message_with_thread_fallback.return_value = SimpleNamespace(
        message_id=11
    )
    await send(adapter, AsyncMock(return_value="Done"))
    release.set()
    await task
    assert adapter._choice_picker_state["chat-1"]["msg_id"] == 11
    q.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_name_and_action_limits_preserve_long_value(adapter):
    callback = AsyncMock(return_value="Done")
    value = "opaque-" + "x" * 900
    action = await send(
        adapter,
        callback,
        choices=[{"label": "資料" * 100, "value": value, "is_current": True}],
    )
    button = adapter._send_message_with_thread_fallback.await_args.kwargs[
        "reply_markup"
    ].inline_keyboard[0][0]
    assert len(button.text) <= 64 and len(action.encode()) <= 64
    await adapter._handle_choice_picker_callback(_query(), action, "chat-1")
    callback.assert_awaited_once_with("chat-1", value)


@pytest.mark.asyncio
async def test_expiry_while_work_waits_ignores_late_page(adapter):
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        entered.set()
        await release.wait()
        return page("late")

    action = await send(adapter, callback)
    state = adapter._choice_picker_state["chat-1"]
    q = _query()
    task = asyncio.create_task(
        adapter._handle_choice_picker_callback(q, action, "chat-1")
    )
    await entered.wait()
    await _expire(adapter, "chat-1", state)
    release.set()
    await task
    q.edit_message_text.assert_not_awaited()
    assert "expired" in adapter._bot.edit_message_text.await_args.kwargs["text"]
    assert not adapter._choice_picker_state


@pytest.mark.asyncio
async def test_task_cancellation_cleans_state_and_timer(adapter):
    entered = asyncio.Event()

    async def callback(*args):
        entered.set()
        await asyncio.Event().wait()

    action = await send(adapter, callback)
    state = adapter._choice_picker_state["chat-1"]
    handle = state["expiry_handle"]
    task = asyncio.create_task(
        adapter._handle_choice_picker_callback(_query(), action, "chat-1")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter._choice_picker_state and handle.cancelled()


@pytest.mark.asyncio
async def test_failed_progress_display_never_starts_work(adapter):
    complete = AsyncMock(return_value="Sent")
    action = await send(
        adapter, AsyncMock(return_value=ChoiceProgress("Sending", complete))
    )
    q = _query()
    q.edit_message_text.side_effect = RuntimeError("API failed")
    await adapter._handle_choice_picker_callback(q, action, "chat-1")
    complete.assert_not_awaited()
    assert not adapter._choice_picker_state


@pytest.mark.asyncio
async def test_idle_timer_expires_without_another_tap(adapter, monkeypatch):
    from plugins.platforms.telegram import choice_picker

    monkeypatch.setattr(choice_picker, "PAGE_TIMEOUT_SECONDS", 0.01)
    expired = asyncio.Event()
    adapter._bot.edit_message_text.side_effect = lambda **kwargs: expired.set()
    await send(adapter, AsyncMock(return_value=page()))
    await asyncio.wait_for(expired.wait(), timeout=2)
    assert not adapter._choice_picker_state


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", ["expire", "cancel", "replace"])
async def test_retired_page_cannot_start_work_after_delayed_ack(adapter, retire):
    entered, release = asyncio.Event(), asyncio.Event()
    callback = AsyncMock(return_value="Done")
    action = await send(adapter, callback)
    state = adapter._choice_picker_state["chat-1"]
    query = _query()

    async def delayed_ack(*args, **kwargs):
        entered.set()
        await release.wait()

    query.answer.side_effect = delayed_ack
    task = asyncio.create_task(
        adapter._handle_choice_picker_callback(query, action, "chat-1")
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if retire == "expire":
            await _expire(adapter, "chat-1", state)
        elif retire == "cancel":
            cancel_choice_pages(adapter)
        else:
            adapter._send_message_with_thread_fallback.return_value = SimpleNamespace(
                message_id=20
            )
            await send(adapter, AsyncMock(return_value="New page"))
        callback.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    callback.assert_not_awaited()
