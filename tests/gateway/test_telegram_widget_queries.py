"""Widget interactions stay with the authenticated conversation that created them."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.telegram.telegram_widget_queries import handle_widget_query
from plugins.platforms.telegram.telegram_widget_store import WidgetStore


def host(tmp_path):
    adapter = SimpleNamespace(
        config=SimpleNamespace(token="123:test"), _update_receipt_dir=tmp_path,
        _accept_update=lambda: None, _is_callback_user_authorized=lambda *a, **k: True,
        _session_store=SimpleNamespace(peek_session_id=lambda key: "session-1"),
        answer_widget_query=AsyncMock(return_value=True), handle_message=AsyncMock())
    record = {"chat_id": "42", "user_id": "42", "thread_id": "7", "session_key": "telegram:42:7",
              "session_id": "session-1", "profile": "default", "message_id": "99", "widget_id": "shared-1",
              "allow_queries": True, "actions": {"details": "Explain the current report"},
              "public_data": '{"progress":50}', "chat_data": '{"selected":1}'}
    WidgetStore(tmp_path, "123").save_message(record)
    return adapter, record


def update(data="details", user=42, message=99, **extra):
    return SimpleNamespace(update_id=11, api_kwargs={"widget_query": {
        "id": "query-1", "from": {"id": user, "first_name": "Owner"}, "chat": {"id": 42, "type": "private"},
        "message_id": message, "type": {"type": "data"}, "data": data, **extra}})


@pytest.mark.asyncio
async def test_widget_action_acknowledged_before_dispatch_and_never_gateway_command(tmp_path):
    adapter, _ = host(tmp_path)
    async def receive(event):
        adapter.answer_widget_query.assert_awaited_once()
        assert event.text == "Telegram widget action: Explain the current report"
        assert not event.allow_gateway_control and not event.internal
        assert event.source.thread_id == "7"
        assert event.reply_anchor_override == "99"
        assert event.user_id == "42"
    adapter.handle_message.side_effect = receive
    await handle_widget_query(adapter, update())
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"user": 43}, {"message": 100}, {"data": "/approve"}, {"widget_id": "foreign"}])
async def test_unknown_or_foreign_widget_request_never_dispatches(tmp_path, change):
    adapter, _ = host(tmp_path)
    await handle_widget_query(adapter, update(**change))
    adapter.handle_message.assert_not_awaited()
    assert '"ok": false' in adapter.answer_widget_query.call_args.args[1]


@pytest.mark.asyncio
async def test_widget_from_replaced_session_cannot_wake_successor(tmp_path):
    adapter, _ = host(tmp_path)
    adapter._session_store.peek_session_id = lambda key: "session-2"
    await handle_widget_query(adapter, update())
    adapter.handle_message.assert_not_awaited()
    assert "conversation has ended" in adapter.answer_widget_query.call_args.args[1]


@pytest.mark.asyncio
async def test_refresh_reads_persisted_state_without_llm_turn(tmp_path):
    adapter, _ = host(tmp_path)
    await handle_widget_query(adapter, update(data="refresh"))
    adapter.handle_message.assert_not_awaited()
    assert "progress" in adapter.answer_widget_query.call_args.args[1]


@pytest.mark.asyncio
async def test_repeated_widget_action_does_not_start_background_request_loop(tmp_path):
    adapter, _ = host(tmp_path)
    await handle_widget_query(adapter, update())
    await handle_widget_query(adapter, update())
    adapter.handle_message.assert_awaited_once()
    assert "Please wait" in adapter.answer_widget_query.call_args.args[1]


def test_widget_state_scopes_bot_chat_topic_and_conversation(tmp_path):
    _, record = host(tmp_path)
    assert WidgetStore(tmp_path, "456").message("42", "99") is None
    loaded = WidgetStore(tmp_path, "123").message("42", "99")
    assert WidgetStore.require_owner(loaded, record)["actions"] == record["actions"]
    for field in ("chat_id", "user_id", "thread_id", "session_id", "session_key"):
        with pytest.raises(ValueError):
            WidgetStore.require_owner(loaded, {**record, field: "other"})
