"""Native Telegram cards must never resolve a successor's foreground consent."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['stale', 'valid', 'unauthorized', 'chat', 'thread', 'legacy', 'metadata_only'])
async def test_native_card_exact_request(monkeypatch, mode):
    # Optional Telegram dependency is stubbed by conftest in this environment.
    import plugins.platforms.telegram.adapter as telegram
    monkeypatch.setattr(telegram, 'InlineKeyboardButton', lambda label, callback_data: SimpleNamespace(callback_data=callback_data))
    monkeypatch.setattr(telegram, 'InlineKeyboardMarkup', lambda rows: SimpleNamespace(inline_keyboard=rows))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='test-token'))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=42)
    monkeypatch.setattr(adapter, '_is_callback_user_authorized', lambda *a, **kw: mode != 'unauthorized')
    key = 'native-request-binding'
    old = _ApprovalEntry({'request_id': 'old'})
    new = _ApprovalEntry({'request_id': 'new'})
    approval._gateway_queues[key] = [old]
    try:
        if mode == 'legacy':
            adapter._approval_state[1] = key
            callback = 'ea:once:1'
        else:
            await adapter.send_exec_approval(chat_id='123', command='fixture', session_key=key,
                metadata={'thread_id': '9', 'approval_request_id': 'old'},
                request_id=None if mode == 'metadata_only' else 'old')
            markup = adapter._bot.send_message.call_args.kwargs['reply_markup']
            callback = markup.inline_keyboard[0][0].callback_data
        if mode == 'stale':
            assert approval.resolve_gateway_approval(key, 'deny', request_id='old') == 1
            approval._gateway_queues[key] = [new]
        query = SimpleNamespace(data=callback, from_user=SimpleNamespace(id=123, first_name='owner'),
            message=SimpleNamespace(chat_id=456 if mode == 'chat' else 123,
                chat=SimpleNamespace(type='supergroup'), message_thread_id=10 if mode == 'thread' else 9),
            answer=AsyncMock(), edit_message_text=AsyncMock())
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        target = new if mode == 'stale' else old
        assert target.event.is_set() is (mode == 'valid'), (callback, adapter._approval_state, query.answer.call_args_list)
        if mode == 'valid':
            assert target.result == 'once'
    finally:
        approval.clear_session(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_runner_transport_binds_callback_to_request_not_metadata(monkeypatch, stale):
    import asyncio
    import plugins.platforms.telegram.adapter as telegram
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(telegram, "InlineKeyboardButton", lambda label, callback_data: SimpleNamespace(callback_data=callback_data))
    monkeypatch.setattr(telegram, "InlineKeyboardMarkup", lambda rows: SimpleNamespace(inline_keyboard=rows))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=42)
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    route = "transport-native-binding"
    metadata = {"thread_id": "9", "approval_request_id": "metadata-spoof"}
    ctx = TurnContext(session_key=route, session_id="session", source=None,
        _loop_for_step=asyncio.get_running_loop(), _status_adapter=adapter,
        _status_chat_id="123", _status_thread_metadata=metadata, _run_still_current=lambda: True)
    turn = TurnRunner(SimpleNamespace(), ctx)
    old = _ApprovalEntry({"request_id": "old"})
    successor = _ApprovalEntry({"request_id": "new"})
    approval._gateway_queues[route] = [old]
    try:
        await asyncio.to_thread(turn._approval_notify_sync, {"request_id": "old", "command": "fixture"})
        assert metadata == {"thread_id": "9", "approval_request_id": "metadata-spoof"}
        markup = adapter._bot.send_message.call_args.kwargs["reply_markup"]
        callback = markup.inline_keyboard[0][0].callback_data
        if stale:
            assert approval.resolve_gateway_approval(route, "deny", request_id="old") == 1
            approval._gateway_queues[route] = [successor]
        query = SimpleNamespace(data=callback, from_user=SimpleNamespace(id=123, first_name="owner"),
            message=SimpleNamespace(chat_id=123, chat=SimpleNamespace(type="supergroup"), message_thread_id=9),
            answer=AsyncMock(), edit_message_text=AsyncMock())
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        target = successor if stale else old
        assert target.event.is_set() is not stale
        if not stale:
            assert target.result == "once"
    finally:
        approval.clear_session(route)
