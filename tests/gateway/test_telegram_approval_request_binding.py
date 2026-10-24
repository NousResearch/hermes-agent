"""Native Telegram cards must never resolve a successor's foreground consent."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['stale', 'valid', 'unauthorized', 'chat', 'thread', 'legacy'])
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
                metadata={'thread_id': '9', 'approval_request_id': 'old'})
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
