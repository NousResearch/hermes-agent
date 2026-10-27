"""A retained Telegram card cannot acquire a new adapter's request identity."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.mark.asyncio
async def test_old_card_cannot_resolve_first_request_after_adapter_restart(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram
    monkeypatch.setattr(telegram, 'InlineKeyboardButton', lambda label, callback_data: SimpleNamespace(callback_data=callback_data))
    monkeypatch.setattr(telegram, 'InlineKeyboardMarkup', lambda rows: SimpleNamespace(inline_keyboard=rows))
    route = 'restart-native-binding'
    adapters, callbacks = [], []
    try:
        for request_id in ('old', 'new'):
            adapter = TelegramAdapter(PlatformConfig(enabled=True, token='test-token'))
            adapter._bot = AsyncMock()
            adapter._bot.send_message.return_value = SimpleNamespace(message_id=42)
            monkeypatch.setattr(adapter, '_is_callback_user_authorized', lambda *a, **kw: True)
            await adapter.send_exec_approval(chat_id='123', command='fixture', session_key=route,
                metadata={'thread_id': '9', 'approval_request_id': request_id})
            markup = adapter._bot.send_message.call_args.kwargs['reply_markup']
            callbacks.append(markup.inline_keyboard[0][0].callback_data)
            assert all(len(button.callback_data.encode('utf-8')) <= 64
                       for row in markup.inline_keyboard for button in row)
            adapters.append(adapter)
        successor = _ApprovalEntry({'request_id': 'new'})
        approval._gateway_queues[route] = [successor]
        for callback, expected in ((callbacks[0], False), (callbacks[1], True)):
            query = SimpleNamespace(data=callback, from_user=SimpleNamespace(id=123, first_name='owner'),
                message=SimpleNamespace(chat_id=123, chat=SimpleNamespace(type='supergroup'), message_thread_id=9),
                answer=AsyncMock(), edit_message_text=AsyncMock())
            await adapters[1]._handle_callback_query(SimpleNamespace(callback_query=query), None)
            assert successor.event.is_set() is expected
        assert successor.result == 'once'
        assert callbacks[0] != callbacks[1]
    finally:
        approval.clear_session(route)
