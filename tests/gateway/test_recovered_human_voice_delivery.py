"""Recovered human voice replies are spoken only after successful settlement."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_ingress import executing_admission
from tests.gateway.test_auto_voice_reply_format import _make_adapter, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize('native_waiter', [False, True])
async def test_admitted_human_voice_defers_only_without_native_waiter(native_waiter):
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._send_voice_reply = AsyncMock()
    event = MessageEvent(text='voice', source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'),
                         message_type=MessageType.VOICE)
    event._adapter_drains_after = native_waiter
    result = {'final_response': 'answer'}
    token = executing_admission.set(True)
    try:
        await runner._hmwa_deliver_turn_response(event, event.source, SimpleNamespace(session_id='s'),
                                                 's', 1, result, [], 'answer', '', False)
    finally:
        executing_admission.reset(token)
    assert (getattr(event, '_deferred_voice_execution', None) is result) is not native_waiter
    runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_voice_skips_synthesis_for_media_response():
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    runner.adapters[Platform.TELEGRAM] = adapter
    event = MessageEvent(text='voice', source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'),
                         message_type=MessageType.VOICE)
    event._adapter_drains_after = False
    event._voice_result_committed = True
    runner._session_key_for_source = lambda source: 'session'
    adapter._extract_response_content = AsyncMock(return_value=SimpleNamespace(text_content='answer', media_files=[('a.ogg', False)]))
    runner._deliver_voice_reply = AsyncMock()
    await runner._send_voice_reply(event, 'answer MEDIA:a.ogg')
    runner._deliver_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_settled_recovered_voice_uses_real_delivery_path(monkeypatch):
    from gateway import session_ingress
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
    adapter._unwrap_ephemeral = lambda value: (value, 0)
    adapter._extract_response_content = AsyncMock(return_value=SimpleNamespace(
        text_content='answer', media_files=[]))
    async def send_text(event, key, text, metadata, ephemeral, ttl, record):
        record(SimpleNamespace(success=True))
    adapter._send_final_text = AsyncMock(side_effect=send_text)
    adapter._deliver_attachments = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._send_voice_reply = AsyncMock()
    event = MessageEvent(text='voice', source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'),
                         message_type=MessageType.VOICE)
    event._adapter_drains_after = False
    event._voice_result_committed = True
    event._deferred_voice_response = 'answer'
    event._deferred_voice_messages = []
    authority = SimpleNamespace(runner=runner, pending_deliveries={
        'id': (adapter, event, 'session', event._deferred_voice_response, None)},
        native_waiters=set())
    monkeypatch.setattr(session_ingress, '_held_markers', lambda authority: {})
    monkeypatch.setattr(session_ingress, '_execution_scope', AsyncMock(return_value=__import__('contextlib').nullcontext()))
    monkeypatch.setattr(session_ingress, '_release_turn_marker', AsyncMock())
    monkeypatch.setattr(session_ingress, 'close_producer_turn', AsyncMock())
    await session_ingress.deliver_settled(authority, 'id')
    adapter._send_final_text.assert_awaited_once()
    runner._send_voice_reply.assert_awaited_once_with(event, 'answer')
