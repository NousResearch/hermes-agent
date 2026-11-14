"""Regression coverage for voice completion delivery across settlement."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_ingress import executing_admission
from tests.gateway.test_auto_voice_reply_format import _make_adapter, _make_runner


@pytest.mark.asyncio
async def test_streamed_completion_retains_voice_candidate_when_mode_is_initially_off():
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=False)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode['telegram:123'] = 'off'
    runner._send_voice_reply = AsyncMock()
    event = MessageEvent(text='completion', source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'),
                         internal=True, metadata={'presentation_origin': 'voice'})
    token = executing_admission.set(True)
    try:
        result = await runner._hmwa_deliver_turn_response(
            event, event.source, SimpleNamespace(session_id='s'), 's', 1,
            {'already_sent': True}, [], 'answer', '', False)
    finally:
        executing_admission.reset(token)
    assert result is None  # the text was already streamed
    runner._send_voice_reply.assert_not_awaited()  # no audio before settlement
    assert getattr(event, '_deferred_voice_response', None) == 'answer'
