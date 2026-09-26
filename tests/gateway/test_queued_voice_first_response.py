"""Queued first replies preserve the originating voice modality."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_accepted", [True, False])
async def test_queued_voice_first_reply_speaks_once_with_text_fallback(tmp_path, monkeypatch, voice_accepted):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner._run_agent_stream_confirmed_final_delivery = lambda *_a, **_kw: False
    runner._is_intentional_silence = lambda *_a: False
    runner._pop_post_delivery_callback = lambda *_a: None
    turn_ctx = SimpleNamespace(
        mute_notification_reply=False, session_key="agent:main:telegram:dm:chat-1",
        stream_consumer_holder=[None], source=source, _status_thread_metadata={},
        event_message_id="message-1", inbound_message_id="message-1", run_generation=1,
    )
    audio = tmp_path / "speech.ogg"
    audio.write_bytes(b"audio")
    adapter._should_auto_tts_for_chat = lambda _chat: True
    adapter._synthesize_auto_tts = AsyncMock(return_value=([str(audio)], str(audio)))
    adapter.play_tts = AsyncMock(return_value=SendResult(success=voice_accepted, message_id="voice"))
    adapter.send_final_ledgered = AsyncMock(return_value=(SendResult(success=True, message_id="text"), None))
    result = {"final_response": "The answer", "messages": []}

    await runner._run_agent_deliver_first_response(
        turn_ctx, adapter, result, result, None, message_type=MessageType.VOICE,
    )

    adapter.play_tts.assert_awaited_once()
    assert adapter.play_tts.await_args.kwargs["caption"] == "The answer"
    assert adapter.send_final_ledgered.await_count == (0 if voice_accepted else 1)
    assert result["already_sent"] is True
    assert not audio.exists()
    # An early return to the normal completion path must not speak an accepted
    # queued voice reply a second time, but may retry TTS after failed audio.
    runner._delivery_adapter_for = lambda _source: adapter
    runner._should_send_voice_reply = lambda *_args, **_kwargs: True
    runner._send_voice_reply = AsyncMock()
    await runner._hmwa_deliver_turn_response(
        MessageEvent(text="question", source=source, message_type=MessageType.VOICE),
        source, SimpleNamespace(session_id="sid"), turn_ctx.session_key, 1,
        result, [], "The answer", None, False,
    )
    assert runner._send_voice_reply.await_count == (0 if voice_accepted else 1)
