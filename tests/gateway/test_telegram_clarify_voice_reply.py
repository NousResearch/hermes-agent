from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    cm._entries.clear()
    cm._session_index.clear()
    cm._notify_cbs.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
        user_name="tester",
    )


def _make_runner(adapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._is_user_authorized = lambda _source: True
    runner._session_key_for_source = lambda _source: "telegram:dm:12345"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda _event: None
    return runner


@pytest.mark.asyncio
async def test_pending_telegram_clarify_accepts_voice_reply():
    from tools import clarify_gateway as cm

    _clear_clarify_state()
    cm.register("cid-voice", "telegram:dm:12345", "How should I proceed?", None)

    adapter = SimpleNamespace(send=AsyncMock())
    runner = _make_runner(adapter)
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=('"spoken answer"', ["spoken answer"])
    )

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_type=MessageType.VOICE,
        media_urls=["/tmp/reply.ogg"],
        media_types=["audio/ogg"],
    )

    result = await runner._handle_message(event)

    assert result == ""
    assert cm.wait_for_response("cid-voice", timeout=0.1) == "spoken answer"
    runner._enrich_message_with_transcription.assert_awaited_once_with(
        "", ["/tmp/reply.ogg"]
    )
    adapter.send.assert_awaited_once_with("12345", '🎙️ "spoken answer"', metadata={})
