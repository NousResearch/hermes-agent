"""Final-text delivery evidence passed to generation-bound post-delivery callbacks."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _ResultAdapter(BasePlatformAdapter):
    def __init__(self, result: SendResult):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.result = result

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return self.result

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u1"),
        message_id="in-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("send_success", [True, False])
async def test_buffered_callback_observes_exact_final_text_send_result(send_success):
    wire_result = SendResult(
        success=send_success,
        message_id="out-1" if send_success else None,
        error=None if send_success else "connector rejected",
    )
    adapter = _ResultAdapter(wire_result)
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="final text"))
    session_key = "agent:main:telegram:dm:1"
    guard = asyncio.Event()
    guard._hermes_run_generation = 7
    adapter._active_sessions[session_key] = guard
    legacy = []
    observed = []
    adapter.register_post_delivery_callback(
        session_key, lambda: legacy.append("fired"), generation=7)
    adapter.register_post_delivery_callback(
        session_key, lambda result: observed.append(result), generation=7)

    await adapter._process_message_background(_event(), session_key)

    assert observed == [wire_result]
    assert legacy == ["fired"]


@pytest.mark.asyncio
async def test_telegram_auto_tts_caption_preserves_its_send_result(tmp_path):
    wire_result = SendResult(success=True, message_id="voice-1")
    adapter = _ResultAdapter(SendResult(success=True, message_id="unexpected-text"))
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(return_value=wire_result)
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="final text"))
    event = _event()
    event.message_type = MessageType.VOICE
    session_key = "agent:main:telegram:dm:1"
    guard = asyncio.Event()
    guard._hermes_run_generation = 9
    adapter._active_sessions[session_key] = guard
    observed = []
    adapter.register_post_delivery_callback(
        session_key, lambda result: observed.append(result), generation=9)
    tts_path = tmp_path / "reply.ogg"
    tts_path.write_text("audio", encoding="utf-8")

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool",
        return_value=json.dumps({"file_path": str(tts_path)}),
    ):
        await adapter._process_message_background(event, session_key)

    assert observed == [wire_result]
    adapter.play_tts.assert_awaited_once()
    assert adapter.play_tts.await_args.kwargs["caption"] == "final text"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "expected_success", "expected_error"),
    [
        (True, True, None),
        (False, False, "streamed final text delivery failed"),
        (None, False, "streamed final text delivery unknown"),
    ],
)
async def test_streamed_callback_fails_closed_on_exact_text_ledger(
    verdict, expected_success, expected_error,
):
    class _Consumer:
        message_id = "stream-1"

        def delivered_final_matches(self, _text):
            return verdict

    adapter = _ResultAdapter(SendResult(success=True, message_id="unused"))
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: adapter
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._deliver_media_from_response = AsyncMock()

    async def _handler(event):
        agent_result = {
            "already_sent": True,
            "_final_text_delivery_result": (
                GatewayRunner._run_agent_stream_final_delivery_result(
                    _Consumer(), "final text")),
        }
        return await GatewayRunner._hmwa_deliver_turn_response(
            runner, event, event.source, SimpleNamespace(session_id="session-1"),
            "agent:main:telegram:dm:1", 8, agent_result, [], "final text", None, False,
        )

    adapter.set_message_handler(_handler)
    session_key = "agent:main:telegram:dm:1"
    guard = asyncio.Event()
    guard._hermes_run_generation = 8
    adapter._active_sessions[session_key] = guard
    observed = []
    adapter.register_post_delivery_callback(
        session_key, lambda result: observed.append(result), generation=8)

    await adapter._process_message_background(_event(), session_key)

    assert len(observed) == 1
    assert isinstance(observed[0], SendResult)
    assert observed[0].success is expected_success
    assert observed[0].error == expected_error
    assert observed[0].message_id == ("stream-1" if expected_success else None)

    async def _silent_handler(event):
        return await GatewayRunner._hmwa_deliver_turn_response(
            runner, event, event.source, SimpleNamespace(session_id="session-1"),
            session_key, 10, {}, [], "[SILENT]", None, True,
        )

    adapter.set_message_handler(_silent_handler)
    silence_guard = asyncio.Event()
    silence_guard._hermes_run_generation = 10
    adapter._active_sessions[session_key] = silence_guard
    adapter.register_post_delivery_callback(
        session_key, lambda result: observed.append(result), generation=10)

    await adapter._process_message_background(_event(), session_key)
    assert observed[-1] is None

    stale_guard = asyncio.Event()
    stale_guard._hermes_run_generation = 12
    adapter._active_sessions[session_key] = stale_guard
    adapter.register_post_delivery_callback(
        session_key, lambda result: observed.append(result), generation=11)

    await adapter._process_message_background(_event(), session_key)
    assert len(observed) == 2
    assert session_key in adapter._post_delivery_callbacks
