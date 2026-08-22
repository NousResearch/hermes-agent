"""Integration tests for the voice-delivery failure path.

These exercise ``BasePlatformAdapter._process_message_background`` end-to-end
for the reviewer scenario in the ``suppress_text_when_voice`` feature: the
auto-TTS voice is generated but its *send* fails. The written text must still
go out as a fallback, and the turn must be reported honestly — never masked as
SUCCESS by the text-suppression shortcut when no voice actually landed.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_voice_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=platform,
            chat_id="-1001",
            chat_type="group",
        ),
        message_id="voice-1",
    )


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


def _fake_tts(*, text, output_path=None):
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"fake audio")
    return json.dumps({"success": True, "file_path": output_path})


async def _run(adapter, *, voice_send_ok: bool, text_send_ok: bool):
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.config.suppress_text_when_voice = True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=voice_send_ok, message_id="tts-1", error=None if voice_send_ok else "rpc down")
    )
    if not text_send_ok:
        adapter._send_with_retry = AsyncMock(
            return_value=SendResult(success=False, error="rpc down")
        )
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="reply text"))
    event = _make_voice_event(Platform.SIGNAL)

    outcomes = []

    async def _capture(hook_name, *args, **kwargs):
        if hook_name == "on_processing_complete":
            outcomes.append(args[-1])

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=_fake_tts
    ), patch.object(adapter, "_run_processing_hook", side_effect=_capture):
        await adapter._process_message_background(event, build_session_key(event.source))
    return adapter, outcomes


@pytest.mark.asyncio
async def test_voice_send_failure_falls_back_to_text():
    """When the auto-TTS voice *send* fails, the text reply still goes out."""
    adapter, outcomes = await _run(
        _DummyAdapter(Platform.SIGNAL), voice_send_ok=False, text_send_ok=True
    )
    adapter.play_tts.assert_awaited()  # voice was attempted
    assert any(m["content"] == "reply text" for m in adapter.sent), adapter.sent


@pytest.mark.asyncio
async def test_voice_send_failure_with_failed_text_reports_failure():
    """Voice send fails *and* text fallback fails -> turn reports FAILURE.

    This is the case the old ``if _suppress_text:`` shortcut got wrong: it
    would mark the turn SUCCESS even though neither audio nor text landed.
    """
    adapter, outcomes = await _run(
        _DummyAdapter(Platform.SIGNAL), voice_send_ok=False, text_send_ok=False
    )
    assert outcomes, "on_processing_complete hook did not fire"
    assert outcomes[-1] is ProcessingOutcome.FAILURE


@pytest.mark.asyncio
async def test_voice_delivered_suppresses_text_and_reports_success():
    """Voice delivered OK -> text suppressed and turn reported SUCCESS."""
    adapter, outcomes = await _run(
        _DummyAdapter(Platform.SIGNAL), voice_send_ok=True, text_send_ok=True
    )
    adapter.play_tts.assert_awaited()
    # Text was suppressed (no redundant text), but the turn still succeeded.
    assert not any(m["content"] == "reply text" for m in adapter.sent), adapter.sent
    assert outcomes and outcomes[-1] is ProcessingOutcome.SUCCESS


@pytest.mark.asyncio
async def test_media_voice_clip_suppresses_text():
    """An agent-attached voice clip (``[[audio_as_voice]]`` + ``MEDIA:`` tag)
    also suppresses the redundant text, matching the auto-TTS path.

    This is the agent-tool voice path: the model calls ``text_to_speech``,
    which returns ``[[audio_as_voice]]\\nMEDIA:<path>``, and the gateway must
    drop the written text the same way it does for auto-TTS voice.
    """
    from pathlib import Path

    adapter = _DummyAdapter(Platform.SIGNAL)
    adapter.config.suppress_text_when_voice = True
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: False  # no auto-TTS
    adapter.send_voice = AsyncMock(
        return_value=SendResult(success=True, message_id="voice-1")
    )

    voice_path = Path("/tmp/voice_test_clip.ogg")
    voice_path.write_bytes(b"fake audio")

    adapter.set_message_handler(
        lambda _event: asyncio.sleep(
            0, result=f"reply text\n\n[[audio_as_voice]]\nMEDIA:{voice_path}"
        )
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SIGNAL, chat_id="-1001", chat_type="group"
        ),
        message_id="m-1",
    )

    with patch("tools.tts_tool.check_tts_requirements", return_value=True):
        await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited()
    assert not any(m["content"] == "reply text" for m in adapter.sent), adapter.sent


@pytest.mark.asyncio
async def test_media_voice_clip_send_failure_falls_back_to_text():
    """If an agent-attached voice clip fails to send, the text fallback goes out
    (deferred text is released instead of leaving the user with nothing)."""
    from pathlib import Path

    adapter = _DummyAdapter(Platform.SIGNAL)
    adapter.config.suppress_text_when_voice = True
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: False
    adapter.send_voice = AsyncMock(
        return_value=SendResult(success=False, message_id=None, error="rpc down")
    )

    voice_path = Path("/tmp/voice_test_clip_fail.ogg")
    voice_path.write_bytes(b"fake audio")

    adapter.set_message_handler(
        lambda _event: asyncio.sleep(
            0, result=f"reply text\n\n[[audio_as_voice]]\nMEDIA:{voice_path}"
        )
    )
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SIGNAL, chat_id="-1001", chat_type="group"
        ),
        message_id="m-2",
    )

    with patch("tools.tts_tool.check_tts_requirements", return_value=True):
        await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited()
    assert any(m["content"] == "reply text" for m in adapter.sent), adapter.sent
