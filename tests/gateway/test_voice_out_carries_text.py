"""Contract tests for ``voice_out_carries_text`` (audio/text suppression).

Some platforms deliver the reply TEXT as part of a successful ``play_tts()``
send — e.g. Carbon Voice transcribes the audio server-side and renders the
transcript inline with the voice memo. On those platforms the follow-up text
send in ``_process_message_background`` is pure duplication: the user gets
the same reply twice (audio memo + text bubble).

Adapters declare this with the ``voice_out_carries_text`` class attribute
(default False on ``BasePlatformAdapter``). The response flow suppresses the
text send only when the flag is set AND ``play_tts()`` reported success — a
failed audio send must still fall back to text so the reply is never lost.

Telegram's caption-based suppression (length-conditional, caption attached to
the voice message itself) is a separate mechanism and remains unchanged.
"""

import asyncio
import json
import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _VoiceDummy(BasePlatformAdapter):
    """Minimal adapter recording text sends and play_tts calls."""

    def __init__(self, platform: Platform, *, tts_success: bool = True):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent: list[dict] = []
        self.tts_calls: list[dict] = []
        self._tts_success = tts_success

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def play_tts(self, chat_id, audio_path, caption=None, metadata=None, **kwargs):
        self.tts_calls.append({"chat_id": chat_id, "audio_path": audio_path})
        if self._tts_success:
            return SendResult(success=True, message_id="voice-1")
        return SendResult(success=False, error="upload failed")


class _CarriesTextDummy(_VoiceDummy):
    voice_out_carries_text = True


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _make_voice_event(platform: Platform, chat_id: str = "chan-1") -> MessageEvent:
    return MessageEvent(
        text="hola, ¿cómo va todo?",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
        message_id="m1",
    )


def _wire_tts(adapter, monkeypatch, tmp_path):
    """Make the auto-TTS gate fire and the TTS tool produce a real file."""
    adapter._keep_typing = _hold_typing
    adapter._auto_tts_default = True  # _should_auto_tts_for_chat → True

    def _fake_tts(text: str):
        audio = tmp_path / "reply.mp3"
        audio.write_bytes(b"fake-audio")
        return json.dumps({"file_path": str(audio)})

    import tools.tts_tool as tts_tool

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _fake_tts)

    async def handler(_event):
        return "El deploy terminó sin errores."

    adapter.set_message_handler(handler)


@pytest.mark.asyncio
async def test_carries_text_suppresses_followup_text(monkeypatch, tmp_path, caplog):
    """Flag set + play_tts success → audio only, no duplicate text bubble."""
    adapter = _CarriesTextDummy(Platform.DISCORD)
    _wire_tts(adapter, monkeypatch, tmp_path)

    event = _make_voice_event(Platform.DISCORD)
    with caplog.at_level(logging.ERROR, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.tts_calls) == 1, "play_tts must be invoked once"
    assert adapter.sent == [], f"text must be suppressed, got {adapter.sent}"
    # The audio counted as a delivery — no false silent-drop alarm.
    assert "response_delivery_dropped" not in caplog.text


@pytest.mark.asyncio
async def test_default_flag_still_sends_text_after_tts(monkeypatch, tmp_path):
    """Default (False) keeps today's behavior: audio AND text are sent."""
    adapter = _VoiceDummy(Platform.DISCORD)
    _wire_tts(adapter, monkeypatch, tmp_path)

    event = _make_voice_event(Platform.DISCORD)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.tts_calls) == 1
    assert len(adapter.sent) == 1, "text send must NOT be suppressed by default"


@pytest.mark.asyncio
async def test_carries_text_falls_back_to_text_when_tts_send_fails(
    monkeypatch, tmp_path
):
    """Flag set but play_tts FAILED → the reply must still arrive as text."""
    adapter = _CarriesTextDummy(Platform.DISCORD, tts_success=False)
    _wire_tts(adapter, monkeypatch, tmp_path)

    event = _make_voice_event(Platform.DISCORD)
    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.tts_calls) == 1
    assert len(adapter.sent) == 1, "failed audio send must fall back to text"


def test_carbonvoice_adapter_declares_carries_text():
    """The Carbon Voice adapter opts in: its transcript IS the text."""
    from plugins.platforms.carbonvoice.adapter import CarbonVoiceAdapter

    assert CarbonVoiceAdapter.voice_out_carries_text is True
    assert BasePlatformAdapter.voice_out_carries_text is False
