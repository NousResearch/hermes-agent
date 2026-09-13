"""The written reply of a voice turn is delivered before the spoken one.

Regression for #109996: the post-handler auto-TTS block awaited every synthesized file
(``_play_tts_file`` plays it in the voice channel) and only then called ``_send_final_text``,
so the reader watched "typing…" for the entire playback — 35 s on a 517-character reply in a
Discord voice channel — with the answer already written. The order is now text first, skipped
only when the text rides the FIRST voice file as its caption: Telegram, first file, and still
within the caption limit. A Telegram reply past the limit gets no caption at all, so treating
"Telegram" as "the caption will carry it" brings the whole wait back on the longest replies.
"""

import asyncio
import json
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key

from unittest.mock import patch

CAPTION_LIMIT = 1024


class _DummyAdapter(BasePlatformAdapter):
    """A voice-capable adapter that records the order of its outbound deliveries."""

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.events = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.events.append(("text", content))
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        self.events.append(("audio", kwargs.get("caption")))
        return SendResult(success=True, message_id="tts-1")


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


def _voice_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=platform, chat_id="-1001", chat_type="group"),
        message_id="voice-1",
    )


async def _run_voice_turn(adapter: _DummyAdapter, reply: str) -> list:
    """Drive the real post-handler path: one voice turn answering with ``reply``."""
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=reply))
    event = _voice_event(adapter.platform)

    def fake_tts(*, text, output_path=None):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake audio")
        return json.dumps({"success": True, "file_path": output_path})

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(event, build_session_key(event.source))
    return adapter.events


def _reply_texts(events, reply):
    """Text deliveries of the reply itself (a long one may be split/truncated on the way out)."""
    return [value for kind, value in events if kind == "text" and value.startswith(reply[:24])]


@pytest.mark.parametrize(
    "platform,reply,text_first",
    [
        (Platform.DISCORD, "A reply that arrives together with its audio.", True),
        (Platform.TELEGRAM, "x" * (CAPTION_LIMIT + 1), True),
        (Platform.TELEGRAM, "x" * CAPTION_LIMIT, False),
        (Platform.TELEGRAM, "short reply", False),
    ],
)
@pytest.mark.asyncio
async def test_reply_text_and_audio_are_ordered_by_caption_eligibility(platform, reply, text_first):
    adapter = _DummyAdapter(platform)
    events = await _run_voice_turn(adapter, reply)

    audio_index = next(i for i, (kind, _) in enumerate(events) if kind == "audio")
    if text_first:
        text_index = next(
            i for i, (kind, value) in enumerate(events)
            if kind == "text" and value.startswith(reply[:24])
        )
        assert text_index < audio_index, events
    else:
        # The caption carries the reply: no separate send, and the audio is not waiting on it.
        assert _reply_texts(events, reply) == [], events
        assert events[audio_index][1] == reply, events
