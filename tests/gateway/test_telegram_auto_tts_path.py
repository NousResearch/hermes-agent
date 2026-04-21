import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, text=None, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_telegram_voice_input_auto_tts_requests_ogg_output(tmp_path, monkeypatch):
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._keep_typing = AsyncMock(return_value=None)
    adapter._run_processing_hook = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=None)
    adapter.play_tts = AsyncMock(return_value=None)
    adapter._message_handler = AsyncMock(return_value="Hello from Hermes")

    captured = {}
    output_file = tmp_path / "reply.ogg"
    output_file.write_bytes(b"ogg-bytes")

    def fake_tts_tool(*, text, output_path=None):
        captured["text"] = text
        captured["output_path"] = output_path
        return json.dumps({"success": True, "file_path": str(output_file)})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts_tool)
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)

    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm"),
    )
    event.message_id = "msg42"
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    assert captured["output_path"] is not None
    assert captured["output_path"].endswith(".ogg")
    adapter.play_tts.assert_awaited_once()
    played_path = adapter.play_tts.await_args.kwargs["audio_path"]
    assert played_path.endswith(".ogg")
    assert adapter._send_with_retry.await_count == 1
