"""Tests for chunked gateway voice replies."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_voice_command import _make_event, _make_runner


class TestChunkedSendVoiceReply:
    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_send_voice_reply_sends_multiple_audio_chunks(self, runner):
        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        mock_adapter.send = AsyncMock()
        event = _make_event()
        runner.adapters[event.source.platform] = mock_adapter
        runner._voice_mode[event.source.chat_id] = "all"

        first_sentence = "A" * 2000 + "."
        second_sentence = "B" * 2000 + "."
        text = f"{first_sentence} {second_sentence}"

        tts_results = [
            json.dumps({"success": True, "file_path": "/tmp/test_part1.ogg"}),
            json.dumps({"success": True, "file_path": "/tmp/test_part2.ogg"}),
        ]

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=tts_results) as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda value: value), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            meta = await runner._send_voice_reply(event, text)

        assert mock_tts.call_count == 2
        assert mock_adapter.send_voice.await_count == 2
        assert mock_adapter.send.await_count == 2
        assert meta == {
            "sent_chunks": 2,
            "audio_primary": True,
            "status_text": None,
            "mirrored_text": True,
        }

    def test_audio_primary_requires_voice_tts_mode_and_multiple_chunks(self, runner):
        event = _make_event()

        runner._voice_mode[event.source.chat_id] = "all"
        assert runner._should_use_audio_primary_voice_reply(event, ["chunk 1", "chunk 2"]) is True

        runner._voice_mode[event.source.chat_id] = "voice_only"
        assert runner._should_use_audio_primary_voice_reply(event, ["chunk 1", "chunk 2"]) is False

        runner._voice_mode[event.source.chat_id] = "all"
        assert runner._should_use_audio_primary_voice_reply(event, ["single chunk"]) is False

    def test_disables_streaming_for_telegram_voice_tts_mode(self, runner):
        event = _make_event(chat_id="chat-1")
        event.source.platform = MagicMock()
        event.source.platform.value = "telegram"

        runner._voice_mode[event.source.chat_id] = "all"
        assert runner._should_disable_streaming_for_voice_mode(event.source) is True

        runner._voice_mode[event.source.chat_id] = "voice_only"
        assert runner._should_disable_streaming_for_voice_mode(event.source) is False
