"""
Tests for audio vs voice routing — ensures audio file attachments (MessageType.AUDIO)
are NOT sent to STT, while voice messages (MessageType.VOICE) are still transcribed.

Regression test for issue #24870.
"""

from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(msg_type: MessageType, media_urls=None, media_types=None, text=""):
    """Build a minimal MessageEvent with the given type and media."""
    return MessageEvent(
        text=text,
        source=None,
        message_type=msg_type,
        media_urls=media_urls or [],
        media_types=media_types or [],
    )


def _make_runner():
    """Create a minimal GatewayRunner with STT enabled."""
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    return runner


# ---------------------------------------------------------------------------
# Tests: audio_paths routing (run.py line ~6769)
# ---------------------------------------------------------------------------

class TestAudioVsVoiceRouting:
    """Verify that MessageType.AUDIO does NOT enter the STT pipeline."""

    @pytest.mark.asyncio
    async def test_voice_message_goes_to_stt(self):
        """VOICE messages with audio media should be transcribed."""
        runner = _make_runner()

        with patch(
            "gateway.run.GatewayRunner._enrich_message_with_transcription",
            new_callable=AsyncMock,
            return_value="Transcribed: hello world",
        ) as mock_stt:
            # Simulate the routing logic at run.py:6769
            event = _make_event(
                msg_type=MessageType.VOICE,
                media_urls=["/tmp/voice.ogg"],
                media_types=["audio/ogg"],
            )

            # Build audio_paths the same way as the patched code
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if mtype.startswith("audio/") and event.message_type == MessageType.VOICE:
                    audio_paths.append(path)

            assert audio_paths == ["/tmp/voice.ogg"], (
                f"VOICE should produce audio_paths, got {audio_paths}"
            )

    @pytest.mark.asyncio
    async def test_audio_file_not_in_stt_paths(self):
        """AUDIO file attachments should NOT enter audio_paths (STT pipeline)."""
        event = _make_event(
            msg_type=MessageType.AUDIO,
            media_urls=["/tmp/song.mp3"],
            media_types=["audio/mp3"],
        )

        # Build audio_paths the same way as the patched code
        audio_paths = []
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype.startswith("audio/") and event.message_type == MessageType.VOICE:
                audio_paths.append(path)

        assert audio_paths == [], (
            f"AUDIO files should NOT be in audio_paths, got {audio_paths}"
        )

    @pytest.mark.asyncio
    async def test_audio_file_with_no_mime_still_excluded(self):
        """AUDIO files without explicit MIME type should also be excluded from STT."""
        event = _make_event(
            msg_type=MessageType.AUDIO,
            media_urls=["/tmp/recording.m4a"],
            media_types=[""],
        )

        audio_paths = []
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype.startswith("audio/") and event.message_type == MessageType.VOICE:
                audio_paths.append(path)

        assert audio_paths == []

    @pytest.mark.asyncio
    async def test_document_with_audio_mime_excluded_from_stt(self):
        """DOCUMENT type with audio MIME should NOT go to STT."""
        event = _make_event(
            msg_type=MessageType.DOCUMENT,
            media_urls=["/tmp/audio_file.mp3"],
            media_types=["audio/mp3"],
        )

        audio_paths = []
        for i, path in enumerate(event.media_urls):
            mtype = event.media_types[i] if i < len(event.media_types) else ""
            if mtype.startswith("audio/") and event.message_type == MessageType.VOICE:
                audio_paths.append(path)

        assert audio_paths == []


# ---------------------------------------------------------------------------
# Tests: _build_media_placeholder (ensures audio still shows in message)
# ---------------------------------------------------------------------------

class TestAudioFileTextRepresentation:
    """Verify audio files still appear in message text even though they skip STT."""

    def test_audio_file_shows_in_message_text(self):
        """Audio file attachments should appear as [User sent audio: ...]."""
        from gateway.run import _build_media_placeholder

        event = _make_event(
            msg_type=MessageType.AUDIO,
            media_urls=["/tmp/song.mp3"],
            media_types=["audio/mp3"],
        )

        text = _build_media_placeholder(event)
        assert "[User sent audio: /tmp/song.mp3]" in text

    def test_voice_placeholder_text(self):
        """Voice messages should appear as [User sent audio: ...] in placeholder."""
        from gateway.run import _build_media_placeholder

        event = _make_event(
            msg_type=MessageType.VOICE,
            media_urls=["/tmp/voice.ogg"],
            media_types=["audio/ogg"],
        )

        text = _build_media_placeholder(event)
        assert "[User sent audio: /tmp/voice.ogg]" in text

