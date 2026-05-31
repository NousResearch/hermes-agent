"""Regression test for issue #36005.

Telegram's Bot API only auto-derives duration from container metadata for
short clips.  For voice/audio longer than ~4 min 50 s it delivers the message
with duration 0 unless the sender passes an explicit ``duration`` kwarg.

This test verifies that:
1. ``_probe_audio_duration`` returns a sensible integer for OGG and MP3 files.
2. ``TelegramAdapter.send_voice`` passes ``duration`` through to the Bot API
   for both voice (ogg/opus) and audio (mp3/m4a) paths.
"""

import os
import struct
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.telegram import _probe_audio_duration


# ---------------------------------------------------------------------------
# _probe_audio_duration unit tests
# ---------------------------------------------------------------------------

class TestProbeAudioDuration:
    """Unit tests for the ``_probe_audio_duration`` helper."""

    def test_returns_none_for_missing_file(self):
        assert _probe_audio_duration("/nonexistent/path.ogg") is None

    def test_ogg_file_size_fallback(self, tmp_path):
        """Without mutagen, falls back to file-size estimate for OGG."""
        ogg = tmp_path / "voice.ogg"
        # ~100 KB → 100000 / 2000 = 50 seconds
        ogg.write_bytes(b"\x00" * 100_000)
        result = _probe_audio_duration(str(ogg))
        assert result is not None
        assert result >= 1
        # Should be roughly 50s (±20% tolerance for rounding)
        assert 40 <= result <= 60

    def test_mp3_file_size_fallback(self, tmp_path):
        """Without mutagen, falls back to file-size estimate for MP3."""
        mp3 = tmp_path / "audio.mp3"
        # ~256 KB → 256000 / 16000 = 16 seconds
        mp3.write_bytes(b"\x00" * 256_000)
        result = _probe_audio_duration(str(mp3))
        assert result is not None
        assert result >= 1
        assert 12 <= result <= 20

    def test_minimum_duration_is_one(self, tmp_path):
        """Even a tiny file should report at least 1 second."""
        tiny = tmp_path / "tiny.ogg"
        tiny.write_bytes(b"\x00" * 10)
        result = _probe_audio_duration(str(tiny))
        assert result is not None
        assert result >= 1


# ---------------------------------------------------------------------------
# Integration: send_voice passes duration
# ---------------------------------------------------------------------------

class TestSendVoicePassesDuration:
    """Verify that ``TelegramAdapter.send_voice`` forwards ``duration``."""

    @pytest.mark.asyncio
    async def test_voice_path_includes_duration(self, tmp_path):
        """OGG voice calls should include ``duration`` in kwargs."""
        from gateway.platforms.telegram import TelegramAdapter
        from gateway.config import PlatformConfig, Platform

        ogg = tmp_path / "voice.ogg"
        ogg.write_bytes(b"\x00" * 200_000)  # ~100s at 2kB/s

        adapter = object.__new__(TelegramAdapter)
        adapter._bot = MagicMock()
        adapter._reply_to_mode = "quote"

        # Mock internal helpers
        adapter._metadata_thread_id = MagicMock(return_value=None)
        adapter._reply_to_message_id_for_send = MagicMock(return_value=None)
        adapter._thread_kwargs_for_send = MagicMock(return_value={})
        adapter._notification_kwargs = MagicMock(return_value={})

        sent_kwargs = {}

        async def _capture_send_voice(**kwargs):
            sent_kwargs.update(kwargs)
            return SimpleNamespace(message_id=42)

        async def _fake_retry(fn, kw, *args, **kwargs):
            return await _capture_send_voice(**kw)

        adapter._send_with_dm_topic_reply_anchor_retry = AsyncMock(side_effect=_fake_retry)

        with patch("os.path.exists", return_value=True):
            result = await adapter.send_voice(
                chat_id="12345",
                audio_path=str(ogg),
                caption=None,
                reply_to=None,
                metadata=None,
            )

        assert result.success is True
        assert "duration" in sent_kwargs
        assert isinstance(sent_kwargs["duration"], int)
        assert sent_kwargs["duration"] >= 1

    @pytest.mark.asyncio
    async def test_audio_path_includes_duration(self, tmp_path):
        """MP3 audio calls should include ``duration`` in kwargs."""
        from gateway.platforms.telegram import TelegramAdapter

        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"\x00" * 320_000)  # ~20s at 16kB/s

        adapter = object.__new__(TelegramAdapter)
        adapter._bot = MagicMock()
        adapter._reply_to_mode = "quote"

        adapter._metadata_thread_id = MagicMock(return_value=None)
        adapter._reply_to_message_id_for_send = MagicMock(return_value=None)
        adapter._thread_kwargs_for_send = MagicMock(return_value={})
        adapter._notification_kwargs = MagicMock(return_value={})

        sent_kwargs = {}

        async def _capture_send_audio(**kwargs):
            sent_kwargs.update(kwargs)
            return SimpleNamespace(message_id=43)

        async def _fake_retry(fn, kw, *args, **kwargs):
            return await _capture_send_audio(**kw)

        adapter._send_with_dm_topic_reply_anchor_retry = AsyncMock(side_effect=_fake_retry)

        with patch("os.path.exists", return_value=True):
            result = await adapter.send_voice(
                chat_id="12345",
                audio_path=str(mp3),
                caption="test",
                reply_to=None,
                metadata=None,
            )

        assert result.success is True
        assert "duration" in sent_kwargs
        assert isinstance(sent_kwargs["duration"], int)
        assert sent_kwargs["duration"] >= 1
