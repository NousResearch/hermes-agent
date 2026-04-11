"""Test for WhatsApp send_voice() override.

Regression test for #4979: WhatsApp adapter was missing send_voice() override,
causing voice messages to be sent as text instead of native audio.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult


def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._bridge_script = "/tmp/test-bridge.js"
    adapter._session_path = Path("/tmp/test-wa-session")
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = MagicMock()
    return adapter


def test_send_voice_calls_send_media_to_bridge():
    """send_voice() should route audio through _send_media_to_bridge()."""
    async def run():
        adapter = _make_adapter()
        
        # Mock _send_media_to_bridge to verify it's called
        expected_result = SendResult(success=True, message_id="voice-123")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)
        
        result = await adapter.send_voice(
            chat_id="123456789@c.us",
            audio_path="/tmp/test-voice.ogg",
            caption="Test caption",
        )
        
        # Verify _send_media_to_bridge was called with correct args
        adapter._send_media_to_bridge.assert_called_once_with(
            "123456789@c.us",
            "/tmp/test-voice.ogg",
            "audio",
            "Test caption",
        )
        assert result == expected_result
    
    asyncio.run(run())


def test_send_voice_without_caption():
    """send_voice() should work without a caption."""
    async def run():
        adapter = _make_adapter()
        
        expected_result = SendResult(success=True, message_id="voice-456")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)
        
        result = await adapter.send_voice(
            chat_id="123456789@c.us",
            audio_path="/tmp/test-voice.ogg",
        )
        
        adapter._send_media_to_bridge.assert_called_once_with(
            "123456789@c.us",
            "/tmp/test-voice.ogg",
            "audio",
            None,  # No caption
        )
        assert result.success is True
    
    asyncio.run(run())


def test_send_voice_mp3_triggers_ffmpeg_conversion():
    """Non-ogg/opus files should be converted via ffmpeg before sending."""
    async def run():
        import shutil

        adapter = _make_adapter()
        expected_result = SendResult(success=True, message_id="voice-conv-1")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)

        # Create a real temp file — keep fd OPEN so the method's os.close(fd) succeeds
        fd, ogg_path = tempfile.mkstemp(suffix=".ogg")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        mock_subprocess = AsyncMock(return_value=mock_proc)

        try:
            with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch("tempfile.mkstemp", return_value=(fd, ogg_path)), \
                 patch("asyncio.create_subprocess_exec", mock_subprocess):
                result = await adapter.send_voice(
                    chat_id="123456789@c.us",
                    audio_path="/tmp/test-voice.mp3",
                )

            # ffmpeg should have been invoked
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0]
            assert call_args[0] == "/usr/bin/ffmpeg"
            assert "-c:a" in call_args
            assert "libopus" in call_args

            # The converted ogg path should have been sent
            sent_path = adapter._send_media_to_bridge.call_args[0][1]
            assert sent_path.endswith(".ogg")
            assert result.success is True
        finally:
            if os.path.exists(ogg_path):
                os.unlink(ogg_path)

    asyncio.run(run())


def test_send_voice_ffmpeg_not_available_sends_original():
    """When ffmpeg is not installed, send the original file as-is."""
    async def run():
        import shutil

        adapter = _make_adapter()
        expected_result = SendResult(success=True, message_id="voice-noffmpeg")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)

        with patch.object(shutil, "which", return_value=None):
            result = await adapter.send_voice(
                chat_id="123456789@c.us",
                audio_path="/tmp/test-voice.mp3",
            )

        # Original path should be used
        adapter._send_media_to_bridge.assert_called_once_with(
            "123456789@c.us",
            "/tmp/test-voice.mp3",
            "audio",
            None,
        )
        assert result.success is True

    asyncio.run(run())


def test_send_voice_ffmpeg_failure_sends_original():
    """When ffmpeg returns non-zero, fall back to the original file."""
    async def run():
        import shutil

        adapter = _make_adapter()
        expected_result = SendResult(success=True, message_id="voice-fail")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.wait = AsyncMock(return_value=1)

        fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
        # Keep fd open so the method's os.close(fd) succeeds

        try:
            with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch("tempfile.mkstemp", return_value=(fd, ogg_path)), \
                 patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
                result = await adapter.send_voice(
                    chat_id="123456789@c.us",
                    audio_path="/tmp/test-voice.wav",
                )

            # Should fall back to original path
            adapter._send_media_to_bridge.assert_called_once_with(
                "123456789@c.us",
                "/tmp/test-voice.wav",
                "audio",
                None,
            )
            assert result.success is True
        finally:
            if os.path.exists(ogg_path):
                os.unlink(ogg_path)

    asyncio.run(run())


def test_send_voice_ffmpeg_timeout_kills_process():
    """When ffmpeg times out, the process should be killed and temp file cleaned up."""
    async def run():
        import shutil

        adapter = _make_adapter()
        expected_result = SendResult(success=True, message_id="voice-timeout")
        adapter._send_media_to_bridge = AsyncMock(return_value=expected_result)

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=-9)

        fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
        # Keep fd open so the method's os.close(fd) succeeds

        try:
            with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"), \
                 patch("tempfile.mkstemp", return_value=(fd, ogg_path)), \
                 patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)), \
                 patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                result = await adapter.send_voice(
                    chat_id="123456789@c.us",
                    audio_path="/tmp/test-voice.mp3",
                )

            # Process should have been killed
            mock_proc.kill.assert_called_once()

            # Should fall back to original path
            adapter._send_media_to_bridge.assert_called_once_with(
                "123456789@c.us",
                "/tmp/test-voice.mp3",
                "audio",
                None,
            )
            assert result.success is True
        finally:
            if os.path.exists(ogg_path):
                os.unlink(ogg_path)

    asyncio.run(run())


def test_send_voice_method_exists():
    """WhatsAppAdapter should have its own send_voice implementation."""
    from gateway.platforms.whatsapp import WhatsAppAdapter
    from gateway.platforms.base import BasePlatformAdapter
    
    # Verify WhatsAppAdapter has its own send_voice (not inherited from base)
    assert hasattr(WhatsAppAdapter, 'send_voice')
    assert WhatsAppAdapter.send_voice is not BasePlatformAdapter.send_voice
