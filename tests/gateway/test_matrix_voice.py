"""Tests for Matrix voice message support (MSC3245)."""
import io
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Try importing real nio; skip entire file if not available.
# A MagicMock in sys.modules (from another test) is not the real package.
try:
    import nio as _nio_probe
    if not isinstance(_nio_probe, types.ModuleType) or not hasattr(_nio_probe, "__file__"):
        pytest.skip("nio in sys.modules is a mock, not the real package", allow_module_level=True)
except ImportError:
    pytest.skip("matrix-nio not installed", allow_module_level=True)

from gateway.platforms.base import MessageType


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a MatrixAdapter with mocked config."""
    from gateway.platforms.matrix import MatrixAdapter
    from gateway.config import PlatformConfig
    
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    return adapter


def _make_room(room_id: str = "!test:example.org", member_count: int = 2):
    """Create a mock Matrix room."""
    room = MagicMock()
    room.room_id = room_id
    room.member_count = member_count
    return room


def _make_audio_event(
    event_id: str = "$audio_event",
    sender: str = "@alice:example.org",
    body: str = "Voice message",
    url: str = "mxc://example.org/abc123",
    is_voice: bool = False,
    mimetype: str = "audio/ogg",
    timestamp: float = 9999999999000,  # ms
):
    """
    Create a mock RoomMessageAudio event that passes isinstance checks.
    
    Args:
        is_voice: If True, adds org.matrix.msc3245.voice field to content
    """
    import nio
    
    # Build the source dict that nio events expose via .source
    content = {
        "msgtype": "m.audio",
        "body": body,
        "url": url,
        "info": {
            "mimetype": mimetype,
        },
    }
    
    if is_voice:
        content["org.matrix.msc3245.voice"] = {}
    
    # Create a real nio RoomMessageAudio-like object
    # We use MagicMock but configure __class__ to pass isinstance check
    event = MagicMock(spec=nio.RoomMessageAudio)
    event.event_id = event_id
    event.sender = sender
    event.body = body
    event.url = url
    event.server_timestamp = timestamp
    event.source = {
        "type": "m.room.message",
        "content": content,
    }
    # For MIME type extraction - needs to be a dict
    event.content = content
    
    return event


def _make_download_response(body: bytes = b"fake audio data"):
    """Create a mock nio.MemoryDownloadResponse."""
    import nio
    resp = MagicMock()
    resp.body = body
    resp.__class__ = nio.MemoryDownloadResponse
    return resp


# ---------------------------------------------------------------------------
# Tests: MSC3245 Voice Detection (RED -> GREEN)
# ---------------------------------------------------------------------------

class TestMatrixVoiceMessageDetection:
    """Test that MSC3245 voice messages are detected and tagged correctly."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._startup_ts = 0.0
        self.adapter._dm_rooms = {}
        self.adapter._message_handler = AsyncMock()
        # Mock _mxc_to_http to return a fake HTTP URL
        self.adapter._mxc_to_http = lambda url: f"https://matrix.example.org/_matrix/media/v3/download/{url[6:]}"
        # Mock client for authenticated download
        self.adapter._client = MagicMock()
        self.adapter._client.download = AsyncMock(return_value=_make_download_response())

    @pytest.mark.asyncio
    async def test_voice_message_has_type_voice(self):
        """Voice messages (with MSC3245 field) should be MessageType.VOICE."""
        room = _make_room()
        event = _make_audio_event(is_voice=True)
        
        # Capture the MessageEvent passed to handle_message
        captured_event = None
        
        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event
        
        self.adapter.handle_message = capture
        
        await self.adapter._on_room_message_media(room, event)
        
        assert captured_event is not None, "No event was captured"
        assert captured_event.message_type == MessageType.VOICE, \
            f"Expected MessageType.VOICE, got {captured_event.message_type}"

    @pytest.mark.asyncio
    async def test_voice_message_has_local_path(self):
        """Voice messages should have a local cached path in media_urls."""
        room = _make_room()
        event = _make_audio_event(is_voice=True)
        
        captured_event = None
        
        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event
        
        self.adapter.handle_message = capture
        
        await self.adapter._on_room_message_media(room, event)
        
        assert captured_event is not None
        assert captured_event.media_urls is not None
        assert len(captured_event.media_urls) > 0
        # Should be a local path, not an HTTP URL
        assert not captured_event.media_urls[0].startswith("http"), \
            f"media_urls should contain local path, got {captured_event.media_urls[0]}"
        self.adapter._client.download.assert_awaited_once_with(mxc=event.url)
        assert captured_event.media_types == ["audio/ogg"]

    @pytest.mark.asyncio
    async def test_audio_without_msc3245_stays_audio_type(self):
        """Regular audio uploads (no MSC3245 field) should remain MessageType.AUDIO."""
        room = _make_room()
        event = _make_audio_event(is_voice=False)  # NOT a voice message
        
        captured_event = None
        
        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event
        
        self.adapter.handle_message = capture
        
        await self.adapter._on_room_message_media(room, event)
        
        assert captured_event is not None
        assert captured_event.message_type == MessageType.AUDIO, \
            f"Expected MessageType.AUDIO for non-voice, got {captured_event.message_type}"

    @pytest.mark.asyncio
    async def test_regular_audio_has_http_url(self):
        """Regular audio uploads should keep HTTP URL (not cached locally)."""
        room = _make_room()
        event = _make_audio_event(is_voice=False)
        
        captured_event = None
        
        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event
        
        self.adapter.handle_message = capture
        
        await self.adapter._on_room_message_media(room, event)
        
        assert captured_event is not None
        assert captured_event.media_urls is not None
        # Should be HTTP URL, not local path
        assert captured_event.media_urls[0].startswith("http"), \
            f"Non-voice audio should have HTTP URL, got {captured_event.media_urls[0]}"
        self.adapter._client.download.assert_not_awaited()
        assert captured_event.media_types == ["audio/ogg"]


class TestMatrixVoiceCacheFallback:
    """Test graceful fallback when voice caching fails."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        self.adapter._startup_ts = 0.0
        self.adapter._dm_rooms = {}
        self.adapter._message_handler = AsyncMock()
        self.adapter._mxc_to_http = lambda url: f"https://matrix.example.org/_matrix/media/v3/download/{url[6:]}"
        self.adapter._client = MagicMock()

    @pytest.mark.asyncio
    async def test_voice_cache_failure_falls_back_to_http_url(self):
        """If caching fails, voice message should still be delivered with HTTP URL."""
        room = _make_room()
        event = _make_audio_event(is_voice=True)
        
        # Make download fail
        import nio
        error_resp = MagicMock()
        error_resp.__class__ = nio.DownloadError
        self.adapter._client.download = AsyncMock(return_value=error_resp)
        
        captured_event = None
        
        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event
        
        self.adapter.handle_message = capture
        
        await self.adapter._on_room_message_media(room, event)
        
        assert captured_event is not None
        assert captured_event.media_urls is not None
        # Should fall back to HTTP URL
        assert captured_event.media_urls[0].startswith("http"), \
            f"Should fall back to HTTP URL on cache failure, got {captured_event.media_urls[0]}"

    @pytest.mark.asyncio
    async def test_voice_cache_exception_falls_back_to_http_url(self):
        """Unexpected download exceptions should also fall back to HTTP URL."""
        room = _make_room()
        event = _make_audio_event(is_voice=True)

        self.adapter._client.download = AsyncMock(side_effect=RuntimeError("boom"))

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message_media(room, event)

        assert captured_event is not None
        assert captured_event.media_urls is not None
        assert captured_event.media_urls[0].startswith("http"), \
            f"Should fall back to HTTP URL on exception, got {captured_event.media_urls[0]}"


# ---------------------------------------------------------------------------
# Tests: send_voice includes MSC3245 field (RED -> GREEN)
# ---------------------------------------------------------------------------

class TestMatrixSendVoiceMSC3245:
    """Test that send_voice includes MSC3245 field for native voice rendering."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        # Mock client with successful upload
        self.adapter._client = MagicMock()
        self.upload_call = None

        async def mock_upload(*args, **kwargs):
            self.upload_call = (args, kwargs)
            import nio
            resp = MagicMock()
            resp.content_uri = "mxc://example.org/uploaded"
            resp.__class__ = nio.UploadResponse
            return resp, None

        self.adapter._client.upload = mock_upload

    @pytest.mark.asyncio
    async def test_send_voice_includes_msc3245_field(self):
        """send_voice should include org.matrix.msc3245.voice in message content."""
        import tempfile
        import os
        
        # Create a temp audio file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name
        
        try:
            # Capture the message content sent to room_send
            sent_content = None
            
            async def mock_room_send(room_id, event_type, content):
                nonlocal sent_content
                sent_content = content
                resp = MagicMock()
                resp.event_id = "$sent_event"
                import nio
                resp.__class__ = nio.RoomSendResponse
                return resp
            
            self.adapter._client.room_send = mock_room_send
            
            await self.adapter.send_voice(
                chat_id="!room:example.org",
                audio_path=temp_path,
                caption="Test voice",
            )
            
            assert sent_content is not None, "No message was sent"
            assert "org.matrix.msc3245.voice" in sent_content, \
                f"MSC3245 voice field missing from content: {sent_content.keys()}"
            assert sent_content["msgtype"] == "m.audio"
            assert sent_content["info"]["mimetype"] == "audio/ogg"
            assert self.upload_call is not None, "Expected upload() to be called"
            args, kwargs = self.upload_call
            assert isinstance(args[0], io.BytesIO)
            assert kwargs["content_type"] == "audio/ogg"
            assert kwargs["filename"].endswith(".ogg")

        finally:
            os.unlink(temp_path)
