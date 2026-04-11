"""Tests for Matrix voice message support (MSC3245).

Updated for the mautrix-python SDK (no more matrix-nio / nio imports).
"""
import io
import os
import tempfile
import types
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Try importing mautrix; skip entire file if not available.
try:
    import mautrix as _mautrix_probe
    if not isinstance(_mautrix_probe, types.ModuleType) or not hasattr(_mautrix_probe, "__file__"):
        pytest.skip("mautrix in sys.modules is a mock, not the real package", allow_module_level=True)
except ImportError:
    pytest.skip("mautrix not installed", allow_module_level=True)

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


def _make_audio_event(
    event_id: str = "$audio_event",
    sender: str = "@alice:example.org",
    room_id: str = "!test:example.org",
    body: str = "Voice message",
    url: str = "mxc://example.org/abc123",
    is_voice: bool = False,
    mimetype: str = "audio/ogg",
    timestamp: int = 9999999999000,  # ms
):
    """
    Create a mock mautrix room message event.

    In mautrix, the handler receives a single event object with attributes
    ``room_id``, ``sender``, ``event_id``, ``timestamp``, and ``content``
    (a dict-like or serializable object).

    Args:
        is_voice: If True, adds org.matrix.msc3245.voice field to content.
    """
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

    event = SimpleNamespace(
        event_id=event_id,
        sender=sender,
        room_id=room_id,
        timestamp=timestamp,
        content=content,
    )
    return event


def _make_state_store(member_count: int = 2):
    """Create a mock state store with get_members/get_member support."""
    store = MagicMock()
    # get_members returns a list of member user IDs
    members = [MagicMock() for _ in range(member_count)]
    store.get_members = AsyncMock(return_value=members)
    # get_member returns a single member info object
    member = MagicMock()
    member.displayname = "Alice"
    store.get_member = AsyncMock(return_value=member)
    return store


# ---------------------------------------------------------------------------
# Tests: MSC3245 Voice Detection
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
        # Mock client for authenticated download — download_media returns bytes directly
        self.adapter._client = MagicMock()
        self.adapter._client.download_media = AsyncMock(return_value=b"fake audio data")
        # State store for DM detection
        self.adapter._client.state_store = _make_state_store()

    @pytest.mark.asyncio
    async def test_voice_message_has_type_voice(self):
        """Voice messages (with MSC3245 field) should be MessageType.VOICE."""
        event = _make_audio_event(is_voice=True)

        # Capture the MessageEvent passed to handle_message
        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None, "No event was captured"
        assert captured_event.message_type == MessageType.VOICE, \
            f"Expected MessageType.VOICE, got {captured_event.message_type}"

    @pytest.mark.asyncio
    async def test_voice_message_has_local_path(self):
        """Voice messages should have a local cached path in media_urls."""
        event = _make_audio_event(is_voice=True)

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None
        assert captured_event.media_urls is not None
        assert len(captured_event.media_urls) > 0
        # Should be a local path, not an HTTP URL
        assert not captured_event.media_urls[0].startswith("http"), \
            f"media_urls should contain local path, got {captured_event.media_urls[0]}"
        # download_media is called with a ContentURI wrapping the mxc URL
        self.adapter._client.download_media.assert_awaited_once()
        assert captured_event.media_types == ["audio/ogg"]

    @pytest.mark.asyncio
    async def test_audio_without_msc3245_stays_audio_type(self):
        """Regular audio uploads (no MSC3245 field) should remain MessageType.AUDIO."""
        event = _make_audio_event(is_voice=False)  # NOT a voice message

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None
        assert captured_event.message_type == MessageType.AUDIO, \
            f"Expected MessageType.AUDIO for non-voice, got {captured_event.message_type}"

    @pytest.mark.asyncio
    async def test_regular_audio_has_http_url(self):
        """Regular audio uploads should keep HTTP URL (not cached locally)."""
        event = _make_audio_event(is_voice=False)

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None
        assert captured_event.media_urls is not None
        # Should be HTTP URL, not local path
        assert captured_event.media_urls[0].startswith("http"), \
            f"Non-voice audio should have HTTP URL, got {captured_event.media_urls[0]}"
        self.adapter._client.download_media.assert_not_awaited()
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
        self.adapter._client.state_store = _make_state_store()

    @pytest.mark.asyncio
    async def test_voice_cache_failure_falls_back_to_http_url(self):
        """If caching fails (download returns None), voice message should still be delivered with HTTP URL."""
        event = _make_audio_event(is_voice=True)

        # download_media returns None on failure
        self.adapter._client.download_media = AsyncMock(return_value=None)

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None
        assert captured_event.media_urls is not None
        # Should fall back to HTTP URL
        assert captured_event.media_urls[0].startswith("http"), \
            f"Should fall back to HTTP URL on cache failure, got {captured_event.media_urls[0]}"

    @pytest.mark.asyncio
    async def test_voice_cache_exception_falls_back_to_http_url(self):
        """Unexpected download exceptions should also fall back to HTTP URL."""
        event = _make_audio_event(is_voice=True)

        self.adapter._client.download_media = AsyncMock(side_effect=RuntimeError("boom"))

        captured_event = None

        async def capture(msg_event):
            nonlocal captured_event
            captured_event = msg_event

        self.adapter.handle_message = capture

        await self.adapter._on_room_message(event)

        assert captured_event is not None
        assert captured_event.media_urls is not None
        assert captured_event.media_urls[0].startswith("http"), \
            f"Should fall back to HTTP URL on exception, got {captured_event.media_urls[0]}"


# ---------------------------------------------------------------------------
# Tests: send_voice includes MSC3245 field
# ---------------------------------------------------------------------------

class TestMatrixSendVoiceMSC3245:
    """Test that send_voice includes MSC3245 field for native voice rendering."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._user_id = "@bot:example.org"
        # Mock client — upload_media returns a ContentURI string
        self.adapter._client = MagicMock()
        self.upload_call = None

        async def mock_upload_media(data, mime_type=None, filename=None, **kwargs):
            self.upload_call = {"data": data, "mime_type": mime_type, "filename": filename}
            return "mxc://example.org/uploaded"

        self.adapter._client.upload_media = mock_upload_media

    @pytest.mark.asyncio
    @patch("mimetypes.guess_type", return_value=("audio/ogg", None))
    async def test_send_voice_includes_msc3245_field(self, _mock_guess):
        """send_voice should include org.matrix.msc3245.voice in message content."""
        # Create a temp audio file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(b"fake audio data")
            temp_path = f.name

        try:
            # Capture the message content sent via send_message_event
            sent_content = None

            async def mock_send_message_event(room_id, event_type, content):
                nonlocal sent_content
                sent_content = content
                # send_message_event returns an EventID string
                return "$sent_event"

            self.adapter._client.send_message_event = mock_send_message_event

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
            assert self.upload_call is not None, "Expected upload_media() to be called"
            assert isinstance(self.upload_call["data"], bytes)
            assert self.upload_call["mime_type"] == "audio/ogg"
            assert self.upload_call["filename"].endswith(".ogg")

        finally:
            os.unlink(temp_path)
