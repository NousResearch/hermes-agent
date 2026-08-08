"""Regression tests for image replies during a pending clarify (#81117).

When a platform sends an image (image-only or image+caption) while the agent
is blocked in a pending ``clarify``, the gateway must preserve the image
attachment alongside the text reply instead of silently dropping it.

The bug: ``_prepare_clarify_reply_text`` returns ``""`` for image-only events
(no text, no audio). The clarify interception block sees an empty reply and
falls through to the busy-session path, where the image is treated as a media
placeholder interrupt and the actual attachment is lost. For image+caption
events, the caption text resolved the clarify but the image was still dropped
because the normal media routing block was never reached.

The fix mirrors the existing voice-clarify pattern: when a pending clarify
event carries image media, preserve those paths into the session's
``native_image_paths`` (the same mechanism used by the normal media routing
block) so the agent's next turn includes the image natively.
"""
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _event_media_is_image
from gateway.session import SessionSource

try:
    from gateway.config import Platform
except Exception:
    Platform = SimpleNamespace(FEISHU=MagicMock(value="feishu"), TELEGRAM=MagicMock(value="telegram"))


def _source(platform_val=None):
    if platform_val is None:
        platform_val = Platform.FEISHU
    return SessionSource(
        platform=platform_val,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


def _runner():
    """Build a minimal GatewayRunner for testing the clarify interception."""
    runner = object.__new__(GatewayRunner)
    runner._consume_pending_native_image_paths = MagicMock(return_value=[])
    runner._session_key_for_source = lambda _source: "feishu:dm:chat-1"
    runner._adapter_for_source = lambda _source: None
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._reply_anchor_for_event = lambda _event: None
    runner._is_session_running = lambda _key: False
    runner._get_unauthorized_dm_behavior = lambda _plat, profile=None: "reject"
    runner._pairing_store_for = lambda _source: None
    return runner


def _image_event(text="", media_urls=None, media_types=None):
    """Build a MessageEvent that represents a Feishu image message."""
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=_source(),
        message_id="msg-img-1",
        media_urls=media_urls or ["/tmp/feishu_image_001.png"],
        media_types=media_types or ["image/png"],
    )


def _text_event(text="hello"):
    """Build a plain-text MessageEvent."""
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="msg-text-1",
    )


# ---------------------------------------------------------------------------
# _pending_event_image_paths
# ---------------------------------------------------------------------------

class TestPendingEventImagePaths:
    """The new helper that extracts image paths from a pending event."""

    def test_returns_image_paths_for_photo_event(self):
        runner = _runner()
        event = _image_event(
            media_urls=["/tmp/img1.png", "/tmp/img2.jpg"],
            media_types=["image/png", "image/jpeg"],
        )
        paths = runner._pending_event_image_paths(event)
        assert paths == ["/tmp/img1.png", "/tmp/img2.jpg"]

    def test_returns_empty_for_text_only_event(self):
        runner = _runner()
        event = _text_event("hello")
        paths = runner._pending_event_image_paths(event)
        assert paths == []

    def test_returns_empty_for_audio_event(self):
        runner = _runner()
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=_source(),
            message_id="msg-voice",
            media_urls=["/tmp/voice.ogg"],
            media_types=["audio/ogg"],
        )
        paths = runner._pending_event_image_paths(event)
        assert paths == []

    def test_mixed_image_and_audio_returns_only_images(self):
        runner = _runner()
        event = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=_source(),
            message_id="msg-mixed",
            media_urls=["/tmp/img.png", "/tmp/voice.ogg"],
            media_types=["image/png", "audio/ogg"],
        )
        paths = runner._pending_event_image_paths(event)
        assert paths == ["/tmp/img.png"]

    def test_photo_without_mime_uses_message_type(self):
        """A PHOTO event with empty media_types should still classify as image."""
        runner = _runner()
        event = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=_source(),
            message_id="msg-photo-no-mime",
            media_urls=["/tmp/photo.png"],
            media_types=[""],
        )
        paths = runner._pending_event_image_paths(event)
        assert paths == ["/tmp/photo.png"]


# ---------------------------------------------------------------------------
# Clarify interception with images
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClarifyImageReply:
    """When a clarify is pending and the user sends an image, the image must
    be preserved and the clarify resolved."""

    def _setup_pending_clarify(self, runner):
        """Patch clarify_gateway so a pending clarify is visible."""
        mock_entry = SimpleNamespace(clarify_id="clarify-1")
        return mock_entry

    async def test_image_only_preserves_native_paths_and_resolves(self):
        """An image-only message during pending clarify must:
        1. Preserve image paths to native_image_paths
        2. Resolve the clarify (not leave it hanging)
        """
        runner = _runner()
        runner._session_state = MagicMock()
        mock_state = MagicMock()
        mock_state.persistent.native_image_paths = []
        runner._session_state.return_value = mock_state

        event = _image_event()

        resolve_calls = []

        def _fake_resolve(session_key, response):
            resolve_calls.append((session_key, response))
            return True

        mock_clarify = self._setup_pending_clarify(runner)

        # Pre-import so patch can find the attribute
        import tools.clarify_gateway  # noqa: F401
        with patch("tools.clarify_gateway") as mock_cm, \
             patch.object(runner, "_adapter_for_source", return_value=None):
            mock_cm.get_pending_for_session.return_value = mock_clarify
            mock_cm.resolve_text_response_for_session.side_effect = _fake_resolve

            result = await runner._handle_message(event)

        # Clarify was resolved
        assert len(resolve_calls) == 1
        session_key, reply_text = resolve_calls[0]
        assert session_key == "feishu:dm:chat-1"

        # Image paths were preserved for the agent's next turn
        assert mock_state.persistent.native_image_paths == ["/tmp/feishu_image_001.png"]

        # Reply text is not empty — contains a media indicator
        assert reply_text  # not empty

    async def test_image_plus_caption_preserves_image_and_uses_caption(self):
        """An image+caption message during pending clarify must:
        1. Use the caption text as the clarify response
        2. Preserve the image attachment
        """
        runner = _runner()
        runner._session_state = MagicMock()
        mock_state = MagicMock()
        mock_state.persistent.native_image_paths = []
        runner._session_state.return_value = mock_state

        event = _image_event(
            text="Here is the screenshot",
            media_urls=["/tmp/screenshot.png"],
            media_types=["image/png"],
        )

        resolve_calls = []

        def _fake_resolve(session_key, response):
            resolve_calls.append((session_key, response))
            return True

        mock_clarify = self._setup_pending_clarify(runner)

        import tools.clarify_gateway  # noqa: F401
        with patch("tools.clarify_gateway") as mock_cm, \
             patch.object(runner, "_adapter_for_source", return_value=None):
            mock_cm.get_pending_for_session.return_value = mock_clarify
            mock_cm.resolve_text_response_for_session.side_effect = _fake_resolve

            await runner._handle_message(event)

        # Clarify was resolved with the caption text
        assert len(resolve_calls) == 1
        _, reply_text = resolve_calls[0]
        assert "screenshot" in reply_text.lower()

        # Image paths were preserved
        assert mock_state.persistent.native_image_paths == ["/tmp/screenshot.png"]

    async def test_text_only_clarify_does_not_set_native_image_paths(self):
        """A plain text clarify response must NOT set native_image_paths."""
        runner = _runner()
        runner._session_state = MagicMock()
        mock_state = MagicMock()
        mock_state.persistent.native_image_paths = []
        runner._session_state.return_value = mock_state

        event = _text_event("my answer")

        resolve_calls = []

        def _fake_resolve(session_key, response):
            resolve_calls.append((session_key, response))
            return True

        mock_clarify = self._setup_pending_clarify(runner)

        import tools.clarify_gateway  # noqa: F401
        with patch("tools.clarify_gateway") as mock_cm, \
             patch.object(runner, "_adapter_for_source", return_value=None):
            mock_cm.get_pending_for_session.return_value = mock_clarify
            mock_cm.resolve_text_response_for_session.side_effect = _fake_resolve

            await runner._handle_message(event)

        assert len(resolve_calls) == 1
        _, reply_text = resolve_calls[0]
        assert reply_text == "my answer"

        # No image paths should be set
        assert mock_state.persistent.native_image_paths == []

    async def test_image_only_does_not_resolve_with_empty_text(self):
        """An image-only clarify reply must NOT resolve with an empty string
        that would confuse the agent. It should use a descriptive placeholder."""
        runner = _runner()
        runner._session_state = MagicMock()
        mock_state = MagicMock()
        mock_state.persistent.native_image_paths = []
        runner._session_state.return_value = mock_state

        event = _image_event(text="", media_urls=["/tmp/img.png"])

        resolve_calls = []

        def _fake_resolve(session_key, response):
            resolve_calls.append((session_key, response))
            return True

        mock_clarify = self._setup_pending_clarify(runner)

        import tools.clarify_gateway  # noqa: F401
        with patch("tools.clarify_gateway") as mock_cm, \
             patch.object(runner, "_adapter_for_source", return_value=None):
            mock_cm.get_pending_for_session.return_value = mock_clarify
            mock_cm.resolve_text_response_for_session.side_effect = _fake_resolve

            await runner._handle_message(event)

        assert len(resolve_calls) == 1
        _, reply_text = resolve_calls[0]
        # Must NOT be empty
        assert reply_text, "Image-only clarify reply must not be empty"
        # Should contain a reference to the image
        assert "image" in reply_text.lower() or "img" in reply_text.lower()
