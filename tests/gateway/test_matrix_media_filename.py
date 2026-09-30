"""Tests for Matrix adapter media-filename blanking.

Matrix ``m.image`` events were already handled (PR #16821, issue #13482).
These tests verify that ``m.audio``, ``m.file``, and ``m.video`` events
also have their transport filename stripped from ``body`` when no caption
is present, so the filename doesn't leak into ``event.text`` and get fed
to the model as user message text.

They also check the name under which the adapter caches an attachment.
"""

import re
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.matrix.media_content import (
    _looks_like_matrix_image_filename,
    _looks_like_matrix_media_filename,
)


class TestLooksLikeMatrixMediaFilename:
    """Unit tests for _looks_like_matrix_media_filename()."""

    @pytest.mark.parametrize(
        "text",
        [
            "voice-20260725-114233.ogg",
            "recording.wav",
            "clip.opus",
            "audio.amr",
            "data.m4a",
            "clip.webm",
            "video.mkv",
            "video.mov",
            "CaPtIoN.mp3",
        ],
    )
    def test_bare_media_filenames_detected(self, text):
        assert _looks_like_matrix_media_filename(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "remind me to call the dentist",
            "voice note.m4a",
            "document.pdf",
            "file.txt",
            "noextension",
            "data.csv",
            "",
            "  ",
            None,
        ],
    )
    def test_non_filenames_rejected(self, text):
        assert _looks_like_matrix_media_filename(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "/tmp/voice.ogg",
            "./audio.mp3",
        ],
    )
    def test_paths_with_separators_rejected(self, text):
        assert _looks_like_matrix_media_filename(text) is False

    def test_trailing_whitespace_stripped(self):
        assert _looks_like_matrix_media_filename("audio.ogg\n") is True
        assert _looks_like_matrix_media_filename("  photo.mp3  ") is True

    def test_multi_token_filename_rejected(self):
        """A filename with a space is treated as a caption, not a bare token."""
        assert _looks_like_matrix_media_filename("my recording.ogg") is False

    def test_newline_in_text_rejected(self):
        """Multi-line text is never a bare filename."""
        assert _looks_like_matrix_media_filename("voice.ogg\nmore text") is False

    def test_image_extension_not_detected_by_media_function(self):
        """Image extensions are handled by the image function, not the media function."""
        assert _looks_like_matrix_media_filename("photo.jpg") is False
        assert _looks_like_matrix_image_filename("photo.jpg") is True

    def test_audio_extension_not_detected_by_image_function(self):
        """Audio extensions are handled by the media function, not the image function."""
        assert _looks_like_matrix_image_filename("voice.ogg") is False
        assert _looks_like_matrix_media_filename("voice.ogg") is True

    def test_mimetypes_audio_detected(self):
        """mimetypes.guess_type catches audio types not in the hardcoded set."""
        # .aiff is audio/x-aiff, NOT in _MATRIX_MEDIA_FILENAME_EXTS
        assert _looks_like_matrix_media_filename("song.aiff") is True

    def test_mimetypes_video_detected(self):
        """mimetypes.guess_type catches video types."""
        assert _looks_like_matrix_media_filename("clip.avi") is True


def _make_adapter(monkeypatch, payload: bytes):
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter, MatrixRoomIdentity
    from tests.gateway.matrix_helpers import FakeMediaDownload

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org",
               "require_mention": False, "auto_thread": False, "dm_auto_thread": False}))
    adapter._startup_ts = time.time() - 10
    adapter._text_batch_delay_seconds = 0
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._client = MagicMock()
    FakeMediaDownload(payload).install(adapter._client)
    adapter._resolve_room_identity = AsyncMock(return_value=MatrixRoomIdentity(
        room_id="!dm:example.org", room_name=None, room_topic=None, canonical_alias=None,
        server_name="example.org", joined_member_count=2, room_state=None,
        is_direct_account_data=True, display_name="Alice", has_explicit_name=False, chat_type="dm"))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    return adapter


_REPLY = "> <@bob:example.org> what did you think?\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("msgtype, mimetype, content, cached_name", [
    ("m.file", "application/pdf", {"filename": "report.pdf", "body": "please review"}, "doc_report.pdf"),
    ("m.file", "application/pdf", {"body": "report.pdf"}, "doc_report.pdf"),
    ("m.file", "application/pdf", {"body": "Q3 Report.pdf"}, "doc_Q3 Report.pdf"),
    ("m.file", "application/pdf", {"body": _REPLY + "report.pdf"}, "doc_report.pdf"),
    ("m.file", "application/pdf", {"body": "minutes from\nMonday"}, "doc_document.pdf"),
    ("m.file", "application/pdf", {"body": "see https://example.org/notes.pdf"}, "doc_document.pdf"),
    ("m.file", "application/pdf", {"filename": "..\\..\\evil.pdf", "body": "evil.pdf"}, "doc_evil.pdf"),
    ("m.video", "video/mp4", {"body": "look at this"}, "doc_video.mp4"),
    ("m.audio", "audio/mpeg", {"body": _REPLY + "listen to this"}, "audio.mp3"),
    ("m.audio", "audio/mpeg", {"filename": "memo.wav", "body": "listen"}, "audio.wav"),
])
async def test_cached_media_name_comes_from_the_filename_not_a_caption(
        monkeypatch, msgtype, mimetype, content, cached_name):
    """The cached file is named after the declared ``filename``, or after the body when the body is a
    filename once any reply fallback is removed. Otherwise it gets a generic name with the extension
    for its MIME type. The file is always written directly inside the cache directory."""
    from gateway.platforms.base import get_audio_cache_dir, get_document_cache_dir

    payload = b"not a recognisable container"
    adapter = _make_adapter(monkeypatch, payload)
    relates_to = {"m.in_reply_to": {"event_id": "$question"}} if content["body"].startswith("> ") else {}
    await adapter._on_room_message(SimpleNamespace(
        sender="@alice:example.org", event_id="$media", room_id="!dm:example.org",
        timestamp=int(time.time() * 1000),
        content={**content, "msgtype": msgtype, "url": "mxc://example.org/abc",
                 "info": {"mimetype": mimetype, "size": len(payload)}, "m.relates_to": relates_to}))

    event = adapter.handle_message.await_args.args[0]
    cached = Path(event.media_urls[0])
    cache_dir = get_audio_cache_dir() if msgtype == "m.audio" else get_document_cache_dir()
    assert (cached.parent, re.sub(r"_[0-9a-f]{12}", "", cached.name, count=1), cached.read_bytes()) == (
        cache_dir, cached_name, payload)
