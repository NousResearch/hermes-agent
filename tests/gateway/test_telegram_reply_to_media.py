"""Tests for Telegram reply-to-media context hydration.

When a user replies to a *media* message (PDF / photo / voice / video /
sticker), Telegram delivers the metadata on ``msg.reply_to_message`` but
does NOT re-stream the bytes.  Two layers of behavior are verified:

1. Sync description (`_describe_reply_to_message`) — always produces a
   non-empty string for non-empty replies, even when there's no
   text/caption to fall back on.

2. Async hydration (`_hydrate_reply_to_media`) — re-fetches the bytes
   via Telegram's getFile API and appends to ``event.media_urls`` /
   ``event.media_types`` using the same cache helpers the inbound
   media path uses.  Failures are swallowed (non-fatal).
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_telegram_mock():
    """Mock the telegram package for environments without it installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.base import MessageEvent, MessageType  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _attrs(**kw):
    """Build a SimpleNamespace whose missing attrs return None on getattr.

    SimpleNamespace alone raises AttributeError; we need ``getattr(..., default)``
    semantics so the helpers' ``getattr(rtm, 'photo', None)`` lookups work
    naturally on a minimal stub.
    """
    base = {
        "text": None,
        "caption": None,
        "document": None,
        "photo": None,
        "voice": None,
        "audio": None,
        "video": None,
        "video_note": None,
        "sticker": None,
        "animation": None,
        "location": None,
        "contact": None,
    }
    base.update(kw)
    return SimpleNamespace(**base)


# ── _describe_reply_to_message ────────────────────────────────────────────

class TestDescribeReplyTo:
    def test_text_returned_verbatim(self, adapter):
        rtm = _attrs(text="hello world")
        assert adapter._describe_reply_to_message(rtm) == "hello world"

    def test_caption_used_when_no_text(self, adapter):
        rtm = _attrs(caption="caption only")
        assert adapter._describe_reply_to_message(rtm) == "caption only"

    def test_document_only_returns_filename_tag(self, adapter):
        doc = SimpleNamespace(file_name="quarterly_report.pdf", mime_type="application/pdf")
        rtm = _attrs(document=doc)
        out = adapter._describe_reply_to_message(rtm)
        assert "quarterly_report.pdf" in out
        assert "application/pdf" in out
        assert out.startswith("[file:")

    def test_document_with_caption_combines_both(self, adapter):
        doc = SimpleNamespace(file_name="report.pdf", mime_type="application/pdf")
        rtm = _attrs(document=doc, caption="Q3 numbers")
        out = adapter._describe_reply_to_message(rtm)
        assert "report.pdf" in out
        assert "Q3 numbers" in out

    def test_document_without_filename_falls_back(self, adapter):
        doc = SimpleNamespace(file_name=None, mime_type="application/pdf")
        rtm = _attrs(document=doc)
        out = adapter._describe_reply_to_message(rtm)
        assert "[file:" in out

    def test_photo_only(self, adapter):
        rtm = _attrs(photo=[SimpleNamespace()])
        assert adapter._describe_reply_to_message(rtm) == "[photo]"

    def test_voice_with_duration(self, adapter):
        rtm = _attrs(voice=SimpleNamespace(duration=7))
        assert adapter._describe_reply_to_message(rtm) == "[voice 7s]"

    def test_voice_without_duration(self, adapter):
        rtm = _attrs(voice=SimpleNamespace(duration=None))
        assert adapter._describe_reply_to_message(rtm) == "[voice]"

    def test_audio_with_title(self, adapter):
        rtm = _attrs(audio=SimpleNamespace(title="Song", file_name=None))
        assert adapter._describe_reply_to_message(rtm) == "[audio: Song]"

    def test_audio_with_filename_only(self, adapter):
        rtm = _attrs(audio=SimpleNamespace(title=None, file_name="track.mp3"))
        assert adapter._describe_reply_to_message(rtm) == "[audio: track.mp3]"

    def test_video_only(self, adapter):
        rtm = _attrs(video=SimpleNamespace())
        assert adapter._describe_reply_to_message(rtm) == "[video]"

    def test_sticker_with_emoji(self, adapter):
        rtm = _attrs(sticker=SimpleNamespace(emoji="🐱"))
        assert adapter._describe_reply_to_message(rtm) == "[sticker 🐱]"

    def test_sticker_without_emoji(self, adapter):
        rtm = _attrs(sticker=SimpleNamespace(emoji=None))
        assert adapter._describe_reply_to_message(rtm) == "[sticker]"

    def test_animation_only(self, adapter):
        rtm = _attrs(animation=SimpleNamespace())
        assert adapter._describe_reply_to_message(rtm) == "[gif]"

    def test_location_only(self, adapter):
        rtm = _attrs(location=SimpleNamespace())
        assert adapter._describe_reply_to_message(rtm) == "[location]"

    def test_empty_returns_none(self, adapter):
        rtm = _attrs()
        assert adapter._describe_reply_to_message(rtm) is None


# ── _hydrate_reply_to_media ───────────────────────────────────────────────

@pytest.fixture()
def empty_event():
    return MessageEvent(text="reply text", message_type=MessageType.TEXT)


def _mk_file(file_path: str = "documents/file_0.pdf", payload: bytes = b"PDFDATA"):
    """Build a mock telegram File whose download returns *payload*."""
    file_obj = MagicMock()
    file_obj.file_path = file_path
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(payload))
    return file_obj


class TestHydrateReplyToMedia:

    @pytest.mark.asyncio
    async def test_no_reply_is_noop(self, adapter, empty_event):
        msg = SimpleNamespace(reply_to_message=None)
        await adapter._hydrate_reply_to_media(empty_event, msg)
        assert empty_event.media_urls == []
        assert empty_event.media_types == []

    @pytest.mark.asyncio
    async def test_photo_reply_caches_bytes(self, adapter, empty_event, tmp_path):
        file_obj = _mk_file(file_path="photos/file_0.png", payload=b"PNGDATA")
        photo = SimpleNamespace()
        photo.get_file = AsyncMock(return_value=file_obj)
        rtm = _attrs(photo=[photo])
        msg = SimpleNamespace(reply_to_message=rtm)

        with patch(
            "gateway.platforms.telegram.cache_image_from_bytes",
            return_value=str(tmp_path / "img.png"),
        ) as mock_cache:
            await adapter._hydrate_reply_to_media(empty_event, msg)

        mock_cache.assert_called_once_with(b"PNGDATA", ext=".png")
        assert empty_event.media_urls == [str(tmp_path / "img.png")]
        assert empty_event.media_types == ["image/png"]

    @pytest.mark.asyncio
    async def test_voice_reply_caches_audio(self, adapter, empty_event, tmp_path):
        file_obj = _mk_file(payload=b"OGG")
        voice = SimpleNamespace()
        voice.get_file = AsyncMock(return_value=file_obj)
        rtm = _attrs(voice=voice)
        msg = SimpleNamespace(reply_to_message=rtm)

        with patch(
            "gateway.platforms.telegram.cache_audio_from_bytes",
            return_value=str(tmp_path / "v.ogg"),
        ) as mock_cache:
            await adapter._hydrate_reply_to_media(empty_event, msg)

        mock_cache.assert_called_once_with(b"OGG", ext=".ogg")
        assert empty_event.media_urls == [str(tmp_path / "v.ogg")]
        assert empty_event.media_types == ["audio/ogg"]

    @pytest.mark.asyncio
    async def test_document_reply_caches_pdf(self, adapter, empty_event, tmp_path):
        file_obj = _mk_file(payload=b"%PDF-1.7")
        doc = SimpleNamespace(
            file_name="quarterly_report.pdf",
            mime_type="application/pdf",
        )
        doc.get_file = AsyncMock(return_value=file_obj)
        rtm = _attrs(document=doc)
        msg = SimpleNamespace(reply_to_message=rtm)

        with patch(
            "gateway.platforms.telegram.cache_document_from_bytes",
            return_value=str(tmp_path / "doc_abc_quarterly_report.pdf"),
        ) as mock_cache:
            await adapter._hydrate_reply_to_media(empty_event, msg)

        mock_cache.assert_called_once_with(b"%PDF-1.7", "quarterly_report.pdf")
        assert empty_event.media_urls == [
            str(tmp_path / "doc_abc_quarterly_report.pdf")
        ]
        assert empty_event.media_types == ["application/pdf"]

    @pytest.mark.asyncio
    async def test_image_document_reply_routes_to_image_cache(
        self, adapter, empty_event, tmp_path
    ):
        """A PNG sent as 'file' should land in the image cache, not the doc cache."""
        file_obj = _mk_file(payload=b"PNG")
        doc = SimpleNamespace(file_name="screenshot.png", mime_type="image/png")
        doc.get_file = AsyncMock(return_value=file_obj)
        rtm = _attrs(document=doc)
        msg = SimpleNamespace(reply_to_message=rtm)

        with patch(
            "gateway.platforms.telegram.cache_image_from_bytes",
            return_value=str(tmp_path / "img.png"),
        ) as mock_img, patch(
            "gateway.platforms.telegram.cache_document_from_bytes"
        ) as mock_doc:
            await adapter._hydrate_reply_to_media(empty_event, msg)

        mock_img.assert_called_once()
        mock_doc.assert_not_called()
        assert empty_event.media_urls == [str(tmp_path / "img.png")]
        assert empty_event.media_types == ["image/png"]

    @pytest.mark.asyncio
    async def test_fetch_failure_is_non_fatal(self, adapter, empty_event):
        """A getFile() that raises must NOT crash the message handler."""
        doc = SimpleNamespace(file_name="report.pdf", mime_type="application/pdf")
        doc.get_file = AsyncMock(side_effect=RuntimeError("network down"))
        rtm = _attrs(document=doc)
        msg = SimpleNamespace(reply_to_message=rtm)

        # Must not raise
        await adapter._hydrate_reply_to_media(empty_event, msg)
        assert empty_event.media_urls == []
        assert empty_event.media_types == []

    @pytest.mark.asyncio
    async def test_appends_rather_than_replaces(self, adapter, tmp_path):
        """Hydrator must preserve any media already on the event (e.g. inbound)."""
        event = MessageEvent(
            text="reply",
            message_type=MessageType.TEXT,
            media_urls=["/tmp/inbound.jpg"],
            media_types=["image/jpeg"],
        )
        file_obj = _mk_file(payload=b"PDF")
        doc = SimpleNamespace(file_name="quoted.pdf", mime_type="application/pdf")
        doc.get_file = AsyncMock(return_value=file_obj)
        rtm = _attrs(document=doc)
        msg = SimpleNamespace(reply_to_message=rtm)

        with patch(
            "gateway.platforms.telegram.cache_document_from_bytes",
            return_value=str(tmp_path / "quoted_cached.pdf"),
        ):
            await adapter._hydrate_reply_to_media(event, msg)

        assert event.media_urls == [
            "/tmp/inbound.jpg",
            str(tmp_path / "quoted_cached.pdf"),
        ]
        assert event.media_types == ["image/jpeg", "application/pdf"]
