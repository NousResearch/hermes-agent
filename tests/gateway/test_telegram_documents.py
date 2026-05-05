"""
Tests for Telegram document handling in gateway/platforms/telegram.py.

Covers: document type detection, download/cache flow, size limits,
        text injection, error handling.

Note: python-telegram-bot may not be installed in the test environment.
We mock the telegram module at import time to avoid collection errors.
"""

import asyncio
import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_VIDEO_TYPES,
)


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        # Real library is installed — no mocking needed
        return

    telegram_mod = MagicMock()
    # ContextTypes needs DEFAULT_TYPE as an actual attribute for the annotation
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

# Now we can safely import
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build mock Telegram objects
# ---------------------------------------------------------------------------

def _make_file_obj(data: bytes = b"hello"):
    """Create a mock Telegram File with download_as_bytearray."""
    f = AsyncMock()
    f.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    f.file_path = "documents/file.pdf"
    return f


def _make_document(
    file_name="report.pdf",
    mime_type="application/pdf",
    file_size=1024,
    file_obj=None,
):
    """Create a mock Telegram Document object."""
    doc = MagicMock()
    doc.file_name = file_name
    doc.mime_type = mime_type
    doc.file_size = file_size
    doc.get_file = AsyncMock(return_value=file_obj or _make_file_obj())
    return doc


def _make_message(document=None, caption=None, media_group_id=None, photo=None):
    """Build a mock Telegram Message with the given document/photo."""
    msg = MagicMock()
    msg.message_id = 42
    msg.text = caption or ""
    msg.caption = caption
    msg.date = None
    # Media flags — all None except explicit payload
    msg.photo = photo
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.sticker = None
    msg.document = document
    msg.media_group_id = media_group_id
    # Chat / user
    msg.chat = MagicMock()
    msg.chat.id = 100
    msg.chat.type = "private"
    msg.chat.title = None
    msg.chat.full_name = "Test User"
    msg.from_user = MagicMock()
    msg.from_user.id = 1
    msg.from_user.full_name = "Test User"
    msg.message_thread_id = None
    return msg


def _make_update(msg):
    """Wrap a message in a mock Update."""
    update = MagicMock()
    update.message = msg
    return update


def _make_video(file_obj=None):
    video = MagicMock()
    video.get_file = AsyncMock(return_value=file_obj or _make_file_obj(b"video-bytes"))
    return video


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config)
    # Capture events instead of processing them
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document/video cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )


# ---------------------------------------------------------------------------
# TestDocumentTypeDetection
# ---------------------------------------------------------------------------

class TestDocumentTypeDetection:
    @pytest.mark.asyncio
    async def test_document_detected_explicitly(self, adapter):
        doc = _make_document()
        msg = _make_message(document=doc)
        update = _make_update(msg)
        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT

    @pytest.mark.asyncio
    async def test_fallback_is_document(self, adapter):
        """When no specific media attr is set, message_type defaults to DOCUMENT."""
        msg = _make_message()
        msg.document = None  # no media at all
        update = _make_update(msg)
        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.DOCUMENT


# ---------------------------------------------------------------------------
# TestDocumentDownloadBlock
# ---------------------------------------------------------------------------

def _make_photo(file_obj=None):
    photo = MagicMock()
    photo.get_file = AsyncMock(return_value=file_obj or _make_file_obj(b"photo-bytes"))
    return photo


class TestDocumentDownloadBlock:
    @pytest.mark.asyncio
    async def test_supported_pdf_is_cached(self, adapter):
        pdf_bytes = b"%PDF-1.4 fake"
        file_obj = _make_file_obj(pdf_bytes)
        doc = _make_document(file_name="report.pdf", file_size=1024, file_obj=file_obj)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert event.media_types == ["application/pdf"]

    @pytest.mark.asyncio
    async def test_supported_txt_injects_content(self, adapter):
        content = b"Hello from a text file"
        file_obj = _make_file_obj(content)
        doc = _make_document(
            file_name="notes.txt", mime_type="text/plain",
            file_size=len(content), file_obj=file_obj,
        )
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "Hello from a text file" in event.text
        assert "[Content of notes.txt]" in event.text

    @pytest.mark.asyncio
    async def test_supported_md_injects_content(self, adapter):
        content = b"# Title\nSome markdown"
        file_obj = _make_file_obj(content)
        doc = _make_document(
            file_name="readme.md", mime_type="text/markdown",
            file_size=len(content), file_obj=file_obj,
        )
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "# Title" in event.text

    @pytest.mark.asyncio
    async def test_caption_preserved_with_injection(self, adapter):
        content = b"file text"
        file_obj = _make_file_obj(content)
        doc = _make_document(
            file_name="doc.txt", mime_type="text/plain",
            file_size=len(content), file_obj=file_obj,
        )
        msg = _make_message(document=doc, caption="Please summarize")
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "file text" in event.text
        assert "Please summarize" in event.text

    @pytest.mark.asyncio
    async def test_zip_document_cached(self, adapter):
        """A .zip upload should be cached as a supported document."""
        doc = _make_document(file_name="archive.zip", mime_type="application/zip", file_size=100)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls and event.media_urls[0].endswith("archive.zip")
        assert event.media_types == ["application/zip"]

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(self, adapter):
        doc = _make_document(file_name="huge.pdf", file_size=25 * 1024 * 1024)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "too large" in event.text

    @pytest.mark.asyncio
    async def test_none_file_size_rejected(self, adapter):
        """Security fix: file_size=None must be rejected (not silently allowed)."""
        doc = _make_document(file_name="tricky.pdf", file_size=None)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "too large" in event.text or "could not be verified" in event.text

    @pytest.mark.asyncio
    async def test_missing_filename_uses_mime_lookup(self, adapter):
        """No file_name but valid mime_type should resolve to extension."""
        content = b"some pdf bytes"
        file_obj = _make_file_obj(content)
        doc = _make_document(
            file_name=None, mime_type="application/pdf",
            file_size=len(content), file_obj=file_obj,
        )
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert event.media_types == ["application/pdf"]

    @pytest.mark.asyncio
    async def test_missing_filename_and_mime_rejected(self, adapter):
        doc = _make_document(file_name=None, mime_type=None, file_size=100)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "Unsupported" in event.text

    @pytest.mark.asyncio
    async def test_unicode_decode_error_handled(self, adapter):
        """Binary bytes that aren't valid UTF-8 in a .txt — content not injected but file still cached."""
        binary = bytes(range(128, 256))  # not valid UTF-8
        file_obj = _make_file_obj(binary)
        doc = _make_document(
            file_name="binary.txt", mime_type="text/plain",
            file_size=len(binary), file_obj=file_obj,
        )
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        # File should still be cached
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        # Content NOT injected — text should be empty (no caption set)
        assert "[Content of" not in (event.text or "")

    @pytest.mark.asyncio
    async def test_text_injection_capped(self, adapter):
        """A .txt file over 100 KB should NOT have its content injected."""
        large = b"x" * (200 * 1024)  # 200 KB
        file_obj = _make_file_obj(large)
        doc = _make_document(
            file_name="big.txt", mime_type="text/plain",
            file_size=len(large), file_obj=file_obj,
        )
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        # File should be cached
        assert len(event.media_urls) == 1
        # Content should NOT be injected
        assert "[Content of" not in (event.text or "")

    @pytest.mark.asyncio
    async def test_download_exception_handled(self, adapter):
        """If get_file() raises, the handler logs the error without crashing."""
        doc = _make_document(file_name="crash.pdf", file_size=100)
        doc.get_file = AsyncMock(side_effect=RuntimeError("Telegram API down"))
        msg = _make_message(document=doc)
        update = _make_update(msg)

        # Should not raise
        await adapter._handle_media_message(update, MagicMock())
        # handle_message should still be called (the handler catches the exception)
        adapter.handle_message.assert_called_once()


class TestVideoDownloadBlock:
    @pytest.mark.asyncio
    async def test_native_video_is_cached(self, adapter):
        file_obj = _make_file_obj(b"fake-mp4")
        file_obj.file_path = "videos/clip.mp4"
        msg = _make_message()
        msg.video = _make_video(file_obj)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.VIDEO
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert event.media_types == [SUPPORTED_VIDEO_TYPES[".mp4"]]

    @pytest.mark.asyncio
    async def test_mp4_document_is_treated_as_video(self, adapter):
        file_obj = _make_file_obj(b"fake-mp4-doc")
        doc = _make_document(file_name="good.mp4", mime_type="video/mp4", file_size=1024, file_obj=file_obj)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.VIDEO
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert event.media_types == [SUPPORTED_VIDEO_TYPES[".mp4"]]


# ---------------------------------------------------------------------------
# TestMediaGroups — media group (album) buffering
# ---------------------------------------------------------------------------

class TestMediaGroups:
    @pytest.mark.asyncio
    async def test_non_album_photo_burst_is_buffered_and_combined(self, adapter):
        first_photo = _make_photo(_make_file_obj(b"first"))
        second_photo = _make_photo(_make_file_obj(b"second"))

        msg1 = _make_message(caption="two images", photo=[first_photo])
        msg2 = _make_message(photo=[second_photo])

        with patch("gateway.platforms.telegram.cache_image_from_bytes", side_effect=["/tmp/burst-one.jpg", "/tmp/burst-two.jpg"]):
            await adapter._handle_media_message(_make_update(msg1), MagicMock())
            await adapter._handle_media_message(_make_update(msg2), MagicMock())
            assert adapter.handle_message.await_count == 0
            await asyncio.sleep(adapter.MEDIA_GROUP_WAIT_SECONDS + 0.05)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "two images"
        assert event.media_urls == ["/tmp/burst-one.jpg", "/tmp/burst-two.jpg"]
        assert len(event.media_types) == 2

    @pytest.mark.asyncio
    async def test_photo_album_is_buffered_and_combined(self, adapter):
        first_photo = _make_photo(_make_file_obj(b"first"))
        second_photo = _make_photo(_make_file_obj(b"second"))

        msg1 = _make_message(caption="two images", media_group_id="album-1", photo=[first_photo])
        msg2 = _make_message(media_group_id="album-1", photo=[second_photo])

        with patch("gateway.platforms.telegram.cache_image_from_bytes", side_effect=["/tmp/one.jpg", "/tmp/two.jpg"]):
            await adapter._handle_media_message(_make_update(msg1), MagicMock())
            await adapter._handle_media_message(_make_update(msg2), MagicMock())
            assert adapter.handle_message.await_count == 0
            await asyncio.sleep(adapter.MEDIA_GROUP_WAIT_SECONDS + 0.05)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.call_args[0][0]
        assert event.text == "two images"
        assert event.media_urls == ["/tmp/one.jpg", "/tmp/two.jpg"]
        assert len(event.media_types) == 2

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_media_group_flush(self, adapter):
        first_photo = _make_photo(_make_file_obj(b"first"))
        msg = _make_message(caption="two images", media_group_id="album-2", photo=[first_photo])

        with patch("gateway.platforms.telegram.cache_image_from_bytes", return_value="/tmp/one.jpg"):
            await adapter._handle_media_message(_make_update(msg), MagicMock())

        assert "album-2" in adapter._media_group_events
        assert "album-2" in adapter._media_group_tasks

        await adapter.disconnect()
        await asyncio.sleep(adapter.MEDIA_GROUP_WAIT_SECONDS + 0.05)

        assert adapter._media_group_events == {}
        assert adapter._media_group_tasks == {}
        adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestSendVoice — outbound audio delivery
# ---------------------------------------------------------------------------

class TestSendVoice:
    """Tests for TelegramAdapter.send_voice() routing across audio formats."""

    @pytest.fixture()
    def connected_adapter(self, adapter):
        """Adapter with a mock bot attached."""
        bot = AsyncMock()
        adapter._bot = bot
        return adapter

    @pytest.mark.asyncio
    async def test_flac_falls_back_to_document(self, connected_adapter, tmp_path):
        """Telegram sendAudio does not accept FLAC — must fall back to sendDocument."""
        audio_file = tmp_path / "clip.flac"
        audio_file.write_bytes(b"fLaC" + b"\x00" * 32)

        mock_msg = MagicMock()
        mock_msg.message_id = 101
        connected_adapter._bot.send_voice = AsyncMock()
        connected_adapter._bot.send_audio = AsyncMock()
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        result = await connected_adapter.send_voice(
            chat_id="12345",
            audio_path=str(audio_file),
            caption="Audio",
        )

        assert result.success is True
        assert result.message_id == "101"
        connected_adapter._bot.send_document.assert_awaited_once()
        connected_adapter._bot.send_audio.assert_not_awaited()
        connected_adapter._bot.send_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wav_falls_back_to_document(self, connected_adapter, tmp_path):
        """Telegram sendAudio does not accept WAV — must fall back to sendDocument."""
        audio_file = tmp_path / "clip.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 32)

        mock_msg = MagicMock()
        mock_msg.message_id = 102
        connected_adapter._bot.send_voice = AsyncMock()
        connected_adapter._bot.send_audio = AsyncMock()
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        result = await connected_adapter.send_voice(
            chat_id="12345",
            audio_path=str(audio_file),
        )

        assert result.success is True
        connected_adapter._bot.send_document.assert_awaited_once()
        connected_adapter._bot.send_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mp3_routes_to_send_audio(self, connected_adapter, tmp_path):
        """MP3 is Telegram-sendAudio-compatible."""
        audio_file = tmp_path / "clip.mp3"
        audio_file.write_bytes(b"ID3" + b"\x00" * 32)

        mock_msg = MagicMock()
        mock_msg.message_id = 103
        connected_adapter._bot.send_voice = AsyncMock()
        connected_adapter._bot.send_audio = AsyncMock(return_value=mock_msg)
        connected_adapter._bot.send_document = AsyncMock()

        result = await connected_adapter.send_voice(
            chat_id="12345",
            audio_path=str(audio_file),
        )

        assert result.success is True
        connected_adapter._bot.send_audio.assert_awaited_once()
        connected_adapter._bot.send_document.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestSendDocument — outbound file attachment delivery
# ---------------------------------------------------------------------------

class TestSendDocument:
    """Tests for TelegramAdapter.send_document() — sending files to users."""

    @pytest.fixture()
    def connected_adapter(self, adapter):
        """Adapter with a mock bot attached."""
        bot = AsyncMock()
        adapter._bot = bot
        return adapter

    @pytest.mark.asyncio
    async def test_send_document_success(self, connected_adapter, tmp_path):
        """A local file is sent via bot.send_document and returns success."""
        # Create a real temp file
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake content")

        mock_msg = MagicMock()
        mock_msg.message_id = 99
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            caption="Here's the report",
        )

        assert result.success is True
        assert result.message_id == "99"
        connected_adapter._bot.send_document.assert_called_once()
        call_kwargs = connected_adapter._bot.send_document.call_args[1]
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["filename"] == "report.pdf"
        assert call_kwargs["caption"] == "Here's the report"

    @pytest.mark.asyncio
    async def test_send_document_custom_filename(self, connected_adapter, tmp_path):
        """The file_name parameter overrides the basename for display."""
        test_file = tmp_path / "doc_abc123_ugly.csv"
        test_file.write_bytes(b"a,b,c\n1,2,3")

        mock_msg = MagicMock()
        mock_msg.message_id = 100
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            file_name="clean_data.csv",
        )

        assert result.success is True
        call_kwargs = connected_adapter._bot.send_document.call_args[1]
        assert call_kwargs["filename"] == "clean_data.csv"

    @pytest.mark.asyncio
    async def test_send_document_file_not_found(self, connected_adapter):
        """Missing file returns error without calling Telegram API."""
        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path="/nonexistent/file.pdf",
        )

        assert result.success is False
        assert "not found" in result.error.lower()
        connected_adapter._bot.send_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_document_workspace_path_has_docker_hint(self, connected_adapter):
        """Container-local-looking paths get a more actionable Docker hint."""
        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path="/workspace/report.txt",
        )

        assert result.success is False
        assert "docker sandbox" in result.error.lower()
        assert "host-visible path" in result.error.lower()
        connected_adapter._bot.send_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_document_outputs_path_has_docker_hint(self, connected_adapter):
        """Legacy /outputs paths also get the Docker hint."""
        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path="/outputs/report.txt",
        )

        assert result.success is False
        assert "docker sandbox" in result.error.lower()
        assert "host-visible path" in result.error.lower()
        connected_adapter._bot.send_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_document_not_connected(self, adapter):
        """If bot is None, returns not connected error."""
        result = await adapter.send_document(
            chat_id="12345",
            file_path="/some/file.pdf",
        )

        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_document_caption_truncated(self, connected_adapter, tmp_path):
        """Captions longer than 1024 chars are truncated."""
        test_file = tmp_path / "data.json"
        test_file.write_bytes(b"{}")

        mock_msg = MagicMock()
        mock_msg.message_id = 101
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        long_caption = "x" * 2000
        await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            caption=long_caption,
        )

        call_kwargs = connected_adapter._bot.send_document.call_args[1]
        assert len(call_kwargs["caption"]) == 1024

    @pytest.mark.asyncio
    async def test_send_document_api_error_falls_back(self, connected_adapter, tmp_path):
        """If Telegram API raises, falls back to base class text message."""
        test_file = tmp_path / "file.pdf"
        test_file.write_bytes(b"data")

        connected_adapter._bot.send_document = AsyncMock(
            side_effect=RuntimeError("Telegram API error")
        )

        # The base fallback calls self.send() which is also on _bot, so mock it
        # to avoid cascading errors.
        connected_adapter.send = AsyncMock(
            return_value=SendResult(success=True, message_id="fallback")
        )

        result = await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
        )

        # Should have fallen back to base class
        assert result.success is True
        assert result.message_id == "fallback"

    @pytest.mark.asyncio
    async def test_send_document_reply_to(self, connected_adapter, tmp_path):
        """reply_to parameter is forwarded as reply_to_message_id."""
        test_file = tmp_path / "spec.md"
        test_file.write_bytes(b"# Spec")

        mock_msg = MagicMock()
        mock_msg.message_id = 102
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            reply_to="50",
        )

        call_kwargs = connected_adapter._bot.send_document.call_args[1]
        assert call_kwargs["reply_to_message_id"] == 50

    @pytest.mark.asyncio
    async def test_send_document_thread_id(self, connected_adapter, tmp_path):
        """metadata thread_id is forwarded as message_thread_id (required for Telegram forum groups)."""
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"%PDF-1.4 data")

        mock_msg = MagicMock()
        mock_msg.message_id = 103
        connected_adapter._bot.send_document = AsyncMock(return_value=mock_msg)

        await connected_adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            metadata={"thread_id": "789"},
        )

        call_kwargs = connected_adapter._bot.send_document.call_args[1]
        assert call_kwargs["message_thread_id"] == 789


class TestTelegramPhotoBatching:
    @pytest.mark.asyncio
    async def test_flush_photo_batch_does_not_drop_newer_scheduled_task(self, adapter):
        old_task = MagicMock()
        new_task = MagicMock()
        batch_key = "session:photo-burst"
        adapter._pending_photo_batch_tasks[batch_key] = new_task
        adapter._pending_photo_batches[batch_key] = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=SimpleNamespace(channel_id="chat-1"),
            media_urls=["/tmp/a.jpg"],
            media_types=["image/jpeg"],
        )

        with (
            patch("gateway.platforms.telegram.asyncio.current_task", return_value=old_task),
            patch("gateway.platforms.telegram.asyncio.sleep", new=AsyncMock()),
        ):
            await adapter._flush_photo_batch(batch_key)

        assert adapter._pending_photo_batch_tasks[batch_key] is new_task

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_photo_batch_tasks(self, adapter):
        task = MagicMock()
        task.done.return_value = False
        adapter._pending_photo_batch_tasks["session:photo-burst"] = task
        adapter._pending_photo_batches["session:photo-burst"] = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=SimpleNamespace(channel_id="chat-1"),
        )
        adapter._app = MagicMock()
        adapter._app.updater.stop = AsyncMock()
        adapter._app.stop = AsyncMock()
        adapter._app.shutdown = AsyncMock()

        await adapter.disconnect()

        task.cancel.assert_called_once()
        assert adapter._pending_photo_batch_tasks == {}
        assert adapter._pending_photo_batches == {}


# ---------------------------------------------------------------------------
# TestSendVideo — outbound video delivery
# ---------------------------------------------------------------------------

class TestSendVideo:
    """Tests for TelegramAdapter.send_video() — sending videos to users."""

    @pytest.fixture()
    def connected_adapter(self, adapter):
        bot = AsyncMock()
        adapter._bot = bot
        return adapter

    @pytest.mark.asyncio
    async def test_send_video_success(self, connected_adapter, tmp_path):
        test_file = tmp_path / "clip.mp4"
        test_file.write_bytes(b"\x00\x00\x00\x1c" + b"ftyp" + b"\x00" * 100)

        mock_msg = MagicMock()
        mock_msg.message_id = 200
        connected_adapter._bot.send_video = AsyncMock(return_value=mock_msg)

        result = await connected_adapter.send_video(
            chat_id="12345",
            video_path=str(test_file),
            caption="Check this out",
        )

        assert result.success is True
        assert result.message_id == "200"
        connected_adapter._bot.send_video.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_video_file_not_found(self, connected_adapter):
        result = await connected_adapter.send_video(
            chat_id="12345",
            video_path="/nonexistent/video.mp4",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_workspace_path_has_docker_hint(self, connected_adapter):
        result = await connected_adapter.send_video(
            chat_id="12345",
            video_path="/workspace/video.mp4",
        )

        assert result.success is False
        assert "docker sandbox" in result.error.lower()
        assert "host-visible path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_video_not_connected(self, adapter):
        result = await adapter.send_video(
            chat_id="12345",
            video_path="/some/video.mp4",
        )

        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_video_thread_id(self, connected_adapter, tmp_path):
        """metadata thread_id is forwarded as message_thread_id (required for Telegram forum groups)."""
        test_file = tmp_path / "clip.mp4"
        test_file.write_bytes(b"\x00\x00\x00\x1c" + b"ftyp" + b"\x00" * 100)

        mock_msg = MagicMock()
        mock_msg.message_id = 201
        connected_adapter._bot.send_video = AsyncMock(return_value=mock_msg)

        await connected_adapter.send_video(
            chat_id="12345",
            video_path=str(test_file),
            metadata={"thread_id": "789"},
        )

        call_kwargs = connected_adapter._bot.send_video.call_args[1]
        assert call_kwargs["message_thread_id"] == 789
