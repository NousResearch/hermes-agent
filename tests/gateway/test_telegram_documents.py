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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SUPPORTED_DOCUMENT_TYPES,
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

    for name in ("telegram", "telegram.ext", "telegram.constants"):
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


def _make_message(document=None, caption=None):
    """Build a mock Telegram Message with the given document."""
    msg = MagicMock()
    msg.message_id = 42
    msg.text = caption or ""
    msg.caption = caption
    msg.date = None
    # Media flags — all None except document
    msg.photo = None
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.sticker = None
    msg.document = document
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
    """Point document cache to tmp_path so tests don't touch ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
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
    async def test_unsupported_type_rejected(self, adapter):
        doc = _make_document(file_name="archive.zip", mime_type="application/zip", file_size=100)
        msg = _make_message(document=doc)
        update = _make_update(msg)

        await adapter._handle_media_message(update, MagicMock())
        event = adapter.handle_message.call_args[0][0]
        assert "Unsupported document type" in event.text
        assert ".zip" in event.text

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
