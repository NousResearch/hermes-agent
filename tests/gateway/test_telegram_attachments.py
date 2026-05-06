"""
Tests for issue #20899 — Telegram inbound media surfaced as tool-accessible
attachments.

These cover:
  * Photos land in the per-chat attachment cache and populate
    ``MessageEvent.attachments`` with a usable absolute path.
  * Documents preserve their original filename in the cached path and
    populate ``MessageEvent.attachments``.
  * Multiple photos in a single media-group event are all represented.
  * Cross-chat isolation: messages from different chat ids land in distinct
    on-disk subdirectories.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig  # noqa: F401
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    cache_inbound_attachment,
    get_attachment_cache_dir,
)


# Re-use the telegram mocking shim from the existing documents test so this
# test module can be collected without the real python-telegram-bot.
def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Magic bytes the image cache validator requires.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 32


def _file_obj(data: bytes, file_path: str = "photos/file.jpg"):
    f = AsyncMock()
    f.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    f.file_path = file_path
    return f


def _photo_size(data: bytes, file_path: str = "photos/file.jpg"):
    """Mock a single PhotoSize variant."""
    p = MagicMock()
    p.get_file = AsyncMock(return_value=_file_obj(data, file_path))
    return p


def _document(data: bytes, file_name: str, mime_type: str = "application/pdf"):
    d = MagicMock()
    d.file_name = file_name
    d.mime_type = mime_type
    d.file_size = len(data)
    d.get_file = AsyncMock(return_value=_file_obj(data, f"docs/{file_name}"))
    return d


def _message(*, chat_id: int = 100, message_id: int = 42,
             photo=None, document=None, caption=None, media_group_id=None):
    msg = MagicMock()
    msg.message_id = message_id
    msg.text = caption or ""
    msg.caption = caption
    msg.date = None
    msg.photo = photo
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.sticker = None
    msg.document = document
    msg.media_group_id = media_group_id
    msg.chat = MagicMock()
    msg.chat.id = chat_id
    msg.chat.type = "private"
    msg.chat.title = None
    msg.chat.full_name = "Test User"
    msg.from_user = MagicMock()
    msg.from_user.id = 1
    msg.from_user.full_name = "Test User"
    msg.message_thread_id = None
    return msg


def _update(msg):
    u = MagicMock()
    u.message = msg
    return u


@pytest.fixture()
def adapter():
    cfg = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config=cfg)
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_caches(tmp_path, monkeypatch):
    """Redirect every cache used by the inbound-media path into tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "images"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "docs"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.ATTACHMENT_CACHE_DIR", tmp_path / "attachments"
    )


# ---------------------------------------------------------------------------
# cache_inbound_attachment unit tests
# ---------------------------------------------------------------------------

class TestCacheInboundAttachment:
    def test_writes_under_platform_chat_message(self, tmp_path):
        path = cache_inbound_attachment(
            b"hello",
            platform="telegram",
            chat_id=12345,
            message_id=99,
            filename="hello.txt",
        )
        p = Path(path)
        assert p.exists()
        assert p.read_bytes() == b"hello"
        # Layout: <root>/telegram/12345/99/hello.txt
        parts = p.parts
        assert "telegram" in parts
        idx = parts.index("telegram")
        assert parts[idx + 1] == "12345"
        assert parts[idx + 2] == "99"
        assert parts[idx + 3] == "hello.txt"

    def test_path_traversal_in_filename_is_neutralized(self, tmp_path):
        path = cache_inbound_attachment(
            b"x",
            platform="telegram",
            chat_id=1,
            message_id=2,
            filename="../../etc/passwd",
        )
        # The attempt must collapse to a basename inside the cache root.
        root = get_attachment_cache_dir().resolve()
        assert Path(path).resolve().is_relative_to(root)
        assert ".." not in Path(path).name

    def test_collision_disambiguated(self, tmp_path):
        a = cache_inbound_attachment(
            b"one", platform="telegram", chat_id=1, message_id=2, filename="x.bin"
        )
        b = cache_inbound_attachment(
            b"two", platform="telegram", chat_id=1, message_id=2, filename="x.bin"
        )
        assert a != b
        assert Path(a).read_bytes() == b"one"
        assert Path(b).read_bytes() == b"two"


# ---------------------------------------------------------------------------
# Telegram photo inbound -> attachment cache + MessageEvent.attachments
# ---------------------------------------------------------------------------

class TestTelegramPhotoAttachment:
    @pytest.mark.asyncio
    async def test_photo_populates_attachment_metadata(self, adapter, tmp_path):
        # Telegram delivers msg.photo as a list of PhotoSize variants; the
        # adapter must pick the largest (last).
        photo_sizes = [
            _photo_size(_JPEG_BYTES + b"small", "photos/small.jpg"),
            _photo_size(_JPEG_BYTES + b"largeXX", "photos/large.jpg"),
        ]
        msg = _message(chat_id=555, message_id=7, photo=photo_sizes)
        await adapter._handle_media_message(_update(msg), MagicMock())

        # Photo path goes through the batched flush; pull the buffered event
        # directly so we don't have to wait on the debounce timer.
        assert adapter._pending_photo_batches, "photo batch should be queued"
        event = next(iter(adapter._pending_photo_batches.values()))

        assert event.attachments, "MessageEvent.attachments must be populated"
        att = event.attachments[0]
        assert att["platform"] == "telegram"
        assert att["chat_id"] == "555"
        assert att["message_id"] == "7"
        assert att["mime_type"].startswith("image/")
        assert att["size"] > 0
        assert os.path.isabs(att["path"])
        assert os.path.exists(att["path"]), "cached attachment file must exist on disk"
        # Path must live under the chat-scoped attachment subtree
        assert "/telegram/555/7/" in att["path"]


# ---------------------------------------------------------------------------
# Telegram document inbound preserves filename + populates attachments
# ---------------------------------------------------------------------------

class TestTelegramDocumentAttachment:
    @pytest.mark.asyncio
    async def test_document_preserves_original_filename(self, adapter):
        data = b"%PDF-1.4 hi"
        doc = _document(data, file_name="quarterly_report.pdf")
        msg = _message(chat_id=42, message_id=11, document=doc)
        await adapter._handle_media_message(_update(msg), MagicMock())

        event = adapter.handle_message.call_args[0][0]
        assert event.attachments, "document should populate attachments"
        att = event.attachments[0]
        assert att["filename"] == "quarterly_report.pdf"
        # Filename is preserved inside the on-disk path under the chat scope.
        assert att["path"].endswith("quarterly_report.pdf")
        assert "/telegram/42/11/" in att["path"]
        assert os.path.exists(att["path"])
        assert att["mime_type"] == "application/pdf"
        assert att["size"] == len(data)


# ---------------------------------------------------------------------------
# Cross-chat isolation
# ---------------------------------------------------------------------------

class TestCrossChatIsolation:
    @pytest.mark.asyncio
    async def test_two_chats_land_in_distinct_directories(self, adapter):
        doc_a = _document(b"%PDF-A", file_name="a.pdf")
        msg_a = _message(chat_id=111, message_id=1, document=doc_a)
        await adapter._handle_media_message(_update(msg_a), MagicMock())
        event_a = adapter.handle_message.call_args[0][0]

        doc_b = _document(b"%PDF-B", file_name="b.pdf")
        msg_b = _message(chat_id=222, message_id=1, document=doc_b)
        await adapter._handle_media_message(_update(msg_b), MagicMock())
        event_b = adapter.handle_message.call_args[0][0]

        path_a = event_a.attachments[0]["path"]
        path_b = event_b.attachments[0]["path"]
        assert "/telegram/111/" in path_a
        assert "/telegram/222/" in path_b
        # Different chats must not share a parent directory beyond the
        # platform root, even with identical message ids.
        assert os.path.dirname(path_a) != os.path.dirname(path_b)
