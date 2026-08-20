"""
Tests for [[spoiler]] directive support — Telegram photo spoiler (tap to reveal).

Covers:
- [[spoiler]] stripped from cleaned text in extract_media
- has_spoiler passed through metadata to send_photo / InputMediaPhoto
- spoiler flag forwarded in send_image_file, send_image, send_multiple_images
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def _run(coro):
    """Run a coroutine in a fresh event loop for sync-style tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# extract_media: [[spoiler]] directive stripping
# ---------------------------------------------------------------------------


class TestSpoilerDirectiveStripping:
    """[[spoiler]] is a delivery directive — strip it from user-visible text."""

    def test_spoiler_directive_stripped_from_cleaned_text(self):
        """[[spoiler]] must be removed from cleaned output."""
        content = "Here is the image:\n[[spoiler]]\nMEDIA:/tmp/nsfw.jpg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/nsfw.jpg", False)]
        assert "[[spoiler]]" not in cleaned
        assert "Here is the image" in cleaned

    def test_spoiler_does_not_affect_voice_flag(self):
        """[[spoiler]] is independent of [[audio_as_voice]]."""
        content = "[[spoiler]]\nMEDIA:/tmp/photo.png"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/photo.png", False)]  # voice flag stays False
        assert "[[spoiler]]" not in cleaned

    def test_spoiler_coexists_with_as_document(self):
        """[[spoiler]] and [[as_document]] can appear in the same response."""
        content = "[[spoiler]]\n[[as_document]]\nMEDIA:/tmp/img.png"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/img.png", False)]
        assert "[[spoiler]]" not in cleaned
        assert "[[as_document]]" not in cleaned

    def test_spoiler_coexists_with_audio_as_voice(self):
        """All three directives can coexist."""
        content = "[[audio_as_voice]]\n[[spoiler]]\nMEDIA:/tmp/x.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/x.ogg", True)]
        assert "[[spoiler]]" not in cleaned
        assert "[[audio_as_voice]]" not in cleaned


# ---------------------------------------------------------------------------
# Telegram adapter: has_spoiler in send_photo / InputMediaPhoto
# ---------------------------------------------------------------------------


def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
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

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class TestTelegramSendImageFileSpoiler:
    """send_image_file passes has_spoiler to bot.send_photo when metadata requests it."""

    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = TelegramAdapter(config)
        a._bot = MagicMock()
        return a

    def test_has_spoiler_true_forwarded(self, adapter, tmp_path):
        """When metadata has has_spoiler=True, send_photo gets has_spoiler=True."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_msg = MagicMock()
        mock_msg.message_id = 1
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        result = _run(
            adapter.send_image_file(
                chat_id="12345",
                image_path=str(img),
                metadata={"has_spoiler": True},
            )
        )
        assert result.success
        call_kwargs = adapter._bot.send_photo.call_args.kwargs
        assert call_kwargs["has_spoiler"] is True

    def test_has_spoiler_false_when_not_in_metadata(self, adapter, tmp_path):
        """When metadata does not contain has_spoiler, it defaults to False."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_msg = MagicMock()
        mock_msg.message_id = 2
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        result = _run(
            adapter.send_image_file(
                chat_id="12345",
                image_path=str(img),
                metadata={"thread_id": "100"},
            )
        )
        assert result.success
        call_kwargs = adapter._bot.send_photo.call_args.kwargs
        assert call_kwargs["has_spoiler"] is False

    def test_has_spoiler_false_when_metadata_is_none(self, adapter, tmp_path):
        """When metadata is None, has_spoiler defaults to False."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_msg = MagicMock()
        mock_msg.message_id = 3
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        result = _run(
            adapter.send_image_file(
                chat_id="12345",
                image_path=str(img),
            )
        )
        assert result.success
        call_kwargs = adapter._bot.send_photo.call_args.kwargs
        assert call_kwargs["has_spoiler"] is False


class TestTelegramSendImageSpoiler:
    """send_image passes has_spoiler to bot.send_photo for URL-based images."""

    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = TelegramAdapter(config)
        a._bot = MagicMock()
        return a

    def test_has_spoiler_forwarded_for_url_image(self, adapter):
        """URL-based send_image passes has_spoiler through."""
        mock_msg = MagicMock()
        mock_msg.message_id = 10
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        result = _run(
            adapter.send_image(
                chat_id="12345",
                image_url="https://example.com/photo.jpg",
                metadata={"has_spoiler": True},
            )
        )
        assert result.success
        call_kwargs = adapter._bot.send_photo.call_args.kwargs
        assert call_kwargs["has_spoiler"] is True


class TestTelegramSendMultipleImagesSpoiler:
    """send_multiple_images passes has_spoiler to InputMediaPhoto."""

    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = TelegramAdapter(config)
        a._bot = MagicMock()
        return a

    def test_has_spoiler_passed_to_input_media_photo(self, adapter, tmp_path):
        """InputMediaPhoto receives has_spoiler=True from metadata."""
        from urllib.parse import quote

        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Capture InputMediaPhoto constructor calls
        captured_kwargs = []
        original_InputMediaPhoto = None

        try:
            from telegram import InputMediaPhoto as _IMP
            original_InputMediaPhoto = _IMP
        except (ImportError, TypeError):
            pass

        class SpyInputMediaPhoto:
            def __init__(self, **kwargs):
                captured_kwargs.append(kwargs)

        mock_msgs = [MagicMock(message_id=1)]
        adapter._bot.send_media_group = AsyncMock(return_value=mock_msgs)

        images = [(f"file://{quote(str(img))}", "")]

        with patch("telegram.InputMediaPhoto", SpyInputMediaPhoto):
            _run(
                adapter.send_multiple_images(
                    chat_id="12345",
                    images=images,
                    metadata={"has_spoiler": True},
                )
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["has_spoiler"] is True

    def test_has_spoiler_false_by_default(self, adapter, tmp_path):
        """InputMediaPhoto receives has_spoiler=False when not in metadata."""
        from urllib.parse import quote

        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        captured_kwargs = []

        class SpyInputMediaPhoto:
            def __init__(self, **kwargs):
                captured_kwargs.append(kwargs)

        mock_msgs = [MagicMock(message_id=1)]
        adapter._bot.send_media_group = AsyncMock(return_value=mock_msgs)

        images = [(f"file://{quote(str(img))}", "")]

        with patch("telegram.InputMediaPhoto", SpyInputMediaPhoto):
            _run(
                adapter.send_multiple_images(
                    chat_id="12345",
                    images=images,
                    metadata={"thread_id": "100"},
                )
            )

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["has_spoiler"] is False
