"""
Tests for send_image_file() on Telegram, Discord, and Slack platforms,
and MEDIA: .png extraction/routing in the base platform adapter.

Covers: local image file sending, file-not-found handling, fallback on error,
        MEDIA: tag extraction for image extensions, and routing to send_image_file.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


def _run(coro):
    """Run a coroutine in a fresh event loop for sync-style tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# MEDIA: extraction tests for image files
# ---------------------------------------------------------------------------


class TestExtractMediaImages:
    """Test that MEDIA: tags with image extensions are correctly extracted."""

    def test_png_image_extracted(self):
        content = "Here is the screenshot:\nMEDIA:/home/user/.hermes/browser_screenshots/shot.png"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/home/user/.hermes/browser_screenshots/shot.png"
        assert "MEDIA:" not in cleaned
        assert "Here is the screenshot" in cleaned

    def test_jpg_image_extracted(self):
        content = "MEDIA:/tmp/photo.jpg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/tmp/photo.jpg"

    def test_webp_image_extracted(self):
        content = "MEDIA:/tmp/image.webp"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1

    def test_mixed_audio_and_image(self):
        content = "MEDIA:/audio.ogg\nMEDIA:/screenshot.png"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert len(media) == 2
        paths = [m[0] for m in media]
        assert "/audio.ogg" in paths
        assert "/screenshot.png" in paths


# ---------------------------------------------------------------------------
# Telegram send_image_file tests
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

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


class TestTelegramSendImageFile:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = TelegramAdapter(config)
        a._bot = MagicMock()
        return a

    def test_sends_local_image_as_photo(self, adapter, tmp_path):
        """send_image_file should call bot.send_photo with the opened file."""
        img = tmp_path / "screenshot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # Minimal PNG-like

        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        result = _run(
            adapter.send_image_file(chat_id="12345", image_path=str(img))
        )
        assert result.success
        assert result.message_id == "42"
        adapter._bot.send_photo.assert_awaited_once()

        # Verify photo arg was a file object (opened in rb mode)
        call_kwargs = adapter._bot.send_photo.call_args
        assert call_kwargs.kwargs["chat_id"] == 12345

    def test_returns_error_when_file_missing(self, adapter):
        """send_image_file should return error for nonexistent file."""
        result = _run(
            adapter.send_image_file(chat_id="12345", image_path="/nonexistent/image.png")
        )
        assert not result.success
        assert "not found" in result.error

    def test_returns_error_when_not_connected(self, adapter):
        """send_image_file should return error when bot is None."""
        adapter._bot = None
        result = _run(
            adapter.send_image_file(chat_id="12345", image_path="/tmp/img.png")
        )
        assert not result.success
        assert "Not connected" in result.error

    def test_caption_truncated_to_1024(self, adapter, tmp_path):
        """Telegram captions have a 1024 char limit."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        mock_msg = MagicMock()
        mock_msg.message_id = 1
        adapter._bot.send_photo = AsyncMock(return_value=mock_msg)

        long_caption = "A" * 2000
        _run(
            adapter.send_image_file(chat_id="12345", image_path=str(img), caption=long_caption)
        )

        call_kwargs = adapter._bot.send_photo.call_args.kwargs
        assert len(call_kwargs["caption"]) == 1024


# ---------------------------------------------------------------------------
# Discord send_image_file tests
# ---------------------------------------------------------------------------


def _ensure_discord_mock():
    """Install mock discord module so DiscordAdapter can be imported."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock

    for name in ("discord", "discord.ext", "discord.ext.commands"):
        sys.modules.setdefault(name, discord_mod)


_ensure_discord_mock()

import discord as discord_mod_ref  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class TestDiscordSendImageFile:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = DiscordAdapter(config)
        a._client = MagicMock()
        return a

    def test_sends_local_image_as_attachment(self, adapter, tmp_path):
        """send_image_file should create discord.File and send to channel."""
        img = tmp_path / "screenshot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        mock_channel = MagicMock()
        mock_msg = MagicMock()
        mock_msg.id = 99
        mock_channel.send = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=mock_channel)

        result = _run(
            adapter.send_image_file(chat_id="67890", image_path=str(img))
        )
        assert result.success
        assert result.message_id == "99"
        mock_channel.send.assert_awaited_once()

    def test_returns_error_when_file_missing(self, adapter):
        result = _run(
            adapter.send_image_file(chat_id="67890", image_path="/nonexistent.png")
        )
        assert not result.success
        assert "not found" in result.error

    def test_returns_error_when_not_connected(self, adapter):
        adapter._client = None
        result = _run(
            adapter.send_image_file(chat_id="67890", image_path="/tmp/img.png")
        )
        assert not result.success
        assert "Not connected" in result.error

    def test_handles_missing_channel(self, adapter):
        adapter._client.get_channel = MagicMock(return_value=None)
        adapter._client.fetch_channel = AsyncMock(return_value=None)

        result = _run(
            adapter.send_image_file(chat_id="99999", image_path="/tmp/img.png")
        )
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Slack send_image_file tests
# ---------------------------------------------------------------------------


def _ensure_slack_mock():
    """Install mock slack_bolt module so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_mod = MagicMock()
    for name in ("slack_bolt", "slack_bolt.async_app", "slack_sdk", "slack_sdk.web.async_client"):
        sys.modules.setdefault(name, slack_mod)


_ensure_slack_mock()

from gateway.platforms.slack import SlackAdapter  # noqa: E402


class TestSlackSendImageFile:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        a = SlackAdapter(config)
        a._app = MagicMock()
        return a

    def test_sends_local_image_via_upload(self, adapter, tmp_path):
        """send_image_file should call files_upload_v2 with the local path."""
        img = tmp_path / "screenshot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)

        mock_result = MagicMock()
        adapter._app.client.files_upload_v2 = AsyncMock(return_value=mock_result)

        result = _run(
            adapter.send_image_file(chat_id="C12345", image_path=str(img))
        )
        assert result.success
        adapter._app.client.files_upload_v2.assert_awaited_once()

        call_kwargs = adapter._app.client.files_upload_v2.call_args.kwargs
        assert call_kwargs["file"] == str(img)
        assert call_kwargs["filename"] == "screenshot.png"
        assert call_kwargs["channel"] == "C12345"

    def test_returns_error_when_file_missing(self, adapter):
        result = _run(
            adapter.send_image_file(chat_id="C12345", image_path="/nonexistent.png")
        )
        assert not result.success
        assert "not found" in result.error

    def test_returns_error_when_not_connected(self, adapter):
        adapter._app = None
        result = _run(
            adapter.send_image_file(chat_id="C12345", image_path="/tmp/img.png")
        )
        assert not result.success
        assert "Not connected" in result.error


# ---------------------------------------------------------------------------
# browser_vision screenshot cleanup tests
# ---------------------------------------------------------------------------


class TestScreenshotCleanup:
    def test_cleanup_removes_old_screenshots(self, tmp_path):
        """_cleanup_old_screenshots should remove files older than max_age_hours."""
        import time
        from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir

        _last_screenshot_cleanup_by_dir.clear()

        # Create a "fresh" file
        fresh = tmp_path / "browser_screenshot_fresh.png"
        fresh.write_bytes(b"new")

        # Create an "old" file and backdate its mtime
        old = tmp_path / "browser_screenshot_old.png"
        old.write_bytes(b"old")
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(str(old), (old_time, old_time))

        _cleanup_old_screenshots(tmp_path, max_age_hours=24)

        assert fresh.exists(), "Fresh screenshot should not be removed"
        assert not old.exists(), "Old screenshot should be removed"

    def test_cleanup_is_throttled_per_directory(self, tmp_path):
        import time
        from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir

        _last_screenshot_cleanup_by_dir.clear()

        old = tmp_path / "browser_screenshot_old.png"
        old.write_bytes(b"old")
        old_time = time.time() - (25 * 3600)
        os.utime(str(old), (old_time, old_time))

        _cleanup_old_screenshots(tmp_path, max_age_hours=24)
        assert not old.exists()

        old.write_bytes(b"old-again")
        os.utime(str(old), (old_time, old_time))
        _cleanup_old_screenshots(tmp_path, max_age_hours=24)

        assert old.exists(), "Repeated cleanup should be skipped while throttled"

    def test_cleanup_ignores_non_screenshot_files(self, tmp_path):
        """Only files matching browser_screenshot_*.png should be cleaned."""
        import time
        from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir

        _last_screenshot_cleanup_by_dir.clear()

        other_file = tmp_path / "important_data.txt"
        other_file.write_bytes(b"keep me")
        old_time = time.time() - (48 * 3600)
        os.utime(str(other_file), (old_time, old_time))

        _cleanup_old_screenshots(tmp_path, max_age_hours=24)

        assert other_file.exists(), "Non-screenshot files should not be touched"

    def test_cleanup_handles_empty_dir(self, tmp_path):
        """Cleanup should not fail on empty directory."""
        from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir
        _last_screenshot_cleanup_by_dir.clear()
        _cleanup_old_screenshots(tmp_path, max_age_hours=24)  # Should not raise

    def test_cleanup_handles_nonexistent_dir(self):
        """Cleanup should not fail if directory doesn't exist."""
        from pathlib import Path
        from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir
        _last_screenshot_cleanup_by_dir.clear()
        _cleanup_old_screenshots(Path("/nonexistent/dir"), max_age_hours=24)  # Should not raise
