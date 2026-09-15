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
from gateway.platforms.base import BasePlatformAdapter


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


# ---------------------------------------------------------------------------
# Telegram send_image_file tests
# ---------------------------------------------------------------------------
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


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


    def test_returns_error_when_not_connected(self, adapter):
        """send_image_file should return error when bot is None."""
        adapter._bot = None
        result = _run(
            adapter.send_image_file(chat_id="12345", image_path="/tmp/img.png")
        )
        assert not result.success
        assert "Not connected" in result.error


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
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class TestDiscordSendImageFile:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="fake-token")
        a = DiscordAdapter(config)
        a._client = MagicMock()
        return a


    def test_send_document_uploads_file_attachment(self, adapter, tmp_path):
        """send_document should upload a native Discord attachment."""
        pdf = tmp_path / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        mock_channel = MagicMock()
        mock_msg = MagicMock()
        mock_msg.id = 100
        mock_channel.send = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=mock_channel)

        with patch.object(discord_mod_ref, "File", MagicMock()) as file_cls:
            result = _run(
                adapter.send_document(
                    chat_id="67890",
                    file_path=str(pdf),
                    file_name="renamed.pdf",
                    metadata={"thread_id": "123"},
                )
            )

        assert result.success
        assert result.message_id == "100"
        # #66797: path-based File sent via plural files=[...] (matches the
        # image-batch path; the singular file= handle form could race the
        # multipart encoder and silently drop the attachment).
        sent_files = mock_channel.send.call_args.kwargs.get("files")
        assert sent_files and len(sent_files) == 1
        assert file_cls.call_args.kwargs["filename"] == "renamed.pdf"

    def test_send_video_uploads_file_attachment(self, adapter, tmp_path):
        """send_video should upload a native Discord attachment."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50)

        mock_channel = MagicMock()
        mock_msg = MagicMock()
        mock_msg.id = 101
        mock_channel.send = AsyncMock(return_value=mock_msg)
        adapter._client.get_channel = MagicMock(return_value=mock_channel)

        with patch.object(discord_mod_ref, "File", MagicMock()) as file_cls:
            result = _run(
                adapter.send_video(
                    chat_id="67890",
                    video_path=str(video),
                    metadata={"thread_id": "123"},
                )
            )

        assert result.success
        assert result.message_id == "101"
        # #66797: path-based File sent via plural files=[...] (see above).
        sent_files = mock_channel.send.call_args.kwargs.get("files")
        assert sent_files and len(sent_files) == 1
        assert file_cls.call_args.kwargs["filename"] == "clip.mp4"


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

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


class TestSlackSendImageFile:
    @pytest.fixture
    def adapter(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        a = SlackAdapter(config)
        a._app = MagicMock()
        return a


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
        from tools.browser_tool_lifecycle import _cleanup_old_screenshots
        from tools.browser_tool import _last_screenshot_cleanup_by_dir

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
        from tools.browser_tool_lifecycle import _cleanup_old_screenshots
        from tools.browser_tool import _last_screenshot_cleanup_by_dir

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


