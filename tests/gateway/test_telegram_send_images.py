import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.request import pathname2url

from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter, _MEDIA_SEND_READ_TIMEOUT, _MEDIA_SEND_DEADLINE
from gateway.config import PlatformConfig


@pytest.fixture()
def telegram_adapter():
    config = PlatformConfig(enabled=True, token="123456:fake-token-for-test", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = MagicMock()
    return adapter


class TestTelegramSendImages:

    def test_media_send_kwargs_has_write_timeout(self, telegram_adapter):
        reply_to_id, kwargs = telegram_adapter._media_send_kwargs("12345", None, None)
        assert "write_timeout" in kwargs
        assert kwargs["write_timeout"] == _MEDIA_SEND_DEADLINE
        assert kwargs["read_timeout"] == _MEDIA_SEND_READ_TIMEOUT

    @pytest.mark.asyncio
    async def test_single_photo_bypasses_send_media_group(self, telegram_adapter, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        telegram_adapter._bot.send_media_group = AsyncMock()
        telegram_adapter.send_image_file = AsyncMock(return_value=SendResult(success=True, message_id="101"))

        url = f"file://{pathname2url(str(img_path))}"
        result = await telegram_adapter.send_multiple_images("12345", [(url, "caption")])

        assert result.success is True
        telegram_adapter._bot.send_media_group.assert_not_called()
        telegram_adapter.send_image_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_photos_uses_send_media_group(self, telegram_adapter, tmp_path):
        img1 = tmp_path / "img1.png"
        img1.write_bytes(b"\x89PNG\r\n\x1a\nfake1")
        img2 = tmp_path / "img2.png"
        img2.write_bytes(b"\x89PNG\r\n\x1a\nfake2")

        mock_msg = MagicMock()
        mock_msg.message_id = 202
        telegram_adapter._bot.send_media_group = AsyncMock(return_value=[mock_msg, mock_msg])

        url1 = f"file://{pathname2url(str(img1))}"
        url2 = f"file://{pathname2url(str(img2))}"
        result = await telegram_adapter.send_multiple_images("12345", [(url1, "cap1"), (url2, "cap2")])

        assert result.success is True
        telegram_adapter._bot.send_media_group.assert_called_once()
        call_kwargs = telegram_adapter._bot.send_media_group.call_args[1]
        assert len(call_kwargs["media"]) == 2

    @pytest.mark.asyncio
    async def test_windows_file_url_decoded_correctly(self, telegram_adapter, tmp_path):
        img1 = tmp_path / "test1.png"
        img1.write_bytes(b"\x89PNG\r\n\x1a\nfake1")
        img2 = tmp_path / "test2.png"
        img2.write_bytes(b"\x89PNG\r\n\x1a\nfake2")

        # Explicitly test Windows URL format: file:///C:/path/to/test.png
        url1 = f"file:///{str(img1).replace(os.sep, '/')}"
        url2 = f"file:///{str(img2).replace(os.sep, '/')}"

        mock_msg = MagicMock()
        mock_msg.message_id = 303
        telegram_adapter._bot.send_media_group = AsyncMock(return_value=[mock_msg, mock_msg])

        result = await telegram_adapter.send_multiple_images("12345", [(url1, "cap1"), (url2, "cap2")])
        assert result.success is True
        telegram_adapter._bot.send_media_group.assert_called_once()
        call_kwargs = telegram_adapter._bot.send_media_group.call_args[1]
        assert len(call_kwargs["media"]) == 2
