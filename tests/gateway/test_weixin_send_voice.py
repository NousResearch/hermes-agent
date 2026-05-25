"""Tests for Weixin send_voice — OGG→SILK conversion and native voice bubble delivery.

These tests verify that:
1. .silk files are sent as native voice bubbles (not file attachments)
2. .ogg files are converted to .silk first, then sent as voice bubbles
3. Conversion failure falls back to file attachment
4. Temp files are cleaned up after send
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if aiohttp is unavailable.
aiohttp = pytest.importorskip("aiohttp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter() -> MagicMock:
    """Create a minimal mock WeixinAdapter for send_voice testing."""
    adapter = MagicMock()
    adapter.name = "weixin-test"
    adapter._send_session = object()
    adapter._token = "test-token"
    adapter._account_id = "bot123"
    adapter._base_url = "https://fake.api"
    adapter._cdn_base_url = "https://fake.cdn"
    adapter._token_store = MagicMock()
    adapter._token_store.get.return_value = "ctx-token"

    # Make _send_file return a message_id
    adapter._send_file = AsyncMock(return_value="msg-001")

    # Bind the real send_voice method
    from gateway.platforms.weixin import WeixinAdapter
    adapter.send_voice = WeixinAdapter.send_voice.__get__(adapter, type(adapter))

    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSendVoice:
    """Test send_voice behavior with different audio formats."""

    @pytest.mark.asyncio
    async def test_silk_sent_as_native_voice(self, tmp_path):
        """A .silk file should be sent via _send_file without force_file_attachment."""
        adapter = _make_adapter()
        silk = tmp_path / "voice.silk"
        silk.write_bytes(b"\x02#!SILK_V3" + b"\x00" * 50)

        with patch("gateway.platforms.weixin.get_audio_duration_s", return_value=5):
            result = await adapter.send_voice("wxid_user1", str(silk))

        assert result.success is True
        adapter._send_file.assert_called_once()
        # Should NOT have force_file_attachment=True
        call_kwargs = adapter._send_file.call_args
        assert call_kwargs.kwargs.get("force_file_attachment") is not True

    @pytest.mark.asyncio
    async def test_ogg_converted_to_silk(self, tmp_path):
        """An .ogg file should be converted to .silk, then sent as voice bubble."""
        adapter = _make_adapter()
        ogg = tmp_path / "tts_output.ogg"
        ogg.write_bytes(b"OggS\x00\x02" + b"\x00" * 100)

        silk_output = tmp_path / "tts_output.silk"
        silk_output.write_bytes(b"\x02#!SILK_V3" + b"\x00" * 50)

        with patch("gateway.platforms.weixin.ogg_to_silk", return_value=str(silk_output)), \
             patch("gateway.platforms.weixin.get_audio_duration_s", return_value=3), \
             patch("gateway.platforms.weixin._has_pilk", return_value=True), \
             patch("gateway.platforms.weixin.cleanup_silk_dir"):
            result = await adapter.send_voice("wxid_user1", str(ogg))

        assert result.success is True
        # _send_file should be called with the SILK path, not the OGG path
        adapter._send_file.assert_called_once()
        sent_path = adapter._send_file.call_args.args[1]
        assert sent_path.endswith(".silk")

    @pytest.mark.asyncio
    async def test_ogg_fallback_on_conversion_failure(self, tmp_path):
        """If OGG→SILK conversion fails, fall back to file attachment."""
        adapter = _make_adapter()
        ogg = tmp_path / "tts_output.ogg"
        ogg.write_bytes(b"OggS\x00\x02" + b"\x00" * 100)

        with patch("gateway.platforms.weixin.ogg_to_silk", return_value=None), \
             patch("gateway.platforms.weixin._has_pilk", return_value=True), \
             patch("gateway.platforms.weixin.cleanup_silk_dir"):
            result = await adapter.send_voice("wxid_user1", str(ogg))

        assert result.success is True
        adapter._send_file.assert_called_once()
        # Should have force_file_attachment=True (fallback)
        call_kwargs = adapter._send_file.call_args
        assert call_kwargs.kwargs.get("force_file_attachment") is True

    @pytest.mark.asyncio
    async def test_cleanup_called_in_finally(self, tmp_path):
        """Temporary SILK directory should be cleaned up even on error."""
        adapter = _make_adapter()
        ogg = tmp_path / "tts_output.ogg"
        ogg.write_bytes(b"OggS\x00\x02" + b"\x00" * 100)

        silk_output = tmp_path / "tts_output.silk"
        silk_output.write_bytes(b"\x02#!SILK_V3" + b"\x00" * 50)

        with patch("gateway.platforms.weixin.ogg_to_silk", return_value=str(silk_output)), \
             patch("gateway.platforms.weixin.get_audio_duration_s", return_value=3), \
             patch("gateway.platforms.weixin._has_pilk", return_value=True), \
             patch("gateway.platforms.weixin.cleanup_silk_dir") as mock_cleanup:
            result = await adapter.send_voice("wxid_user1", str(ogg))

        assert result.success is True
        mock_cleanup.assert_called_once_with(str(silk_output))

    @pytest.mark.asyncio
    async def test_returns_error_when_not_connected(self):
        """Returns error when adapter is not connected."""
        adapter = _make_adapter()
        adapter._send_session = None

        result = await adapter.send_voice("wxid_user1", "/fake/path.ogg")
        assert result.success is False
        assert "Not connected" in result.error
