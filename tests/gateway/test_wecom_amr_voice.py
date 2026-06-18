"""Tests for AMR voice routing in WeCom gateway adapter and cross-platform regression."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import SendResult, _AUDIO_EXTS, _TELEGRAM_VOICE_EXTS, should_send_media_as_audio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_live_adapters():
    """Ensure _LIVE_ADAPTERS is clean between tests."""
    from gateway.platforms.wecom import _LIVE_ADAPTERS
    _LIVE_ADAPTERS.clear()
    yield
    _LIVE_ADAPTERS.clear()


def _make_mock_adapter(bot_id="test-bot-123"):
    """Create a mock WeComAdapter with a fake WebSocket."""
    adapter = MagicMock()
    adapter._bot_id = bot_id
    adapter._ws = MagicMock()
    adapter._ws.closed = False
    adapter.is_connected = True
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg_1"))
    adapter.send_image_file = AsyncMock(return_value=SendResult(success=True, message_id="img_1"))
    adapter.send_video = AsyncMock(return_value=SendResult(success=True, message_id="vid_1"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice_1"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc_1"))
    return adapter


# ---------------------------------------------------------------------------
# Test 1: AMR file routes to send_voice in WeCom adapter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_amr_file_routes_to_send_voice(tmp_path):
    """When send_wecom_direct receives an AMR file flagged as voice, it calls send_voice."""
    from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

    adapter = _make_mock_adapter("test-bot-123")
    _LIVE_ADAPTERS["test-bot-123"] = adapter

    amr_file = tmp_path / "voice.amr"
    amr_file.write_bytes(b"#!AMR\n\x00" * 100)

    result = await send_wecom_direct(
        extra={"bot_id": "test-bot-123"},
        chat_id="test_user",
        message="",
        media_files=[(str(amr_file), True)],  # is_voice=True
    )

    assert result["success"] is True
    assert result["message_id"] == "voice_1"
    adapter.send_voice.assert_called_once()
    adapter.send_document.assert_not_called()
    # Verify the audio_path passed to send_voice is the .amr file
    call_args = adapter.send_voice.call_args
    assert call_args[0][1].endswith(".amr")


# ---------------------------------------------------------------------------
# Test 2: send_wecom_direct with AMR file calls send_voice (even without is_voice flag)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_amr_without_voice_flag_routes_to_send_voice(tmp_path):
    """AMR in _WECOM_DIRECT_AUDIO_EXTS falls through to send_voice even if is_voice=False."""
    from gateway.platforms.wecom import _LIVE_ADAPTERS, send_wecom_direct

    adapter = _make_mock_adapter("test-bot-123")
    _LIVE_ADAPTERS["test-bot-123"] = adapter

    amr_file = tmp_path / "audio.amr"
    amr_file.write_bytes(b"#!AMR\n\x00" * 100)

    result = await send_wecom_direct(
        extra={"bot_id": "test-bot-123"},
        chat_id="test_user",
        message="",
        media_files=[(str(amr_file), False)],  # is_voice=False
    )

    assert result["success"] is True
    assert result["message_id"] == "voice_1"
    # .amr is in _WECOM_DIRECT_AUDIO_EXTS, so it routes to send_voice as fallback
    adapter.send_voice.assert_called_once()
    adapter.send_document.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: _AUDIO_EXTS in base.py includes .amr
# ---------------------------------------------------------------------------

def test_audio_extensions_include_amr():
    """The global _AUDIO_EXTS frozenset in base.py includes '.amr'."""
    assert ".amr" in _AUDIO_EXTS
    # Also verify other known extensions are still present (regression check)
    assert ".ogg" in _AUDIO_EXTS
    assert ".opus" in _AUDIO_EXTS
    assert ".mp3" in _AUDIO_EXTS


# ---------------------------------------------------------------------------
# Test 4: Telegram regression — AMR files do NOT route to sendVoice
# ---------------------------------------------------------------------------

def test_telegram_does_not_accept_amr_as_voice():
    """Telegram's Bot API only accepts Opus/OGG for sendVoice; AMR is not included."""
    assert ".amr" not in _TELEGRAM_VOICE_EXTS
    assert ".ogg" in _TELEGRAM_VOICE_EXTS
    assert ".opus" in _TELEGRAM_VOICE_EXTS

    # should_send_media_as_audio: AMR recognized as audio, but Telegram voice
    # path is guarded by _TELEGRAM_VOICE_EXTS
    assert should_send_media_as_audio("telegram", ".amr", is_voice=True) is False
    # Regular audio attachment: AMR not in _TELEGRAM_AUDIO_ATTACHMENT_EXTS either
    assert should_send_media_as_audio("telegram", ".amr", is_voice=False) is False


# ---------------------------------------------------------------------------
# Test 5: WeChat/WeCom regression — voice extension sets include .amr
# ---------------------------------------------------------------------------

def test_wecom_voice_extensions_include_amr():
    """WeCom's _WECOM_DIRECT_VOICE_EXTS and _AUDIO_EXTS include .amr."""
    from gateway.platforms.wecom import _WECOM_DIRECT_VOICE_EXTS, _WECOM_DIRECT_AUDIO_EXTS

    assert ".amr" in _WECOM_DIRECT_VOICE_EXTS
    assert ".amr" in _WECOM_DIRECT_AUDIO_EXTS

    # For non-Telegram platforms, AMR should return True from should_send_media_as_audio
    assert should_send_media_as_audio("wecom", ".amr", is_voice=True) is True
    assert should_send_media_as_audio("wecom", ".amr", is_voice=False) is True
    assert should_send_media_as_audio("weixin", ".amr", is_voice=True) is True
