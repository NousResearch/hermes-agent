"""Test that Telegram voice messages set the correct MessageType for STT transcription.

Related issue: #16185 - Telegram voice messages not transcribed automatically.

The gateway's STT pipeline (gateway/run.py) routes voice/audio messages through
_enrich_message_with_transcription only when event.message_type is MessageType.VOICE
or MessageType.AUDIO. Without this, voice messages are cached but never transcribed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.platforms.base import MessageType, Platform


@pytest.fixture
def telegram_adapter():
    """Create a TelegramAdapter instance for testing."""
    with patch("plugins.platforms.telegram.adapter.TelegramAdapter.__init__", return_value=None):
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.platform = Platform.TELEGRAM
        adapter._max_doc_bytes = 20 * 1024 * 1024
        adapter._telegram_media_size_allowed = MagicMock(return_value=(True, None))
        return adapter


def test_voice_message_sets_message_type(telegram_adapter):
    """Verify that voice messages set event.message_type to MessageType.VOICE."""
    msg = MagicMock()
    msg.sticker = None
    msg.photo = None
    msg.video = None
    msg.audio = None
    msg.voice = MagicMock()
    msg.voice.file_size = 100000
    msg.document = None
    msg.caption = ""

    message_type = telegram_adapter._media_message_type(msg)
    assert message_type == MessageType.VOICE, (
        f"Expected MessageType.VOICE for voice message, got {message_type}"
    )


def test_audio_file_sets_message_type(telegram_adapter):
    """Verify that audio file messages set event.message_type to MessageType.AUDIO."""
    msg = MagicMock()
    msg.sticker = None
    msg.photo = None
    msg.video = None
    msg.audio = MagicMock()
    msg.audio.file_size = 200000
    msg.voice = None
    msg.document = None
    msg.caption = ""

    message_type = telegram_adapter._media_message_type(msg)
    assert message_type == MessageType.AUDIO, (
        f"Expected MessageType.AUDIO for audio file, got {message_type}"
    )


def test_media_type_precedence(telegram_adapter):
    """Verify that audio takes precedence over voice when both are present."""
    msg = MagicMock()
    msg.sticker = None
    msg.photo = None
    msg.video = None
    msg.audio = MagicMock()
    msg.voice = MagicMock()
    msg.document = None

    message_type = telegram_adapter._media_message_type(msg)
    # _media_message_type checks audio before voice, so audio wins.
    assert message_type == MessageType.AUDIO, (
        f"Expected MessageType.AUDIO (audio checked before voice), got {message_type}"
    )
