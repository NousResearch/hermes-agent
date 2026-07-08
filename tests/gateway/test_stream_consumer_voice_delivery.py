"""Regression test for Telegram streaming voice message delivery.

When streaming is enabled, [[audio_as_voice]] + MEDIA: tags must be
delivered via adapter.send_voice, not as plain text in the streamed
message. This ensures the audio appears as a voice bubble on Telegram
instead of a file attachment.

Issue: #60556
"""

import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.fixture
def mock_adapter():
    """Mock platform adapter with send_voice support."""
    adapter = MagicMock()
    adapter.message_len_fn = len
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="msg123"))
    adapter.send_voice = AsyncMock()
    return adapter


@pytest.fixture
def consumer(mock_adapter):
    """Create a stream consumer for testing."""
    return GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(edit_interval=0.1, buffer_only=True),
    )


@pytest.mark.asyncio
async def test_voice_media_extraction_from_stream(consumer, mock_adapter):
    """Verify [[audio_as_voice]] + MEDIA: paths are extracted and sent via send_voice."""
    # Stream content with voice directive
    voice_text = "[[audio_as_voice]]\nMEDIA:/path/to/audio.ogg\nHere is the response."

    # Simulate streaming
    consumer.on_delta(voice_text)
    consumer.finish()

    # Run the consumer
    await consumer.run()

    # Verify send_voice was called with the correct path
    mock_adapter.send_voice.assert_called_once()
    call_kwargs = mock_adapter.send_voice.call_args.kwargs
    assert call_kwargs["chat_id"] == "test_chat"
    assert call_kwargs["audio_path"] == "/path/to/audio.ogg"
    # reply_to may be None (first message) or msg123 (if message was already sent)

    # Verify the voice directive was stripped from the streamed text
    mock_adapter.send.assert_called_once()
    sent_text = mock_adapter.send.call_args.kwargs["content"]
    assert "[[audio_as_voice]]" not in sent_text
    assert "MEDIA:/path/to/audio.ogg" not in sent_text
    assert "Here is the response." in sent_text


@pytest.mark.asyncio
async def test_multiple_voice_media_paths(consumer, mock_adapter):
    """Verify multiple voice messages are sent correctly."""
    # Stream content with multiple voice directives
    voice_text = (
        "[[audio_as_voice]]\nMEDIA:/path/to/audio1.ogg\n"
        "First response.\n"
        "[[audio_as_voice]]\nMEDIA:/path/to/audio2.ogg\n"
        "Second response."
    )

    # Simulate streaming
    consumer.on_delta(voice_text)
    consumer.finish()

    # Run the consumer
    await consumer.run()

    # Verify send_voice was called twice
    assert mock_adapter.send_voice.call_count == 2

    # Verify both paths were sent
    call_args_list = [call.kwargs["audio_path"] for call in mock_adapter.send_voice.call_args_list]
    assert "/path/to/audio1.ogg" in call_args_list
    assert "/path/to/audio2.ogg" in call_args_list


@pytest.mark.asyncio
async def test_voice_media_without_send_voice_support():
    """Verify graceful handling when adapter lacks send_voice."""
    # Mock adapter without send_voice
    adapter = MagicMock()
    adapter.message_len_fn = len
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="msg123"))
    # Intentionally no send_voice method

    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(edit_interval=0.1, buffer_only=True),
    )

    # Stream content with voice directive
    voice_text = "[[audio_as_voice]]\nMEDIA:/path/to/audio.ogg\nResponse."

    # Simulate streaming
    consumer.on_delta(voice_text)
    consumer.finish()

    # Run the consumer (should not raise)
    await consumer.run()

    # Verify the text was still sent (directive stripped)
    adapter.send.assert_called_once()
    sent_text = adapter.send.call_args.kwargs["content"]
    assert "[[audio_as_voice]]" not in sent_text
    assert "Response." in sent_text


def test_extract_voice_media_paths_single(consumer):
    """Verify _extract_voice_media_paths extracts a single path."""
    text = "[[audio_as_voice]]\nMEDIA:/path/to/audio.ogg\nRest of message."
    clean_text, paths = consumer._extract_voice_media_paths(text)

    assert "/path/to/audio.ogg" in paths
    assert "[[audio_as_voice]]" not in clean_text
    assert "MEDIA:/path/to/audio.ogg" not in clean_text
    assert "Rest of message." in clean_text


def test_extract_voice_media_paths_multiple(consumer):
    """Verify _extract_voice_media_paths extracts multiple paths."""
    text = (
        "[[audio_as_voice]]\nMEDIA:/path1.ogg\n"
        "Text between.\n"
        "[[audio_as_voice]]\nMEDIA:/path2.ogg\n"
        "End."
    )
    clean_text, paths = consumer._extract_voice_media_paths(text)

    assert "/path1.ogg" in paths
    assert "/path2.ogg" in paths
    assert "[[audio_as_voice]]" not in clean_text
    assert "MEDIA:" not in clean_text
    assert "Text between." in clean_text
    assert "End." in clean_text


def test_extract_voice_media_paths_none(consumer):
    """Verify _extract_voice_media_paths handles text without directives."""
    text = "Just plain text with no voice directives."
    clean_text, paths = consumer._extract_voice_media_paths(text)

    assert len(paths) == 0
    assert clean_text == text


def test_extract_voice_media_paths_whitespace_handling(consumer):
    """Verify _extract_voice_media_paths handles whitespace correctly."""
    text = "[[audio_as_voice]]\n  MEDIA:  /path/to/audio.ogg  \nEnd."
    clean_text, paths = consumer._extract_voice_media_paths(text)

    # Path should be stripped
    assert "/path/to/audio.ogg" in paths
    # Original whitespace preserved in clean text (directive removed)
    assert "[[audio_as_voice]]" not in clean_text
    assert "End." in clean_text