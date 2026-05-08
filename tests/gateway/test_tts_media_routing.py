"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner, _collect_media_tags_from_tool_results
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _FeedbackStore:
    def __init__(self):
        self.messages = []

    def get_or_create_session(self, source):
        return SimpleNamespace(session_id="sid-1")

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.messages.append((session_id, message))


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def test_tool_result_media_collection_ignores_napcat_history_media_examples():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_napcat",
                    "type": "function",
                    "function": {"name": "napcat_call", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "napcat_call",
            "tool_call_id": "call_napcat",
            "content": (
                '{"messages":[{"message":"old example MEDIA:https://example.com/meme.png '
                '[[audio_as_voice]] MEDIA:/tmp/voice.ogg"}]}'
            ),
        },
    ]

    assert _collect_media_tags_from_tool_results(messages) == []


def test_tool_result_media_collection_allows_text_to_speech_media_tags():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_tts",
                    "type": "function",
                    "function": {"name": "text_to_speech", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tts",
            "content": '{"success":true,"media_tag":"[[audio_as_voice]]\\nMEDIA:/tmp/speech.ogg"}',
        },
    ]

    assert _collect_media_tags_from_tool_results(messages) == [
        "[[audio_as_voice]]",
        "MEDIA:/tmp/speech.ogg",
    ]


@pytest.mark.asyncio
async def test_base_adapter_routes_telegram_flac_media_tag_to_document_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/speech.flac")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.flac",
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_non_voice_telegram_ogg_media_tag_to_document_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/speech.ogg")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.ogg",
        metadata=None,
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender():
    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(
        return_value="[[audio_as_voice]]\nMEDIA:/tmp/speech.ogg"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path="/tmp/speech.ogg",
        metadata=None,
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_persists_media_delivery_failure_feedback():
    adapter = _MediaRoutingAdapter()
    store = _FeedbackStore()
    adapter.set_session_store(store)
    event = _event()
    adapter._message_handler = AsyncMock(return_value="MEDIA:/tmp/missing.ogg")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=False, error="BAD_FILE")
    )

    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(store.messages) == 1
    _, message = store.messages[0]
    assert message["role"] == "user"
    assert "[Hermes gateway delivery feedback]" in message["content"]
    assert "/tmp/missing.ogg" in message["content"]
    assert "BAD_FILE" in message["content"]


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_flac_media_tag_to_document_sender():
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        object(),
        "MEDIA:/tmp/speech.flac",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.flac",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_persists_media_delivery_failure_feedback():
    event = _event(thread_id="topic-1")
    store = _FeedbackStore()
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=False, error="BAD_FILE")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        SimpleNamespace(session_store=store),
        "MEDIA:/tmp/speech.flac",
        event,
        adapter,
    )

    assert len(store.messages) == 1
    _, message = store.messages[0]
    assert message["role"] == "user"
    assert "[Hermes gateway delivery feedback]" in message["content"]
    assert "/tmp/speech.flac" in message["content"]
    assert "BAD_FILE" in message["content"]


@pytest.mark.asyncio
async def test_streaming_delivery_routes_non_voice_telegram_ogg_media_tag_to_document_sender():
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        object(),
        "MEDIA:/tmp/speech.ogg",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path="/tmp/speech.ogg",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_mp3_media_tag_to_voice_sender():
    """MP3 audio on Telegram must go through send_voice (which routes to
    sendAudio internally); Telegram accepts MP3 for the audio player."""
    event = _event(thread_id="topic-1")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        object(),
        "MEDIA:/tmp/speech.mp3",
        event,
        adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path="/tmp/speech.mp3",
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_not_awaited()
