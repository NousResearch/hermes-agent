"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
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


class _LocalMediaAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform("napcat")):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.send_image_file = AsyncMock(
            return_value=SendResult(success=True, message_id="image")
        )
        self.send_video = AsyncMock(
            return_value=SendResult(success=True, message_id="video")
        )
        self.send_voice = AsyncMock(
            return_value=SendResult(success=True, message_id="voice")
        )
        self.send_document = AsyncMock(
            return_value=SendResult(success=True, message_id="doc")
        )

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


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


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


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
        _fake_runner({"thread_id": "topic-1"}),
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
        _fake_runner({"thread_id": "topic-1"}),
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
        _fake_runner({"thread_id": "topic-1"}),
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


@pytest.mark.asyncio
async def test_streaming_delivery_routes_napcat_group_image_to_original_group(tmp_path):
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"fake png")
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:610066383",
        chat_type="group",
        user_id="111",
        user_name="Alice",
    )
    event = MessageEvent(
        text="画一张图",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    adapter = _LocalMediaAdapter(Platform("napcat"))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        f"画好了\nMEDIA:{image_path}",
        event,
        adapter,
    )

    adapter.send_image_file.assert_awaited_once_with(
        chat_id="group:610066383",
        image_path=str(image_path),
        caption=None,
        metadata=None,
    )
    adapter.send_video.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_napcat_private_video_to_original_private_chat(tmp_path):
    video_path = tmp_path / "result.mp4"
    video_path.write_bytes(b"fake mp4")
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
        user_name="Dad",
    )
    event = MessageEvent(
        text="做一个视频",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )
    adapter = _LocalMediaAdapter(Platform("napcat"))

    await GatewayRunner._deliver_media_from_response(
        _fake_runner(None),
        f"视频好了\nMEDIA:{video_path}",
        event,
        adapter,
    )

    adapter.send_video.assert_awaited_once_with(
        chat_id="12345",
        video_path=str(video_path),
        metadata=None,
    )
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
