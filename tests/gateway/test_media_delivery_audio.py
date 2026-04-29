from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SessionSource, MessageEvent, MessageType
from gateway.run import GatewayRunner


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


def _make_event(chat_id: str = "123", thread_id: str | None = None) -> MessageEvent:
    source = SessionSource(chat_id=chat_id, user_id="user1", platform=SimpleNamespace(value="discord"))
    source.thread_id = thread_id
    return MessageEvent(text="", message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_audio_media_without_voice_directive_is_sent_as_document():
    runner = _make_runner()
    event = _make_event(thread_id="thread-1")
    adapter = SimpleNamespace(
        name="discord",
        extract_media=lambda response: ([("/tmp/plain.ogg", False)], response),
        extract_images=lambda response: ([], response),
        extract_local_files=lambda response: ([], response),
        send_voice=AsyncMock(),
        send_document=AsyncMock(),
        send_video=AsyncMock(),
        send_image_file=AsyncMock(),
    )

    await GatewayRunner._deliver_media_from_response(runner, "MEDIA:/tmp/plain.ogg", event, adapter)

    adapter.send_document.assert_awaited_once_with(
        chat_id="123",
        file_path="/tmp/plain.ogg",
        metadata={"thread_id": "thread-1"},
    )
    adapter.send_voice.assert_not_called()


@pytest.mark.asyncio
async def test_audio_media_with_voice_directive_is_sent_as_voice():
    runner = _make_runner()
    event = _make_event()
    adapter = SimpleNamespace(
        name="discord",
        extract_media=lambda response: ([("/tmp/voice.ogg", True)], response),
        extract_images=lambda response: ([], response),
        extract_local_files=lambda response: ([], response),
        send_voice=AsyncMock(),
        send_document=AsyncMock(),
        send_video=AsyncMock(),
        send_image_file=AsyncMock(),
    )

    await GatewayRunner._deliver_media_from_response(runner, "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg", event, adapter)

    adapter.send_voice.assert_awaited_once_with(
        chat_id="123",
        audio_path="/tmp/voice.ogg",
        metadata=None,
    )
    adapter.send_document.assert_not_called()
