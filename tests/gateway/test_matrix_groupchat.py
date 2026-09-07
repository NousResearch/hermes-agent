"""Matrix normalization required by the transport-independent Groupchat plugin."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.session import SessionSource
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_voice_filename_without_msc3245_is_classified_as_voice():
    kind, mime, is_voice = MatrixAdapter._classify_inbound_media(
        "m.audio",
        "audio/ogg",
        {"filename": "voice_message.ogg"},
    )
    assert kind is MessageType.VOICE
    assert mime == "audio/ogg"
    assert is_voice


def test_ordinary_audio_filename_remains_audio_attachment():
    kind, _mime, is_voice = MatrixAdapter._classify_inbound_media(
        "m.audio",
        "audio/mpeg",
        {"filename": "meeting.mp3"},
    )
    assert kind is MessageType.AUDIO
    assert not is_voice


@pytest.mark.anyio
async def test_normalized_event_carries_native_mention_signal():
    adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "@agent:example.test"}))
    source = SessionSource(
        platform=adapter.platform,
        chat_id="!room:example.test",
        chat_type="group",
        user_id="@human:example.test",
    )
    adapter._resolve_message_context = AsyncMock(
        return_value=("hello", False, "group", None, "Human", source)
    )
    adapter._extract_reply_context = AsyncMock(
        return_value=("hello", None, None, None, None)
    )
    adapter._is_bot_mentioned = MagicMock(return_value=True)

    event = await adapter._build_inbound_event(
        "!room:example.test",
        "@human:example.test",
        "$event",
        "hello",
        {
            "body": "hello @agent:example.test",
            "m.mentions": {"user_ids": ["@agent:example.test"]},
        },
        {},
    )

    assert event is not None
    assert event.metadata["conversation_mentioned"] is True


@pytest.mark.anyio
async def test_peer_typing_reaches_conversation_policy():
    adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "@agent:example.test"}))
    middleware = SimpleNamespace(delays_messages=True, typing=MagicMock())
    adapter._conversation_middleware = middleware
    adapter._client = SimpleNamespace(mxid="@agent:example.test")

    await adapter._on_typing(SimpleNamespace(
        room_id="!room:example.test",
        content={"user_ids": ["@peer:example.test"]},
    ))

    middleware.typing.assert_called_once_with("!room:example.test")


@pytest.mark.anyio
async def test_own_only_typing_is_ignored():
    adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "@agent:example.test"}))
    middleware = SimpleNamespace(delays_messages=True, typing=MagicMock())
    adapter._conversation_middleware = middleware
    adapter._client = SimpleNamespace(mxid="@agent:example.test")

    await adapter._on_typing(SimpleNamespace(
        room_id="!room:example.test",
        content={"user_ids": ["@agent:example.test"]},
    ))

    middleware.typing.assert_not_called()
