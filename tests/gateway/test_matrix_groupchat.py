"""Matrix normalization required by the transport-independent Groupchat plugin."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
@pytest.mark.parametrize("native", [False, True])
async def test_peer_typing_reaches_conversation_policy(native):
    adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "@agent:example.test"}))
    middleware = SimpleNamespace(delays_messages=True, typing=MagicMock())
    adapter._conversation_middleware = middleware
    adapter._client = SimpleNamespace(mxid="@agent:example.test")

    typing_event = SimpleNamespace(
        room_id="!room:example.test",
        content={"user_ids": ["@peer:example.test"]},
    )
    if native:
        from mautrix.types import EphemeralEvent
        typing_event = EphemeralEvent.deserialize({
            "type": "m.typing", "room_id": str(typing_event.room_id),
            "content": typing_event.content,
        })
    await adapter._on_typing(typing_event)

    middleware.typing.assert_called_once_with("!room:example.test")


@pytest.mark.anyio
@pytest.mark.parametrize("native", [False, True])
async def test_own_only_typing_is_ignored(native):
    adapter = MatrixAdapter(PlatformConfig(extra={"user_id": "@agent:example.test"}))
    middleware = SimpleNamespace(delays_messages=True, typing=MagicMock())
    adapter._conversation_middleware = middleware
    adapter._client = SimpleNamespace(mxid="@agent:example.test")

    typing_event = SimpleNamespace(
        room_id="!room:example.test",
        content={"user_ids": ["@agent:example.test"]},
    )
    if native:
        from mautrix.types import EphemeralEvent
        typing_event = EphemeralEvent.deserialize({
            "type": "m.typing", "room_id": str(typing_event.room_id),
            "content": typing_event.content,
        })
    await adapter._on_typing(typing_event)

    middleware.typing.assert_not_called()
