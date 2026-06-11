"""Active-session bypass tests for protected sensitive input."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id: str):
        return {}


@pytest.mark.asyncio
async def test_pending_secret_bypasses_active_session_queue(monkeypatch):
    from tools import secret_capture_gateway

    adapter = _Adapter(PlatformConfig(enabled=True, extra={}), Platform.TELEGRAM)
    adapter._message_handler = AsyncMock(return_value="")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="42",
        user_name="Tester",
    )
    event = MessageEvent(
        text="/looks-like-command-but-is-secret",
        message_type=MessageType.COMMAND,
        source=source,
    )
    session_key = build_session_key(source, group_sessions_per_user=True, thread_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()
    entry = secret_capture_gateway.register("sid-active", session_key, "MY_SECRET", "Send it")
    entry.bind_source(source)
    try:
        await adapter.handle_message(event)
    finally:
        secret_capture_gateway.clear_session(session_key)

    adapter._message_handler.assert_awaited_once_with(event)
    assert session_key not in adapter._pending_messages
    assert entry.value is None  # Base adapter only routes; GatewayRunner resolves.
