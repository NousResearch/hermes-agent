"""Invariants for the gateway's confirmed-delivery observer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Adapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.result = result
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return self.result

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_post_delivery_observer_is_ordered_after_confirmed_final_delivery(streaming):
    """Both final-delivery paths defer the observer until the adapter releases it."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner(GatewayConfig())
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    adapter = _Adapter(SendResult(success=True, message_id="out-9"))

    hook_ctx = {
        "platform": "telegram", "user_id": "user-1", "chat_id": "chat-1", "thread_id": "thread-1",
        "session_id": "session-1", "message": "inbound",
    }

    await runner._hmwa_post_turn_hooks(
        hook_ctx, {"model": "m", "provider": "p"}, "final", adapter=adapter,
        session_key="session-key", message_id="out-9", streaming=streaming,
    )

    assert [call.args[0] for call in runner.hooks.emit.await_args_list] == ["agent:end"]
    if not streaming:
        adapter._post_delivery_receipts["session-key"] = adapter.result
    callback = adapter.pop_post_delivery_callback("session-key")
    await callback()
    assert [call.args[0] for call in runner.hooks.emit.await_args_list] == ["agent:end", "agent:post_delivery"]
    payload = runner.hooks.emit.await_args_list[-1].args[1]
    assert payload == {
        "platform": "telegram", "user_id": "user-1", "chat_id": "chat-1", "thread_id": "thread-1",
        "session_id": "session-1", "message_id": "out-9", "response": "final", "model": "m",
        "provider": "p", "delivery_confirmed": True,
    }


@pytest.mark.asyncio
async def test_post_delivery_observer_skips_failed_final_send():
    """A failed final transport send cannot produce a confirmed-delivery event."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner(GatewayConfig())
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    adapter = _Adapter(SendResult(success=False, error="offline"))
    adapter._post_delivery_receipts["session-key"] = adapter.result
    await runner._hmwa_post_turn_hooks(
        {"platform": "telegram", "user_id": "u", "chat_id": "c", "thread_id": "", "session_id": "s", "message": "in"},
        {"model": "m", "provider": "p"}, "final", adapter=adapter, session_key="session-key",
    )

    await adapter.pop_post_delivery_callback("session-key")()
    assert [call.args[0] for call in runner.hooks.emit.await_args_list] == ["agent:end"]
