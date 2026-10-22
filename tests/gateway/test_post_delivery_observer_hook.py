"""Invariants for the gateway's confirmed-delivery observer."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.turn_context import TurnContext


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


def _hook_context():
    return {
        "platform": "telegram", "user_id": "user-1", "chat_id": "chat-1",
        "thread_id": "thread-1", "session_id": "session-1", "message": "inbound",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("send_result", "expected_events"),
    [
        (SendResult(success=True, message_id="out-9"), ["agent:end", "agent:post_delivery"]),
        (SendResult(success=False, error="offline"), ["agent:end"]),
    ],
)
async def test_nonstream_observer_follows_real_final_send_and_generation(
    send_result, expected_events,
):
    """Only a confirmed real-path final releases the generation-owned observer."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner(GatewayConfig())
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    adapter = _Adapter(send_result)
    source = adapter.build_source("chat-1", user_id="user-1", thread_id="thread-1")
    event = MessageEvent(text="inbound", source=source, message_id="in-4")
    session_key, run_generation = "session-key", 7
    interrupt_event = asyncio.Event()
    interrupt_event._hermes_run_generation = run_generation
    adapter._active_sessions[session_key] = interrupt_event

    async def handler(_event):
        await runner._hmwa_post_turn_hooks(
            _hook_context(), {"model": "m", "provider": "p"}, "final",
            adapter=adapter, session_key=session_key, run_generation=run_generation,
        )
        return "final"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, session_key)

    assert adapter.sent[0] == "final"
    assert [call.args[0] for call in runner.hooks.emit.await_args_list] == expected_events
    if send_result.success:
        assert runner.hooks.emit.await_args_list[-1].args[1]["message_id"] == "out-9"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_stream_observer_requires_confirmed_receipt_and_uses_outbound_id(confirmed):
    """Only this task's confirmed stream emits, never a stale success from an earlier lane."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner(GatewayConfig())
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    adapter = _Adapter(SendResult(success=True, message_id="unused"))
    runner.adapters[Platform.TELEGRAM] = adapter
    source = adapter.build_source("chat-1", user_id="user-1", thread_id="thread-1")
    session_key, run_generation = "session-key", 11
    consumer = SimpleNamespace(
        final_response_sent=confirmed,
        final_content_delivered=False,
        message_id="out-stream-3",
        delivered_final_matches=lambda text: text == "final",
    )
    result = {"final_response": "final", "model": "m", "provider": "p"}
    turn_ctx = TurnContext(
        source=source, session_key=session_key, run_generation=run_generation,
        stream_consumer_holder=[consumer],
    )
    interrupt_event = asyncio.Event()
    interrupt_event._hermes_run_generation = run_generation
    adapter._active_sessions[session_key] = interrupt_event
    if not confirmed:
        adapter.record_post_delivery_receipt(
            session_key, SendResult(success=True, message_id="stale-queued-final"))

    async def handler(_event):
        await runner._run_agent_mark_streamed_delivery(result, turn_ctx)
        await runner._hmwa_post_turn_hooks(
            _hook_context(), result, "final", adapter=adapter, session_key=session_key,
            run_generation=run_generation,
        )

    adapter.set_message_handler(handler)
    await adapter._process_message_background(
        MessageEvent(text="inbound", source=source, message_id="in-4"), session_key)

    events = [call.args[0] for call in runner.hooks.emit.await_args_list]
    assert events == (["agent:end", "agent:post_delivery"] if confirmed else ["agent:end"])
    if confirmed:
        payload = runner.hooks.emit.await_args_list[-1].args[1]
        assert payload["message_id"] == "out-stream-3"
        assert payload["delivery_confirmed"] is True
