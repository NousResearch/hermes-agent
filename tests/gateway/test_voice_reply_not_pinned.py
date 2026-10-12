"""Regression tests for #126865: the auto voice reply must not pin the turn.

With voice.auto_tts on, _hmwa_deliver_turn_response awaited _send_voice_reply,
which awaits the TTS synthesis in a worker thread. When the edge-tts provider
wedges, the tool's executor join blocks past its own timeout with no upper
bound, so the final text was never delivered, subsequent messages queued
behind the busy slot forever, and the durable active_turn_token stayed set
until /new or a gateway restart. The voice reply is now detached: the call
must spawn the task and return without awaiting synthesis.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner_with_adapter() -> tuple:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._background_tasks = set()
    adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    return runner, adapter


def _event() -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123",
                             user_id="u1", user_name="User"),
        message_type=MessageType.TEXT,
        message_id="456",
    )


@pytest.mark.asyncio
async def test_voice_reply_delivery_does_not_await_synthesis():
    """A synthesis that never returns must not block the delivery coroutine.

    Regression for #126865: the old code awaited _send_voice_reply inline, so a
    wedged TTS worker held the turn (final text + busy slot + active-turn
    marker) for hours with no error logged. Now the spawner returns while
    synthesis is still pending.
    """
    runner, adapter = _runner_with_adapter()
    gate = asyncio.Event()
    started = asyncio.Event()

    async def hung_tts(*_a, **_k):
        started.set()
        await gate.wait()

    runner._send_voice_reply = AsyncMock(side_effect=hung_tts)
    event = _event()

    runner._spawn_voice_reply_task(event, "reply text")

    # The spawner returned; give the detached task a chance to start.
    await asyncio.wait_for(started.wait(), timeout=2)
    assert runner._send_voice_reply.await_count == 1
    assert len(runner._background_tasks) == 1

    # The delivery coroutine is free; only the detached task is parked.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gate.wait(), timeout=0.05)
    gate.set()


@pytest.mark.asyncio
async def test_deliver_turn_response_spawns_voice_reply_and_returns_text():
    """_hmwa_deliver_turn_response hands the voice reply to a task and still
    returns the text for the adapter to send."""
    runner, adapter = _runner_with_adapter()
    runner._streaming_tts_turn_completed_gate = None

    event = _event()
    runner._should_send_voice_reply = MagicMock(return_value=True)
    runner._spawn_voice_reply_task = MagicMock()
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)

    delivered = await GatewayRunner._hmwa_deliver_turn_response(
        runner, event, event.source, None, "sess-key", 1,
        {}, [], "final reply", None, False,
    )

    assert delivered == "final reply"
    runner._spawn_voice_reply_task.assert_called_once_with(event, "final reply")
