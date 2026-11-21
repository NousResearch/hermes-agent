"""What a relay delivery's running chat turn consumes before the durable FIFO (#156411).

``tests/gateway/test_relay_authority.py`` asserts a follow-up sent behind a running relay turn is
admitted to the FIFO, which holds only for a follow-up the turn does not consume. It pins
``busy_input_mode: queue`` for that; the consumed cases are covered here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_ingress_live_turn import answers_live_turn, was_consumed


def _running_turn(message_id, *, busy_input_mode, pending_reply=None):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="live-chat", chat_type="dm", user_id="u1")
    event = MessageEvent(text="one more detail", message_type=MessageType.TEXT, source=source,
                         message_id=message_id)
    key = "telegram:live-chat"
    runner = object.__new__(GatewayRunner)
    state = SimpleNamespace(turn=SimpleNamespace(started_ts=100.0, agent=object()))
    runner._draining = False
    runner._peek_session_state = lambda _key: state
    runner._effective_busy_input_mode = lambda _source: busy_input_mode
    runner._effective_busy_text_mode = lambda _source: "queue" if busy_input_mode == "queue" else "interrupt"
    runner._delivery_adapter_for = lambda _source: object()
    runner._session_key_for_source = lambda _source: key
    runner._is_session_running = lambda _key: True
    runner._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=pending_reply)
    runner._resolve_busy_steer_or_redirect = AsyncMock(
        return_value=SimpleNamespace(steered=True, redirected=False))
    runner._busy_interrupt_and_ack = AsyncMock()
    return runner, event


@pytest.mark.asyncio
async def test_queue_mode_followup_is_left_for_the_fifo():
    runner, event = _running_turn("live-queue-1", busy_input_mode="queue")

    assert await answers_live_turn(runner, event) is False

    runner._resolve_busy_steer_or_redirect.assert_not_awaited()
    assert not was_consumed(event)


@pytest.mark.asyncio
async def test_steer_mode_followup_is_consumed_and_its_resend_recognized():
    runner, event = _running_turn("live-steer-1", busy_input_mode="steer")

    assert await answers_live_turn(runner, event) is True

    runner._resolve_busy_steer_or_redirect.assert_awaited_once()
    assert was_consumed(event)  # a relay re-send of it (lost ACK) is dropped, not admitted twice


@pytest.mark.asyncio
async def test_pending_prompt_reply_is_consumed_and_its_resend_recognized():
    runner, event = _running_turn("live-reply-1", busy_input_mode="queue", pending_reply="Got it.")

    assert await answers_live_turn(runner, event) is True

    assert event._consumer_reply == "Got it."
    runner._resolve_busy_steer_or_redirect.assert_not_awaited()
    assert was_consumed(event)
