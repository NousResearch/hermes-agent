"""Regression tests for #75164 — /steer fallback must not overwrite FIFO head.

Uses the real gateway test fixtures (tests/gateway/test_steer_command.py) to
exercise both /steer fallback paths through _handle_message, proving that a
pre-queued Q1/Q2 survives a /steer Q3 dispatch.

Covers:
- pending-sentinel path (agent not booted yet) -> _enqueue_fifo preserves Q1
- no-steer() path (agent lacks steer()) -> _enqueue_fifo preserves Q1
"""
from __future__ import annotations

import pytest

from gateway.run import _AGENT_PENDING_SENTINEL
from tests.gateway.test_steer_command import (
    _make_event,
    _make_runner,
    _session_entry,
    _make_source,
)
from gateway.session import build_session_key
from unittest.mock import MagicMock


def _prequeue(runner, adapter, sk):
    """Pre-stage Q1 in the pending slot and Q2 in the overflow tail."""
    from gateway.platforms.base import MessageEvent, MessageType

    # Ensure _queued_events is initialized (mirrors GatewayRunner.__init__)
    if not hasattr(runner, "_queued_events"):
        runner._queued_events = {}

    q1 = MessageEvent(
        text="Q1",
        source=_make_source(),
        message_id="m1",
        channel_context="ctx1",
        message_type=MessageType.TEXT,
    )
    q2 = MessageEvent(
        text="Q2",
        source=_make_source(),
        message_id="m2",
        channel_context="ctx2",
        message_type=MessageType.TEXT,
    )
    # Q1 -> pending slot (head), Q2 -> overflow tail
    runner._enqueue_fifo(sk, q1, adapter)
    runner._enqueue_fifo(sk, q2, adapter)
    assert adapter._pending_messages[sk].text == "Q1"
    assert runner._queued_events[sk][0].text == "Q2"


@pytest.mark.asyncio
async def test_steer_pending_sentinel_preserves_fifo_head():
    """Issue #75164: /steer Q3 must not overwrite pre-queued Q1."""
    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

    _prequeue(runner, adapter, sk)

    result = await runner._handle_message(
        _make_event("/steer wait up", channel_context="ctx3")
    )
    assert result is not None
    assert "queued" in result.lower()

    # Q1 still in slot, Q2 in overflow, Q3 appended to overflow — order preserved
    assert adapter._pending_messages[sk].text == "Q1"
    overflow = runner._queued_events[sk]
    texts = [msg.text for msg in overflow]
    assert texts == ["Q2", "wait up"], f"FIFO order broken: {texts}"


@pytest.mark.asyncio
async def test_steer_no_steer_method_preserves_fifo_head():
    """Issue #75164: no-steer() fallback must not overwrite pre-queued Q1."""
    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())

    # Bare mock with NO steer() method
    runner._running_agents[sk] = MagicMock(spec=[])

    _prequeue(runner, adapter, sk)

    result = await runner._handle_message(
        _make_event("/steer fallback", channel_context="ctx3")
    )
    assert result is not None
    assert "queued" in result.lower()

    assert adapter._pending_messages[sk].text == "Q1"
    overflow = runner._queued_events[sk]
    texts = [msg.text for msg in overflow]
    assert texts == ["Q2", "fallback"], f"FIFO order broken: {texts}"
