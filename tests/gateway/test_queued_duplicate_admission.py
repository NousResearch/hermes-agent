"""Regression tests for duplicate admission while a gateway turn is active."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


ACTIVE_MESSAGE_ID = "1712345678.000100"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK, chat_id="D123", chat_type="dm", user_id="U123"
    )


def _event(message_id: str | None, text: str = "follow-up") -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=_source(), message_id=message_id
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="test-token")}
    )
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._queued_events = {}
    runner._pending_approvals = {}
    runner.session_store = MagicMock()
    runner.session_store.peek_session_id.return_value = None
    runner.session_store.has_platform_message_id.return_value = False
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True

    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.SLACK
    runner.adapters = {Platform.SLACK: adapter}

    active = _event(ACTIVE_MESSAGE_ID, "original request")
    session_key = build_session_key(active.source)
    agent = MagicMock()
    agent._active_children = []
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
    state = runner._session_state(session_key)
    state.turn.agent = agent
    state.turn.event = active
    state.turn.started_ts = time.time()
    return GatewayRunner, runner, adapter, agent, session_key


@pytest.mark.asyncio
async def test_active_message_replay_is_not_admitted_to_pending_queue():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    replay = _event(ACTIVE_MESSAGE_ID, "original request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages == {}
    assert runner._queued_events == {}
    agent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_persisted_message_replay_is_not_admitted_to_pending_queue():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    replay = _event("already-persisted", "completed request")
    runner.session_store.peek_session_id.return_value = "session-1"
    runner.session_store.has_platform_message_id.return_value = True

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages == {}
    assert runner._queued_events == {}
    agent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_awaited()
    runner.session_store.has_platform_message_id.assert_called_once_with(
        "session-1", "already-persisted"
    )


@pytest.mark.asyncio
async def test_pending_message_replay_is_not_admitted_twice():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    pending = _event("already-pending", "queued request")
    adapter._pending_messages[session_key] = pending
    replay = _event("already-pending", "queued request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is pending
    assert runner._queued_events == {}
    agent.interrupt.assert_not_called()
    runner.session_store.peek_session_id.assert_not_called()


@pytest.mark.asyncio
async def test_overflow_message_replay_is_not_admitted_twice():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    pending = _event("pending-head", "first queued request")
    overflow = _event("already-overflowing", "second queued request")
    adapter._pending_messages[session_key] = pending
    runner._session_state(session_key).conversation.queued_events.append(overflow)
    replay = _event("already-overflowing", "second queued request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is pending
    assert runner._queued_events[session_key] == [overflow]
    agent.interrupt.assert_not_called()
    runner.session_store.peek_session_id.assert_not_called()


@pytest.mark.asyncio
async def test_persisted_lookup_failure_preserves_new_message():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    followup = _event("new-message", "new request")
    runner.session_store.peek_session_id.side_effect = RuntimeError("database unavailable")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, followup, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is followup
    agent.interrupt.assert_called_once_with("new request")


@pytest.mark.asyncio
async def test_distinct_message_is_still_admitted_as_followup():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    followup = _event("1712345678.000200", "new request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, followup, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is followup
    agent.interrupt.assert_called_once_with("new request")


@pytest.mark.asyncio
async def test_idless_message_is_preserved_fail_open():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    followup = _event(None, "synthetic request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, followup, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is followup
    agent.interrupt.assert_called_once_with("synthetic request")


@pytest.mark.asyncio
async def test_active_message_replay_does_not_disturb_existing_fifo():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    queued = _event("1712345678.000200", "new request")
    adapter._pending_messages[session_key] = queued
    replay = _event(ACTIVE_MESSAGE_ID, "original request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages[session_key] is queued
    assert runner._queued_events == {}
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_redirected_turn_uses_live_context_message_id():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    state = runner._session_state(session_key)
    state.turn.event.message_id = "opening-message"
    state.turn.ctx = SimpleNamespace(inbound_message_id=ACTIVE_MESSAGE_ID)
    replay = _event(ACTIVE_MESSAGE_ID, "redirecting request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages == {}
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_redirected_turn_still_rejects_opening_message_replay():
    GatewayRunner, runner, adapter, agent, session_key = _runner()
    runner._session_state(session_key).turn.ctx = SimpleNamespace(
        inbound_message_id="redirecting-message"
    )
    replay = _event(ACTIVE_MESSAGE_ID, "original request")

    handled = await GatewayRunner._handle_active_session_busy_message(
        runner, replay, session_key
    )

    assert handled is True
    assert adapter._pending_messages == {}
    agent.interrupt.assert_not_called()
