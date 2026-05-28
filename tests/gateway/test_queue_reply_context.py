"""Tests for reply context propagation through /queue and /steer event construction.

When a user replies to a message and uses /queue or /steer, the queued
MessageEvent must carry reply_to_message_id and reply_to_text so that
_prepare_inbound_message_text can inject the [Replying to: "..."] prefix
when the turn eventually runs.

Regression for: t_1cc95bc9 — /queue drops reply_to_message_id + reply_to_text

IMPORTANT: The core tests exercise _handle_message() end-to-end so that
reverting the 6-line fix in gateway/run.py would make them fail.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_steer_command.py pattern)
# ---------------------------------------------------------------------------

def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, **kwargs) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
        **kwargs,
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner, adapter


def _session_entry() -> SessionEntry:
    source = _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


# ---------------------------------------------------------------------------
# Integration tests: _handle_message() exercises the changed lines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queue_via_handle_message_preserves_reply_to_message_id():
    """/queue routed through _handle_message() must propagate reply_to_message_id.

    This test exercises the 6 lines added in gateway/run.py. Reverting them
    would cause this test to fail because the queued event would have
    reply_to_message_id=None instead of '999'.
    """
    runner, adapter = _make_runner()
    sk = build_session_key(_make_source())

    # Simulate an agent actively running so /queue is processed.
    running_agent = MagicMock()
    runner._running_agents[sk] = running_agent

    inbound = _make_event(
        "/queue this is approved",
        reply_to_message_id="999",
        reply_to_text="Task t_8698e650 is ready for your review.",
    )

    result = await runner._handle_message(inbound)

    # Must acknowledge the queue.
    assert result is not None
    assert "queued" in result.lower()

    # The fix: reply context must flow onto the queued event.
    assert sk in adapter._pending_messages, (
        "No queued event stored — /queue handler didn't call _enqueue_fifo"
    )
    queued: MessageEvent = adapter._pending_messages[sk]
    assert queued.reply_to_message_id == "999", (
        f"reply_to_message_id not propagated: got {queued.reply_to_message_id!r}"
    )
    assert queued.reply_to_text == "Task t_8698e650 is ready for your review.", (
        f"reply_to_text not propagated: got {queued.reply_to_text!r}"
    )


@pytest.mark.asyncio
async def test_queue_via_handle_message_no_reply_context_stays_none():
    """/queue with no reply context must NOT invent reply fields."""
    runner, adapter = _make_runner()
    sk = build_session_key(_make_source())

    running_agent = MagicMock()
    runner._running_agents[sk] = running_agent

    inbound = _make_event("/queue plain follow-up")

    result = await runner._handle_message(inbound)

    assert result is not None
    assert "queued" in result.lower()
    assert sk in adapter._pending_messages
    queued: MessageEvent = adapter._pending_messages[sk]
    assert queued.reply_to_message_id is None
    assert queued.reply_to_text is None


@pytest.mark.asyncio
async def test_steer_pending_sentinel_via_handle_message_preserves_reply_context():
    """/steer when agent is PENDING-sentinel must preserve reply context on fallback.

    This exercises the second /steer path added in gateway/run.py.
    """
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner, adapter = _make_runner()
    sk = build_session_key(_make_source())

    # Sentinel = agent booting, /steer falls back to pending-slot queue
    runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

    inbound = _make_event(
        "/steer also check auth.log",
        reply_to_message_id="42",
        reply_to_text="PR approved, please merge.",
    )

    result = await runner._handle_message(inbound)

    assert result is not None
    assert sk in adapter._pending_messages
    queued: MessageEvent = adapter._pending_messages[sk]
    assert queued.reply_to_message_id == "42"
    assert queued.reply_to_text == "PR approved, please merge."


@pytest.mark.asyncio
async def test_steer_no_active_agent_via_handle_message_preserves_reply_context():
    """/steer when no agent is active must preserve reply context on fallback.

    This exercises the third /steer path added in gateway/run.py.
    """
    runner, adapter = _make_runner()
    sk = build_session_key(_make_source())

    # spec=[] → hasattr(agent, "steer") returns False → triggers fallback
    running_agent = MagicMock(spec=[])
    runner._running_agents[sk] = running_agent

    inbound = _make_event(
        "/steer focus on error handling",
        reply_to_message_id="77",
        reply_to_text="Test output attached.",
    )

    result = await runner._handle_message(inbound)

    assert result is not None
    assert "queued" in result.lower()
    assert sk in adapter._pending_messages
    queued: MessageEvent = adapter._pending_messages[sk]
    assert queued.reply_to_message_id == "77"
    assert queued.reply_to_text == "Test output attached."


# ---------------------------------------------------------------------------
# Unit tests: _prepare_inbound_message_text injects the prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queued_event_with_reply_context_gets_prefix_injected():
    """End-to-end: a queued event carrying reply_to_text produces the
    [Replying to: "..."] prefix when _prepare_inbound_message_text runs."""
    runner, _ = _make_runner()
    source = _make_source()

    queued_event = MessageEvent(
        text="this is approved",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m3",
        reply_to_message_id="999",
        reply_to_text="Task t_8698e650 is ready for your review.",
    )

    result = await runner._prepare_inbound_message_text(
        event=queued_event,
        source=source,
        history=[],
    )

    assert result is not None
    assert '[Replying to: "Task t_8698e650 is ready for your review."]' in result
    assert result.endswith("this is approved")


@pytest.mark.asyncio
async def test_queued_event_without_reply_context_has_no_prefix():
    """A queued event with no reply context must NOT get a spurious prefix."""
    runner, _ = _make_runner()
    source = _make_source()

    queued_event = MessageEvent(
        text="plain queued message",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m4",
    )

    result = await runner._prepare_inbound_message_text(
        event=queued_event,
        source=source,
        history=[],
    )

    assert result == "plain queued message"
