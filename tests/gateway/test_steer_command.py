"""Tests for the gateway /steer command handler.

/steer injects a user message into the agent's next tool result without
interrupting. The gateway runner must:

  1. When an agent IS running → call ``agent.steer(text)``, do NOT set
     ``_interrupt_requested``, do NOT touch ``_pending_messages``.
  2. When the agent is the PENDING sentinel → fall back to /queue
     semantics (store in ``adapter._pending_messages``).
  3. When no agent is active → strip the slash prefix and let the normal
     prompt pipeline handle it as a regular user message.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry):
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
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
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
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


@pytest.mark.asyncio
async def test_steer_calls_agent_steer_and_does_not_interrupt():
    """When an agent is running, /steer must call agent.steer(text) and
    leave interrupt state untouched."""
    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())

    running_agent = MagicMock()
    running_agent.steer.return_value = True
    runner._running_agents[sk] = running_agent

    result = await runner._handle_message(_make_event("/steer also check auth.log"))

    # The handler replied with a confirmation
    assert result is not None
    assert "steer" in result.lower() or "queued" in result.lower()
    # The agent's steer() was called with the payload (prefix stripped)
    running_agent.steer.assert_called_once_with("also check auth.log")
    # Critically: interrupt was NOT called
    running_agent.interrupt.assert_not_called()
    # And no user-text queueing happened — the steer doesn't go into
    # _pending_messages (that would be turn-boundary /queue semantics).
    assert runner._pending_messages == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_steer_without_payload_returns_usage():
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    running_agent = MagicMock()
    runner._running_agents[sk] = running_agent

    result = await runner._handle_message(_make_event("/steer"))

    assert result is not None
    assert "Usage" in result or "usage" in result
    running_agent.steer.assert_not_called()
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_steer_with_pending_sentinel_falls_back_to_queue():
    """When the agent hasn't finished booting (sentinel), /steer should
    queue as a turn-boundary follow-up instead of crashing."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

    result = await runner._handle_message(_make_event("/steer wait up"))

    assert result is not None
    assert "queued" in result.lower() or "starting" in result.lower()
    # The fallback put the text into the adapter's pending queue.
    assert sk in adapter._pending_messages
    assert adapter._pending_messages[sk].text == "wait up"


@pytest.mark.asyncio
async def test_steer_agent_without_steer_method_falls_back():
    """If the running agent somehow lacks the steer() method (older build,
    test stub), the handler must not explode — fall back to /queue."""
    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())

    # A bare object that does NOT have steer() — use a spec'd Mock so
    # hasattr(agent, "steer") returns False.
    running_agent = MagicMock(spec=[])
    runner._running_agents[sk] = running_agent

    result = await runner._handle_message(_make_event("/steer fallback"))

    assert result is not None
    # Must mention queueing since steer wasn't available
    assert "queued" in result.lower()
    assert sk in adapter._pending_messages
    assert adapter._pending_messages[sk].text == "fallback"


@pytest.mark.asyncio
async def test_steer_rejected_payload_returns_rejection_message():
    """If agent.steer() returns False (e.g. empty after strip — though
    the gateway already guards this), surface a rejection message."""
    runner, _adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())

    running_agent = MagicMock()
    running_agent.steer.return_value = False
    runner._running_agents[sk] = running_agent

    result = await runner._handle_message(_make_event("/steer hello"))

    assert result is not None
    assert "rejected" in result.lower() or "empty" in result.lower()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
