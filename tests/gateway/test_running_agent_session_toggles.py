"""Regression tests for session commands dispatched during an agent run.

When an agent is running, the gateway's running-agent guard rejects most
slash commands with "⏳ Agent is running — /{cmd} can't run mid-turn"
(PR #12334). A small allowlist bypasses that and actually dispatches:

  * /yolo — toggles the session yolo flag; useful to pre-approve a
    pending approval prompt without waiting for the agent to finish.
  * /verbose — cycles the per-platform tool-progress display mode;
    affects the ongoing stream.

Config-only commands still reject changes during the active turn:

  * /fast — uses the generic busy response
  * /reasoning — uses a busy response that reports the active level

These tests lock in the dispatch and rejection behaviors.
"""

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
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """Minimal GatewayRunner with an active running agent for this session."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._service_tier = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()

    # Simulate agent actively running for this session so the guard fires.
    # Note: the stale-eviction branch calls agent.get_activity_summary() and
    # compares seconds_since_activity against HERMES_AGENT_TIMEOUT. Return a
    # dict with recent activity so the eviction path doesn't clear our
    # fake running agent before the toggle guard runs.
    import time
    sk = build_session_key(_make_source())
    agent_mock = MagicMock()
    agent_mock.get_activity_summary.return_value = {
        "seconds_since_activity": 0.0,
        "last_activity_desc": "api_call",
        "api_call_count": 1,
        "max_iterations": 60,
    }
    agent_mock.reasoning_config = {"enabled": True, "effort": "high"}
    runner._running_agents[sk] = agent_mock
    runner._running_agents_ts[sk] = time.time()
    return runner


@pytest.mark.asyncio
async def test_yolo_dispatches_mid_run(monkeypatch):
    """/yolo mid-run must dispatch to its handler, not hit the catch-all."""
    runner = _make_runner()
    runner._handle_yolo_command = AsyncMock(return_value="⚡ YOLO mode **ON** for this session")

    result = await runner._handle_message(_make_event("/yolo"))

    runner._handle_yolo_command.assert_awaited_once()
    assert result == "⚡ YOLO mode **ON** for this session"
    assert "can't run mid-turn" not in (result or "")


@pytest.mark.asyncio
async def test_verbose_dispatches_mid_run(monkeypatch):
    """/verbose mid-run must dispatch to its handler, not hit the catch-all."""
    runner = _make_runner()
    runner._handle_verbose_command = AsyncMock(return_value="tool progress: new")

    result = await runner._handle_message(_make_event("/verbose"))

    runner._handle_verbose_command.assert_awaited_once()
    assert result == "tool progress: new"
    assert "can't run mid-turn" not in (result or "")




@pytest.mark.asyncio
async def test_fast_rejected_mid_run():
    """/fast remains config-only and rejected while a turn is active."""
    runner = _make_runner()
    runner._handle_fast_command = AsyncMock(
        side_effect=AssertionError("/fast should not dispatch mid-run")
    )

    result = await runner._handle_message(_make_event("/fast"))

    runner._handle_fast_command.assert_not_awaited()
    assert result is not None
    assert "can't run mid-turn" in result
    assert "/fast" in result


@pytest.mark.asyncio
async def test_reasoning_effort_change_reports_active_level_mid_run():
    """/reasoning effort changes report the active level without applying."""
    runner = _make_runner()
    runner._apply_reasoning_selection = MagicMock(
        side_effect=AssertionError("effort must not change mid-turn")
    )

    result = await runner._handle_message(_make_event("/reasoning ultra"))

    runner._apply_reasoning_selection.assert_not_called()
    assert result is not None
    assert "running on **high** reasoning" in result
    assert "can't change effort mid-turn" in result


@pytest.mark.asyncio
async def test_reasoning_pending_agent_uses_session_resolution():
    """A pending turn resolves reasoning from that session's model state."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    runner._session_model_overrides = {session_key: {"model": "model-a"}}
    runner._resolve_session_reasoning_config = MagicMock(
        return_value={"enabled": False}
    )

    result = await runner._handle_message(_make_event("/reasoning"))

    assert result is not None
    assert "running on **none (disabled)** reasoning" in result
    runner._resolve_session_reasoning_config.assert_called_once_with(
        source=_make_source(),
        session_key=session_key,
        model="model-a",
    )


@pytest.mark.asyncio
async def test_reasoning_default_does_not_leak_from_another_running_session():
    """An explicit agent default must not fall back to runner-global state."""
    runner = _make_runner()
    active_key = build_session_key(_make_source())
    runner._running_agents[active_key].reasoning_config = None

    other_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u2",
        chat_id="c2",
        user_name="other",
        chat_type="dm",
    )
    other_agent = MagicMock()
    other_agent.reasoning_config = {"enabled": True, "effort": "ultra"}
    runner._running_agents[build_session_key(other_source)] = other_agent
    runner._reasoning_config = other_agent.reasoning_config

    result = await runner._handle_message(_make_event("/reasoning"))

    assert result is not None
    assert "running on **medium (default)** reasoning" in result
    assert "ultra" not in result


@pytest.mark.asyncio
async def test_reasoning_display_toggle_applies_mid_run():
    """/reasoning hide is display-only and may affect the active response."""
    runner = _make_runner()
    runner._show_reasoning = True
    runner._save_gateway_config_key = MagicMock(return_value=True)

    result = await runner._handle_message(_make_event("/reasoning hide"))

    assert result is not None
    assert "can't change effort mid-turn" not in result
    assert runner._show_reasoning is False
    runner._save_gateway_config_key.assert_called_once_with(
        "display.platforms.telegram.show_reasoning", False
    )
