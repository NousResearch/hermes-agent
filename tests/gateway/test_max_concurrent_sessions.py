"""Tests for the gateway max_concurrent_sessions active-session cap."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, build_session_key


@pytest.fixture(autouse=True)
def _isolated_active_session_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}

    async def send(self, chat_id, text, **kwargs):
        return None

    async def interrupt_session_activity(self, session_key, chat_id, metadata=None):
        event = self._active_sessions.get(session_key)
        if event is not None:
            event.set()


def _make_source(chat_id: str = "chat-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=f"user-{chat_id}",
    )


def _make_event(text: str = "hello", chat_id: str = "chat-1") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(chat_id),
    )


def _make_runner(max_concurrent_sessions: int | None = None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        max_concurrent_sessions=max_concurrent_sessions,
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._active_session_leases = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._queued_events = {}
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()
    return runner


def _occupy_session(runner: GatewayRunner, chat_id: str = "busy"):
    source = _make_source(chat_id)
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()
    runner._running_agents_ts[session_key] = time.time()
    return session_key


def _silence_global_gateway_hooks(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])
    monkeypatch.setattr("tools.slash_confirm.get_pending", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.slash_confirm.clear_if_stale", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda *args, **kwargs: False)


def test_new_session_gets_clean_error_at_active_session_limit(monkeypatch):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner(max_concurrent_sessions=1)
    _occupy_session(runner, "busy")
    event = _make_event(chat_id="new")
    new_key = build_session_key(event.source)

    async def fail_if_agent_runs(self_inner, ev, src, qk, generation):
        raise AssertionError("_handle_message_with_agent should not run at capacity")

    with patch.object(GatewayRunner, "_handle_message_with_agent", fail_if_agent_runs):
        result = asyncio.run(runner._handle_message(event))

    assert result == (
        "Hermes is at the active session limit (1/1). "
        "Try again when another session finishes."
    )
    assert new_key not in runner._running_agents
    runner.session_store.get_or_create_session.assert_not_called()


def test_existing_active_session_uses_busy_handling_at_limit(monkeypatch):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner(max_concurrent_sessions=1)
    runner._busy_input_mode = "queue"
    event = _make_event(chat_id="busy")
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()
    runner._running_agents_ts[session_key] = 0

    async def fail_if_agent_runs(self_inner, ev, src, qk, generation):
        raise AssertionError("_handle_message_with_agent should not run for busy follow-up")

    with patch.object(GatewayRunner, "_handle_message_with_agent", fail_if_agent_runs):
        result = asyncio.run(runner._handle_message(event))

    assert result is None
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[session_key] is event


def test_new_session_can_start_after_active_session_released(monkeypatch):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner(max_concurrent_sessions=1)
    busy_key = _occupy_session(runner, "busy")
    runner._release_running_agent_state(busy_key)
    event = _make_event(chat_id="new")

    sentinel_seen = False

    async def mock_agent_run(self_inner, ev, src, qk, generation):
        nonlocal sentinel_seen
        sentinel_seen = runner._running_agents.get(qk) is _AGENT_PENDING_SENTINEL
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", mock_agent_run):
        result = asyncio.run(runner._handle_message(event))

    assert result == "ok"
    assert sentinel_seen is True


def test_status_command_bypasses_active_session_limit(monkeypatch):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner(max_concurrent_sessions=1)
    _occupy_session(runner, "busy")
    runner._handle_status_command = AsyncMock(return_value="status ok")

    result = asyncio.run(runner._handle_message(_make_event("/status", chat_id="new")))

    assert result == "status ok"
    runner._handle_status_command.assert_awaited_once()


def test_skill_command_that_would_start_agent_is_blocked_at_limit(monkeypatch):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner(max_concurrent_sessions=1)
    _occupy_session(runner, "busy")

    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"demo": {"name": "demo-skill"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda command: "demo" if command == "demo" else None,
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda *args, **kwargs: "invoke demo skill",
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names",
        lambda *args, **kwargs: [],
    )

    async def fail_if_agent_runs(self_inner, ev, src, qk, generation):
        raise AssertionError("_handle_message_with_agent should not run at capacity")

    with patch.object(GatewayRunner, "_handle_message_with_agent", fail_if_agent_runs):
        result = asyncio.run(
            runner._handle_message(_make_event("/demo please", chat_id="new"))
        )

    assert result == (
        "Hermes is at the active session limit (1/1). "
        "Try again when another session finishes."
    )
