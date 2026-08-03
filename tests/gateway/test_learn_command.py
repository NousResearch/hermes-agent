"""Handler-level tests for gateway ``/learn`` skill_manage gating."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent.learn_prompt import LEARN_UNAVAILABLE_MESSAGE, build_learn_prompt
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key

_READ_ONLY = {"memory", "skills_list", "skill_view"}
_WRITABLE = _READ_ONLY | {"skill_manage"}


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
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
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
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
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
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._thread_metadata_for_source = MagicMock(return_value={})
    return runner


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_gateway_learn_running_agent_read_only_returns_unavailable():
    import gateway.run as gateway_run

    runner = _make_runner()
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock(valid_tool_names=_READ_ONLY)
    runner._is_session_running = MagicMock(return_value=False)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/learn leaked to agent on read-only toolset")
    )

    with patch.object(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}):
        result = _run(runner._handle_message(_make_event("/learn oauth notes")))

    assert result == LEARN_UNAVAILABLE_MESSAGE
    runner._run_agent.assert_not_called()


def test_gateway_learn_running_agent_writable_rewrites_event():
    import gateway.run as gateway_run

    runner = _make_runner()
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock(valid_tool_names=_WRITABLE)
    runner._is_session_running = MagicMock(return_value=False)
    runner._run_agent = AsyncMock(return_value="ok")

    event = _make_event("/learn oauth notes")

    with patch.object(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}):
        _run(runner._handle_message(event))

    assert event.text == build_learn_prompt("oauth notes")
    runner._run_agent.assert_called()


def test_gateway_learn_fallback_tool_resolution_read_only():
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._is_session_running = MagicMock(return_value=False)
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/learn leaked to agent on read-only fallback")
    )

    read_only_defs = [
        {"function": {"name": "memory"}},
        {"function": {"name": "skills_list"}},
        {"function": {"name": "skill_view"}},
    ]

    with patch.object(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}), \
         patch("gateway.run._load_gateway_config", return_value={"agent": {}}), \
         patch("gateway.run._platform_config_key", return_value="messaging"), \
         patch("hermes_cli.tools_config._get_platform_tools", return_value={"memory", "skills"}), \
         patch("model_tools.get_tool_definitions", return_value=read_only_defs):
        result = _run(runner._handle_message(_make_event("/learn stale paths")))

    assert result == LEARN_UNAVAILABLE_MESSAGE
    runner._run_agent.assert_not_called()
