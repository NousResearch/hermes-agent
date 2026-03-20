from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp_adapter.session import SessionManager
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(**overrides) -> SessionSource:
    data = {
        "platform": Platform.DISCORD,
        "user_id": "u1",
        "chat_id": "c1",
        "user_name": "alan",
        "chat_name": "Hermes / #general",
        "chat_type": "group",
    }
    data.update(overrides)
    return SessionSource(**data)


def _make_event(text: str, *, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source or _make_source(),
        message_id="m1",
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(send=AsyncMock())
    }
    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime(2026, 3, 19, 12, 0),
        updated_at=datetime(2026, 3, 19, 12, 30),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key = lambda session_source: build_session_key(session_source)
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
    runner._effective_model = None
    runner._effective_provider = None
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._runtime_debug_overrides = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._session_db = None
    runner._subagent_runtime = {}
    runner._acp_session_bindings = {}
    runner._acp_session_meta = {}
    runner._acp_running_tasks = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_running_session_allows_subagent_command_without_interrupt():
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    child = MagicMock()
    child.interrupt = MagicMock()

    runner._register_delegate_child(
        session_key=session_key,
        child=child,
        task_index=0,
        goal="inspect logs",
        context="",
        toolsets=["terminal"],
        model="anthropic/claude-opus-4.6",
    )
    runner._start_delegate_child(session_key=session_key, child=child, goal="inspect logs")

    active_agent = MagicMock()
    runner._running_agents[session_key] = active_agent

    result = await runner._handle_message(_make_event("/subagents list", source=source))

    assert "Subagents" in result
    assert "sa-1" in result
    active_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_kill_and_steer_commands_control_live_subagent():
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    child = MagicMock()
    child.interrupt = MagicMock()

    runner._register_delegate_child(
        session_key=session_key,
        child=child,
        task_index=0,
        goal="inspect logs",
        context="",
        toolsets=["terminal"],
        model="anthropic/claude-opus-4.6",
    )
    runner._start_delegate_child(session_key=session_key, child=child, goal="inspect logs")

    steer_result = await runner._handle_steer_command(_make_event("/steer #1 focus on tests", source=source))
    kill_result = await runner._handle_kill_command(_make_event("/kill #1", source=source))

    entry = runner._get_subagent_session_state(session_key)["entries"]["sa-1"]
    assert "Steering subagent" in steer_result
    assert entry["pending_steer"] is None
    assert "Requested stop" in kill_result
    assert child.interrupt.call_count == 2


@pytest.mark.asyncio
async def test_acp_spawn_status_option_commands_and_close():
    runner = _make_runner()

    class FakeACPAgent:
        def __init__(self):
            self.model = "anthropic/claude-opus-4.6"
            self.interrupt = MagicMock()

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            return {
                "final_response": f"ACP:{user_message}",
                "messages": list(conversation_history or []),
                "completed": True,
            }

    runner._acp_session_manager = SessionManager(agent_factory=FakeACPAgent)
    source = _make_source()
    session_key = build_session_key(source)

    spawn_result = await runner._handle_acp_command(_make_event("/acp spawn /tmp/work", source=source))
    session_id = runner._get_acp_bindings()[session_key]

    sessions_result = await runner._handle_acp_command(_make_event("/acp sessions", source=source))
    status_result = await runner._handle_acp_command(_make_event("/acp status", source=source))
    model_result = await runner._handle_acp_command(_make_event("/acp model openai/gpt-5.2", source=source))
    timeout_result = await runner._handle_acp_command(_make_event("/acp timeout 45", source=source))
    set_result = await runner._handle_acp_command(_make_event("/acp set profile fast", source=source))
    reset_result = await runner._handle_acp_command(_make_event("/acp reset-options", source=source))
    close_result = await runner._handle_acp_command(_make_event("/acp close", source=source))

    assert "Created ACP session" in spawn_result
    assert session_id in sessions_result
    assert session_id in status_result
    assert "model set to `openai/gpt-5.2`" in model_result
    assert "timeout set to `45`s" in timeout_result
    assert "option `profile` set to `fast`" in set_result
    assert "reset `2` option" in reset_result
    assert f"Closed ACP session `{session_id}`" in close_result
