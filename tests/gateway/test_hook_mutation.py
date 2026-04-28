from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.LOCAL,
        user_id="u1",
        chat_id="local-chat",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.LOCAL: PlatformConfig(enabled=True)}
    )
    runner.adapters = {Platform.LOCAL: MagicMock()}
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._session_db = None
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._clear_session_env = MagicMock()
    runner._set_session_env = MagicMock(return_value={})
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._send_voice_reply = AsyncMock()
    runner._update_runtime_status = MagicMock()

    session_entry = SessionEntry(
        session_key="local:local-chat:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.LOCAL,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.config = MagicMock()
    runner.session_store.config.get_reset_policy.return_value = SimpleNamespace(
        notify=False,
        notify_exclude_platforms=[],
        idle_minutes=0,
        at_hour=0,
    )
    return runner


@pytest.mark.asyncio
async def test_agent_hooks_can_mutate_context_prompt_and_response(monkeypatch):
    runner = _make_runner()
    event = _make_event("hello")
    captured_contexts: list[str] = []

    async def _emit(event_type: str, context: dict) -> None:
        if event_type == "agent:start":
            context["context_prompt"] = "hooked context"
        if event_type == "agent:end":
            context["response"] = "hooked response"

    runner.hooks = SimpleNamespace(emit=AsyncMock(side_effect=_emit), loaded_hooks=True)
    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")

    async def _run_agent(**kwargs):
        captured_contexts.append(kwargs["context_prompt"])
        return {
            "final_response": "original response",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "original response"},
            ],
            "history_offset": 0,
            "api_calls": 1,
            "last_prompt_tokens": 12,
        }

    runner._run_agent = AsyncMock(side_effect=_run_agent)

    monkeypatch.setattr("gateway.run.build_session_context", lambda *_a, **_kw: {})
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *_a, **_kw: "base context")
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.4")

    response = await runner._handle_message_with_agent(event, event.source, "quick-key")

    assert captured_contexts == ["hooked context"]
    assert response == "hooked response"
    assert any(
        call.args[0] == "sess-1"
        and call.args[1].get("role") == "assistant"
        and call.args[1].get("content") == "hooked response"
        for call in runner.session_store.append_to_transcript.call_args_list
    )
