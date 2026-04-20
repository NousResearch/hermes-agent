from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        thread_id="thread-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m-1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: AsyncMock()}
    runner.adapters[Platform.TELEGRAM].send = AsyncMock()
    runner.adapters[Platform.TELEGRAM].extract_media = MagicMock(return_value=([], "ok"))
    runner.adapters[Platform.TELEGRAM].extract_images = MagicMock(return_value=([], "ok"))
    runner._background_tasks = set()
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._session_db = None
    runner._service_tier = None
    runner._resolve_session_agent_runtime = MagicMock(
        return_value=("test-model", {"api_key": "key", "provider": "openai"})
    )
    runner._resolve_turn_agent_config = MagicMock(
        return_value={"model": "test-model", "runtime": {"api_key": "key", "provider": "openai"}}
    )
    runner._load_reasoning_config = MagicMock(return_value=None)
    runner._load_service_tier = MagicMock(return_value=None)
    runner._run_in_executor_with_context = AsyncMock(side_effect=lambda fn: fn())
    runner._cleanup_agent_resources = MagicMock()
    runner._session_key_for_source = MagicMock(return_value="session-key")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="sess-1")
    runner.session_store.load_transcript.return_value = []
    return runner


def test_resolve_orchestration_backend_defaults_to_legacy():
    from gateway.run import _resolve_orchestration_backend

    assert _resolve_orchestration_backend({}) == "legacy"
    assert _resolve_orchestration_backend({"agent": {}}) == "legacy"


def test_resolve_orchestration_backend_accepts_langgraph():
    from gateway.run import _resolve_orchestration_backend

    assert _resolve_orchestration_backend({"agent": {"orchestration_backend": "langgraph"}}) == "langgraph"


@pytest.mark.asyncio
async def test_run_conversation_via_orchestration_attaches_metadata():
    runner = _make_runner()

    with patch("gateway.run._load_gateway_config", return_value={"agent": {"orchestration_backend": "legacy"}}):
        result = await runner._run_conversation_via_orchestration(
            route_name="gateway.message",
            session_id="sess-1",
            task_id="task-1",
            thread_id="thread-1",
            model="test-model",
            tool_names=["web_search"],
            runner=lambda: {"final_response": "hello", "messages": []},
        )

    assert result["final_response"] == "hello"
    assert result["orchestration"]["backend"] == "legacy"
    assert result["orchestration"]["status"] == "completed"
    assert result["orchestration"]["completed_steps"] == ["gateway.message"]


@pytest.mark.asyncio
async def test_background_route_uses_orchestration_wrapper():
    runner = _make_runner()
    event_source = _make_source()

    with patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent") as MockAgent:
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "background ok", "messages": []}
        MockAgent.return_value = agent
        runner._run_conversation_via_orchestration = AsyncMock(
            return_value={
                "final_response": "background ok",
                "messages": [],
                "orchestration": {"backend": "legacy", "status": "completed"},
            }
        )

        await runner._run_background_task("do thing", event_source, "bg-task")

    runner._run_conversation_via_orchestration.assert_awaited_once()
    assert runner._run_conversation_via_orchestration.await_args.kwargs["route_name"] == "gateway.background"


@pytest.mark.asyncio
async def test_btw_route_uses_orchestration_wrapper():
    runner = _make_runner()
    event_source = _make_source()

    with patch("gateway.run._load_gateway_config", return_value={}), \
         patch("run_agent.AIAgent") as MockAgent:
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "btw ok", "messages": []}
        MockAgent.return_value = agent
        runner._run_conversation_via_orchestration = AsyncMock(
            return_value={
                "final_response": "btw ok",
                "messages": [],
                "orchestration": {"backend": "legacy", "status": "completed"},
            }
        )

        await runner._run_btw_task("quick question", event_source, "session-key", "btw-task")

    runner._run_conversation_via_orchestration.assert_awaited_once()
    assert runner._run_conversation_via_orchestration.await_args.kwargs["route_name"] == "gateway.btw"
