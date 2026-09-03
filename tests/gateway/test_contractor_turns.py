"""Behavior contract for registry-authorized stateless contractor turns.

These tests deliberately exercise the gateway seam rather than the Bloodbank
plugin.  The plugin owns wire validation and routing; Hermes owns the guarantee
that a validated contractor context cannot resume, persist, or learn from a
conversation.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.platforms.base as platform_base
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _context(tmp_path: Path, **overrides):
    context_type = getattr(platform_base, "ContractorTurnContext", None)
    assert context_type is not None, "Hermes must expose typed contractor context"
    values = {
        "contractor_id": "board-cranker",
        "contractor_version": 1,
        "memory_policy": "none",
        "continuity": False,
        "required_skills": ("alpha", "beta"),
        "profile_name": "operations",
        "project_root": str(tmp_path.resolve()),
    }
    values.update(overrides)
    return context_type(**values)


def _source() -> SessionSource:
    return SessionSource(
        # Bloodbank itself is a runtime plugin. WEBHOOK exercises the same
        # platform-neutral gateway seam without mutating the global plugin
        # registry during collection.
        platform=Platform.WEBHOOK,
        chat_id="contractor-thread",
        chat_type="dm",
        user_id="board-cranker-pm",
        user_name="board-cranker-pm",
        profile="operations",
        message_id="command-1",
    )


def _event(tmp_path: Path, **context_overrides):
    return platform_base.MessageEvent(
        text="Implement the story.",
        source=_source(),
        message_id="command-1",
        internal=True,
        contractor_context=_context(tmp_path, **context_overrides),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"contractor_id": ""}, "contractor_id"),
        ({"contractor_version": True}, "contractor_version"),
        ({"contractor_version": 0}, "contractor_version"),
        ({"memory_policy": "profile"}, "memory_policy"),
        ({"continuity": True}, "continuity"),
        ({"required_skills": ()}, "required_skills"),
        ({"required_skills": ("alpha", "")}, "required_skills"),
        ({"required_skills": ("alpha", "alpha")}, "required_skills"),
        ({"profile_name": ""}, "profile_name"),
        ({"project_root": "relative/project"}, "project_root"),
    ],
)
def test_typed_context_rejects_invalid_or_widening_values(tmp_path, overrides, message):
    context_type = getattr(platform_base, "ContractorTurnContext", None)
    assert context_type is not None, "Hermes must expose typed contractor context"
    values = {
        "contractor_id": "board-cranker",
        "contractor_version": 1,
        "memory_policy": "none",
        "continuity": False,
        "required_skills": ("alpha", "beta"),
        "profile_name": "operations",
        "project_root": str(tmp_path.resolve()),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        context_type(**values)


@pytest.mark.asyncio
async def test_contractor_event_bypasses_session_resume_and_transcript_loading(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._handle_contractor_turn = AsyncMock(return_value="done")
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            side_effect=AssertionError("contractor turn must not create or resume a session")
        )
    )

    event = _event(tmp_path)
    response = await runner._handle_message_with_agent(
        event, event.source, "contractor:command-1", 1
    )

    assert response == "done"
    runner._handle_contractor_turn.assert_awaited_once_with(
        event, event.source, event.contractor_context
    )
    runner._async_session_store.get_or_create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_contractor_event_bypasses_all_ordinary_session_bookkeeping(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._handle_contractor_turn = AsyncMock(return_value="done")
    runner._claim_active_session_slot = Mock(
        side_effect=AssertionError(
            "contractor turn must not claim an ordinary session slot"
        )
    )
    event = _event(tmp_path)

    response = await runner._handle_message(event)

    assert response == "done"
    runner._handle_contractor_turn.assert_awaited_once_with(
        event, event.source, event.contractor_context
    )
    runner._claim_active_session_slot.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_event_stays_on_existing_session_path(tmp_path):
    class OrdinaryPathReached(RuntimeError):
        pass

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._handle_contractor_turn = AsyncMock(
        side_effect=AssertionError("ordinary turns must not use contractor execution")
    )
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(side_effect=OrdinaryPathReached)
    )
    source = SessionSource(
        platform=Platform.API_SERVER,
        chat_id="ordinary",
        chat_type="dm",
        user_id="user",
    )
    event = platform_base.MessageEvent(text="hello", source=source)

    with pytest.raises(OrdinaryPathReached):
        await runner._handle_message_with_agent(event, source, "ordinary-key", 1)

    runner._handle_contractor_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_required_skill_fails_before_agent_or_tool_execution(
    tmp_path, monkeypatch
):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=AssertionError("agent construction must not start")
    )
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", lambda *_a, **_k: None)

    event = _event(tmp_path)
    with pytest.raises(rejected_type, match="alpha"):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )

    runner._run_in_executor_with_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_required_skills_resolve_before_any_skill_preprocessing(
    tmp_path, monkeypatch
):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=AssertionError("agent construction must not start")
    )
    loaded = []

    def load_skill(name, task_id=None):
        loaded.append((name, task_id))
        if name == "beta":
            return None
        return ({"content": "alpha"}, tmp_path / name, name)

    build_skill = Mock(
        side_effect=AssertionError(
            "skill preprocessing must wait until all requirements resolve"
        )
    )
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load_skill)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", build_skill)

    event = _event(tmp_path)
    with pytest.raises(rejected_type, match="beta"):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )

    assert [name for name, _task_id in loaded] == ["alpha", "beta"]
    build_skill.assert_not_called()
    runner._run_in_executor_with_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_skills_keep_order_and_registry_cwd_is_task_scoped(
    tmp_path, monkeypatch
):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path / "profile"
    (tmp_path / "profile").mkdir()

    loaded = []

    def load_skill(name, task_id=None):
        loaded.append((name, task_id))
        return ({"content": f"instructions:{name}"}, tmp_path / name, name)

    def build_skill(payload, _skill_dir, note, **_kwargs):
        return f"{note}\n{payload['content']}"

    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load_skill)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", build_skill)

    captured = {}

    def execute(prompt, source, context, session_id, _cancel_event):
        from agent.runtime_cwd import resolve_context_cwd
        from gateway.session_context import get_session_env

        captured.update(
            prompt=prompt,
            source=source,
            context=context,
            session_id=session_id,
            cwd=resolve_context_cwd(),
            profile=get_session_env("HERMES_SESSION_PROFILE"),
        )
        return {"final_response": "done", "completed": True, "failed": False}

    async def run_inline(func, *args):
        return func(*args)

    runner._run_contractor_agent_sync = execute
    runner._run_in_executor_with_context = run_inline

    event = _event(tmp_path)
    response = await runner._handle_contractor_turn(
        event, event.source, event.contractor_context
    )

    assert response == "done"
    assert [name for name, _task_id in loaded] == ["alpha", "beta"]
    assert all(task_id == "command-1" for _name, task_id in loaded)
    assert captured["prompt"].index("instructions:alpha") < captured["prompt"].index(
        "instructions:beta"
    ) < captured["prompt"].index("Implement the story.")
    assert captured["cwd"] == tmp_path.resolve()
    assert captured["profile"] == "operations"
    assert captured["session_id"] == "contractor:board-cranker:command-1"


@pytest.mark.asyncio
async def test_cancellation_signals_worker_before_requesting_hard_interrupt(
    tmp_path, monkeypatch
):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    runner._interrupt_contractor_agent = Mock()

    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, **_kwargs: ({"content": name}, tmp_path / name, name),
    )
    monkeypatch.setattr(
        "agent.skill_commands._build_skill_message",
        lambda payload, *_args, **_kwargs: payload["content"],
    )
    captured = {}

    async def cancel_worker(_func, *args):
        captured["cancel_event"] = args[-1]
        raise asyncio.CancelledError

    runner._run_in_executor_with_context = cancel_worker
    event = _event(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )

    assert captured["cancel_event"].is_set()
    runner._interrupt_contractor_agent.assert_called_once_with(
        "contractor:board-cranker:command-1"
    )


def test_cancelled_worker_stops_before_agent_construction(tmp_path):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(rejected_type, match="cancelled"):
        runner._run_contractor_agent_sync(
            "prompt",
            _source(),
            _context(tmp_path),
            "contractor:board-cranker:command-1",
            cancelled,
        )


def test_agent_is_fresh_memoryless_unpersisted_and_uses_configured_runtime(
    tmp_path, monkeypatch
):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._provider_routing = {
        "only": ["configured-provider"],
        "ignore": None,
        "order": ["configured-provider"],
        "sort": None,
        "require_parameters": True,
        "data_collection": "deny",
    }
    runner._resolve_enabled_toolsets_for_source = lambda *_args: ["web", "memory"]
    runner._resolve_session_agent_runtime = lambda **_kwargs: (
        "configured-model",
        {
            "provider": "configured-provider",
            "api_key": "test-key",
            "base_url": "https://provider.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    runner._resolve_session_reasoning_config = lambda **_kwargs: {"effort": "high"}
    runner._resolve_session_service_tier = lambda **_kwargs: "priority"
    runner._resolve_turn_agent_config = lambda _message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {"service_tier": "priority"},
    }
    runner._refresh_fallback_model = lambda: {"model": "fallback"}
    runner._cleanup_agent_resources = lambda agent: setattr(agent, "cleaned", True)

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"agent": {"disabled_toolsets": ["browser"]}},
    )
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 17)

    instances = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tools = [
                {"type": "function", "function": {"name": "web_search"}}
            ]
            self.model = kwargs["model"]
            self.memory_notifications = "on"
            self.cleaned = False
            instances.append(self)

        def run_conversation(self, message, **kwargs):
            assert self._persist_disabled is True
            self.message = message
            self.run_kwargs = kwargs
            return {
                "final_response": "finished",
                "completed": True,
                "failed": False,
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "finished"},
                ],
            }

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    context = _context(tmp_path)

    first = runner._run_contractor_agent_sync(
        "skill payload\n\nImplement the story.",
        _source(),
        context,
        "contractor:board-cranker:command-1",
    )
    second = runner._run_contractor_agent_sync(
        "skill payload\n\nImplement the story.",
        _source(),
        context,
        "contractor:board-cranker:command-2",
    )

    assert first["final_response"] == second["final_response"] == "finished"
    assert len(instances) == 2 and instances[0] is not instances[1]
    for agent in instances:
        kwargs = agent.kwargs
        assert kwargs["model"] == "configured-model"
        assert kwargs["provider"] == "configured-provider"
        assert kwargs["providers_allowed"] == ["configured-provider"]
        assert kwargs["providers_order"] == ["configured-provider"]
        assert kwargs["session_db"] is None
        assert kwargs["prefill_messages"] is None
        assert kwargs["skip_context_files"] is False
        assert kwargs["skip_memory"] is True
        assert kwargs["skip_background_review"] is True
        assert kwargs["disabled_toolsets"] == ["browser", "memory", "session_search"]
        assert agent.memory_notifications == "off"
        assert agent.run_kwargs["conversation_history"] == []
        assert agent.run_kwargs["task_id"] == kwargs["session_id"]
        assert agent.cleaned is True


def test_memory_or_history_tool_leak_fails_before_model_execution(tmp_path, monkeypatch):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._provider_routing = {}
    runner._resolve_enabled_toolsets_for_source = lambda *_args: ["web"]
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("model", {})
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._resolve_session_service_tier = lambda **_kwargs: None
    runner._resolve_turn_agent_config = lambda _message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {},
    }
    runner._refresh_fallback_model = lambda: None
    runner._cleanup_agent_resources = lambda _agent: None
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda _cfg: {})

    calls = []

    class LeakyAgent:
        def __init__(self, **_kwargs):
            self.tools = [
                {"type": "function", "function": {"name": "session_search"}}
            ]

        def run_conversation(self, *_args, **_kwargs):
            calls.append("model")
            return {"final_response": "unsafe"}

    monkeypatch.setattr("run_agent.AIAgent", LeakyAgent)

    with pytest.raises(rejected_type, match="session_search"):
        runner._run_contractor_agent_sync(
            "prompt",
            _source(),
            _context(tmp_path),
            "contractor:board-cranker:command-1",
        )
    assert calls == []
