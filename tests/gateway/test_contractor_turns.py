"""Behavior contract for registry-authorized stateless contractor turns.

These tests deliberately exercise the gateway seam rather than the Bloodbank
plugin.  The plugin owns wire validation and routing; Hermes owns the guarantee
that a validated contractor context cannot resume, persist, or learn from a
conversation.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.platforms.base as platform_base
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _isolate_profile_secret_hydration(monkeypatch):
    """Contractor behavior tests must not resolve workstation vault refs."""
    monkeypatch.setattr(gateway_run, "_load_profile_secret_scope", lambda _home: {})

    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    @asynccontextmanager
    async def runtime_scope(profile_home):
        with gateway_run._profile_runtime_scope(profile_home, {}):
            yield

    monkeypatch.setattr(gateway_run, "_async_profile_runtime_scope", runtime_scope)


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


class _TestPlatformAdapter(platform_base.BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return platform_base.SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


def _test_adapter() -> _TestPlatformAdapter:
    return _TestPlatformAdapter(
        SimpleNamespace(typing_indicator=False, extra={}),
        Platform.WEBHOOK,
    )


def test_base_platform_exposes_explicit_stateless_turn_contract():
    adapter = _test_adapter()
    target = "agent:main:webhook:contractor"
    sibling = "agent:main:webhook:ordinary"
    adapter._active_sessions = {
        target: asyncio.Event(),
        sibling: asyncio.Event(),
    }
    adapter._pending_messages = {
        target: Mock(),
        sibling: Mock(),
    }
    adapter._session_tasks = {
        target: Mock(),
        sibling: Mock(),
    }
    adapter._post_delivery_callbacks = {
        target: Mock(),
        sibling: Mock(),
    }
    adapter._text_debounce = {
        target: SimpleNamespace(task=None),
        sibling: SimpleNamespace(task=None),
    }

    assert platform_base.BasePlatformAdapter.close_after_turn is False
    adapter.clear_session(target)

    for store in (
        adapter._active_sessions,
        adapter._pending_messages,
        adapter._session_tasks,
        adapter._post_delivery_callbacks,
        adapter._text_debounce,
    ):
        assert target not in store
        assert sibling in store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("close_after_turn", "contractor_turn", "expected_clear"),
    [
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
async def test_platform_clears_session_after_each_stateless_turn(
    tmp_path, close_after_turn, contractor_turn, expected_clear
):
    adapter = _test_adapter()
    adapter.close_after_turn = close_after_turn
    adapter._message_handler = AsyncMock(return_value=None)
    session_key = "agent:main:webhook:contractor"
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()
    real_clear_session = adapter.clear_session
    adapter.clear_session = Mock(side_effect=real_clear_session)
    event = (
        _event(tmp_path)
        if contractor_turn
        else platform_base.MessageEvent(text="hello", source=_source())
    )

    await adapter._process_message_background(event, session_key)

    assert adapter.clear_session.called is expected_clear
    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._session_tasks


@pytest.mark.asyncio
async def test_stateless_contractor_unwind_preserves_command_owned_guard(tmp_path):
    adapter = _test_adapter()
    contractor_event = _event(tmp_path)
    command_event = platform_base.MessageEvent(text="/reset", source=_source())
    unwind_started = asyncio.Event()
    resume_unwind = asyncio.Event()
    command_started = asyncio.Event()
    resume_command = asyncio.Event()

    async def handle_event(event):
        if event is command_event:
            command_started.set()
            await resume_command.wait()
        return None

    async def hold_contractor_unwind(event, _outcome):
        if event is contractor_event:
            unwind_started.set()
            await resume_unwind.wait()

    adapter._message_handler = handle_event
    adapter.on_processing_complete = hold_contractor_unwind

    await adapter.handle_message(contractor_event)
    await asyncio.wait_for(unwind_started.wait(), timeout=1)
    assert len(adapter._session_tasks) == 1
    session_key, old_task = next(iter(adapter._session_tasks.items()))
    old_guard = adapter._active_sessions[session_key]

    command_task = asyncio.create_task(adapter.handle_message(command_event))
    try:
        await asyncio.wait_for(command_started.wait(), timeout=1)
        command_guard = adapter._active_sessions[session_key]
        assert command_guard is not old_guard

        resume_unwind.set()
        await asyncio.wait_for(asyncio.shield(old_task), timeout=1)

        assert adapter._active_sessions.get(session_key) is command_guard
        assert adapter._session_tasks.get(session_key) is old_task
    finally:
        resume_unwind.set()
        resume_command.set()
        await asyncio.wait_for(command_task, timeout=1)


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


@pytest.mark.parametrize("invalid_root", ["missing", "file"])
def test_typed_context_rejects_non_directory_project_root(tmp_path, invalid_root):
    root = tmp_path / invalid_root
    if invalid_root == "file":
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="existing directory"):
        _context(tmp_path, project_root=str(root))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("invalid-type", "invalid type"),
        ("external-event", "internal events"),
        ("profile-mismatch", "profile"),
    ],
)
async def test_contractor_handler_rejects_untrusted_context_before_skills(
    tmp_path, monkeypatch, case, message
):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=AssertionError("agent construction must not start")
    )
    load_skill = Mock(
        side_effect=AssertionError("skill loading must not start")
    )
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load_skill)

    event = _event(tmp_path)
    context = event.contractor_context
    if case == "invalid-type":
        context = {"project_root": str(tmp_path)}
    elif case == "external-event":
        event.internal = False
    else:
        context = _context(tmp_path, profile_name="other-profile")

    with pytest.raises(rejected_type, match=message):
        await runner._handle_contractor_turn(event, event.source, context)

    load_skill.assert_not_called()
    runner._run_in_executor_with_context.assert_not_awaited()


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
    profile_home = tmp_path / "profile"
    project_root = profile_home / "project"
    project_root.mkdir(parents=True)
    runner._resolve_profile_home_for_source = lambda _source: profile_home

    loaded = []

    def load_skill(name, task_id=None):
        loaded.append((name, task_id))
        return ({"content": f"instructions:{name}"}, project_root / name, name)

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

    event = _event(tmp_path, project_root=str(project_root))
    response = await runner._handle_contractor_turn(
        event, event.source, event.contractor_context
    )

    assert response == "done"
    assert [name for name, _task_id in loaded] == ["alpha", "beta"]
    assert all(task_id == "command-1" for _name, task_id in loaded)
    assert captured["prompt"].index("instructions:alpha") < captured["prompt"].index(
        "instructions:beta"
    ) < captured["prompt"].index("Implement the story.")
    assert captured["cwd"] == project_root.resolve()
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
    runner._resolve_turn_agent_config = lambda _message, model, runtime, **kwargs: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {"service_tier": "priority"},
    }
    runner._refresh_fallback_model = lambda: {"model": "fallback"}

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
            self.closed = False
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

        def close(self):
            self.closed = True

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
        assert agent.closed is True


def test_memory_or_history_tool_leak_fails_before_model_execution(tmp_path, monkeypatch):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._provider_routing = {}
    runner._resolve_enabled_toolsets_for_source = lambda *_args: ["web"]
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("model", {})
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._resolve_session_service_tier = lambda **_kwargs: None
    runner._resolve_turn_agent_config = lambda _message, model, runtime, **kwargs: {
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


# ---------------------------------------------------------------------------
# Quality-fix tests for 33GOD-51
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contractor_turn_accepts_disjoint_typed_project_root(
    tmp_path, monkeypatch
):
    """Bloodbank authorizes the project; Hermes only validates the typed path."""
    runner = object.__new__(gateway_run.GatewayRunner)
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    project_root = tmp_path / "authorized-project"
    project_root.mkdir()
    runner._resolve_profile_home_for_source = lambda _source: profile_home
    runner._run_contractor_agent_sync = Mock(
        return_value={"final_response": "done", "completed": True, "failed": False}
    )

    async def run_inline(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = run_inline
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, **_kwargs: ({"content": name}, project_root / name, name),
    )
    monkeypatch.setattr(
        "agent.skill_commands._build_skill_message",
        lambda payload, *_args, **_kwargs: payload["content"],
    )

    event = _event(tmp_path, project_root=str(project_root))
    response = await runner._handle_contractor_turn(
        event, event.source, event.contractor_context
    )

    assert response == "done"
    worker_args = runner._run_contractor_agent_sync.call_args.args
    assert worker_args[2] is event.contractor_context
    assert worker_args[2].project_root == str(project_root.resolve())


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["missing", "file"])
async def test_contractor_turn_revalidates_project_directory_before_skills(
    tmp_path, monkeypatch, replacement
):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    profile_home = tmp_path / "profile"
    project_root = profile_home / "project"
    project_root.mkdir(parents=True)
    runner._resolve_profile_home_for_source = lambda _source: profile_home
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=AssertionError("agent construction must not start")
    )
    load_skill = Mock(
        side_effect=AssertionError("skill loading must not start")
    )
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load_skill)

    event = _event(tmp_path, project_root=str(project_root))
    project_root.rmdir()
    if replacement == "file":
        project_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        rejected_type, match="contractor project root|existing directory"
    ):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )

    load_skill.assert_not_called()
    runner._run_in_executor_with_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_contractor_turn_accepts_project_root_under_profile_home(
    tmp_path, monkeypatch
):
    rejected_type = getattr(gateway_run, "ContractorTurnRejected", RuntimeError)
    runner = object.__new__(gateway_run.GatewayRunner)
    profile_home = tmp_path / "profile"
    project_root = profile_home / "project"
    project_root.mkdir(parents=True)
    runner._resolve_profile_home_for_source = lambda _source: profile_home
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=AssertionError("agent construction must not start")
    )
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload", lambda *_a, **_k: None
    )

    event = _event(tmp_path, project_root=str(project_root))
    with pytest.raises(rejected_type, match="alpha"):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )

    runner._run_in_executor_with_context.assert_not_awaited()


def _minimal_contractor_runner(tmp_path, monkeypatch, **overrides):
    """Build a GatewayRunner stub wired for direct _run_contractor_agent_sync tests."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    runner._provider_routing = {}
    runner._resolve_enabled_toolsets_for_source = lambda *_args: ["web"]
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("model", {})
    runner._resolve_session_reasoning_config = lambda **_kwargs: {"effort": "high"}
    runner._resolve_session_service_tier = lambda **_kwargs: "priority"
    runner._resolve_turn_agent_config = lambda _message, model, runtime, **kwargs: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {},
    }
    runner._refresh_fallback_model = lambda: None
    for name, value in overrides.items():
        setattr(runner, name, value)

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 10)
    return runner


def test_null_contractor_service_tier_is_distinct_from_ordinary_omission(
    monkeypatch,
):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = "priority"
    runtime = {
        "provider": "test-provider",
        "request_overrides": {"extra_body": {"existing": True}},
    }
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda *_args, **_kwargs: {"service_tier": "priority"},
    )

    contractor_route = runner._resolve_turn_agent_config(
        "prompt", "test-model", runtime, service_tier=None
    )
    ordinary_route = runner._resolve_turn_agent_config(
        "prompt", "test-model", runtime
    )

    assert contractor_route["request_overrides"] == {
        "extra_body": {"existing": True}
    }
    assert ordinary_route["request_overrides"] == {
        "extra_body": {"existing": True},
        "service_tier": "priority",
    }


class _CleanFakeAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, *_args, **_kwargs):
        return {"final_response": "ok", "completed": True, "failed": False}

    def close(self):
        pass


def test_contractor_turn_does_not_mutate_shared_runner_state(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(
        tmp_path,
        monkeypatch,
        _reasoning_config="initial-reasoning",
        _service_tier="initial-tier",
    )
    monkeypatch.setattr("run_agent.AIAgent", _CleanFakeAgent)

    runner._run_contractor_agent_sync(
        "prompt",
        _source(),
        _context(tmp_path),
        "contractor:board-cranker:command-1",
    )

    assert runner._reasoning_config == "initial-reasoning"
    assert runner._service_tier == "initial-tier"


def test_concurrent_contractor_turns_leave_runner_config_unchanged(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(
        tmp_path,
        monkeypatch,
        _reasoning_config="initial-reasoning",
        _service_tier="initial-tier",
    )
    monkeypatch.setattr("run_agent.AIAgent", _CleanFakeAgent)

    import concurrent.futures

    def run_turn(i):
        return runner._run_contractor_agent_sync(
            "prompt",
            _source(),
            _context(tmp_path),
            f"contractor:board-cranker:command-{i}",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_turn, i) for i in range(2)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assert runner._reasoning_config == "initial-reasoning"
    assert runner._service_tier == "initial-tier"


class _MessySuccessAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, *_args, **_kwargs):
        return {"final_response": "ok", "completed": True, "failed": False}

    def close(self):
        raise RuntimeError("cleanup failed")


def test_contractor_cleanup_failure_is_observed_on_successful_turn(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _MessySuccessAgent)

    result = runner._run_contractor_agent_sync(
        "prompt",
        _source(),
        _context(tmp_path),
        "contractor:board-cranker:command-1",
    )

    assert result["final_response"] == "ok"
    assert result["completed"] is True
    assert "cleanup_error" in result
    assert "cleanup failed" in result["cleanup_error"]


@pytest.mark.asyncio
async def test_cleanup_failure_raises_from_public_contractor_handler(
    tmp_path, monkeypatch
):
    cleanup_failure_type = getattr(
        gateway_run,
        "ContractorTurnCleanupFailed",
        gateway_run.ContractorTurnRejected,
    )
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _MessySuccessAgent)
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, **_kwargs: ({"content": name}, project_root / name, name),
    )
    monkeypatch.setattr(
        "agent.skill_commands._build_skill_message",
        lambda payload, *_args, **_kwargs: payload["content"],
    )

    async def run_inline(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = run_inline
    event = _event(tmp_path, project_root=str(project_root))

    with pytest.raises(cleanup_failure_type, match="cleanup failed"):
        await runner._handle_contractor_turn(
            event, event.source, event.contractor_context
        )


@pytest.mark.asyncio
async def test_cleanup_failure_marks_platform_processing_failed(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _MessySuccessAgent)
    monkeypatch.setattr(
        "agent.skill_commands._load_skill_payload",
        lambda name, **_kwargs: ({"content": name}, project_root / name, name),
    )
    monkeypatch.setattr(
        "agent.skill_commands._build_skill_message",
        lambda payload, *_args, **_kwargs: payload["content"],
    )

    async def run_inline(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = run_inline
    event = _event(tmp_path, project_root=str(project_root))
    adapter = _test_adapter()
    adapter._message_handler = runner._handle_message
    adapter.on_processing_complete = AsyncMock()
    adapter.send = AsyncMock(
        return_value=platform_base.SendResult(success=True, message_id="error-1")
    )
    session_key = "agent:operations:webhook:contractor"
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._session_tasks[session_key] = asyncio.current_task()

    await adapter._process_message_background(event, session_key)

    adapter.on_processing_complete.assert_awaited_once_with(
        event, platform_base.ProcessingOutcome.FAILURE
    )


class _MessyFailureAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, *_args, **_kwargs):
        raise RuntimeError("model call failed")

    def close(self):
        raise RuntimeError("cleanup failed")


def test_contractor_cleanup_failure_is_observed_on_failed_turn(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _MessyFailureAgent)

    with pytest.raises(RuntimeError, match="model call failed") as exc_info:
        runner._run_contractor_agent_sync(
            "prompt",
            _source(),
            _context(tmp_path),
            "contractor:board-cranker:command-1",
        )

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("cleanup failed" in note for note in notes)


def test_contractor_agent_registry_is_evicted_after_successful_turn(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _CleanFakeAgent)

    session_id = "contractor:board-cranker:command-1"
    runner._run_contractor_agent_sync(
        "prompt",
        _source(),
        _context(tmp_path),
        session_id,
    )

    assert runner._contractor_agents.get(session_id) is None


class _FailingAgent:
    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, *_args, **_kwargs):
        raise RuntimeError("model call failed")

    def close(self):
        pass


def test_contractor_agent_registry_is_evicted_after_failed_turn(
    tmp_path, monkeypatch
):
    runner = _minimal_contractor_runner(tmp_path, monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", _FailingAgent)

    session_id = "contractor:board-cranker:command-1"
    with pytest.raises(RuntimeError):
        runner._run_contractor_agent_sync(
            "prompt",
            _source(),
            _context(tmp_path),
            session_id,
        )

    assert runner._contractor_agents.get(session_id) is None
