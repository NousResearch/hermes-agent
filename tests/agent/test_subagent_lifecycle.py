"""Contract tests for the public plugin subagent lifecycle API."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    bind_subagent_parent,
    get_active_subagent_parent,
)


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None
        self.steering = []

    def steer(self, text):
        self.steering.append(text)
        return True

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)






def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)








def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"




def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None


@pytest.fixture
def steer_lifecycle(monkeypatch):
    from agent.subagent_lifecycle import _REGISTRY
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent

    child = FakeChild("sa-steer-contract")
    started, finish = threading.Event(), threading.Event()
    service = SubagentLifecycleService(lambda: SimpleNamespace(session_id="parent-steer", enabled_toolsets=["file"]))

    def run(_index, _goal, child, _parent):
        _register_subagent({"subagent_id": child._subagent_id, "agent": child})
        try:
            started.set()
            assert finish.wait(5)
            return {"status": "interrupted" if child.interrupted else "completed", "summary": "done"}
        finally:
            _unregister_subagent(child._subagent_id, agent=child)

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    yield service, child, started, finish
    finish.set()
    record = _REGISTRY.records.get(child._subagent_id)
    if record is not None:
        record.future.result(timeout=5)


@pytest.mark.parametrize("wire", [False, True])
def test_steer_queues_to_running_child_and_rejects_invalid_calls(steer_lifecycle, wire):
    from hermes_cli.plugin_host_wire import decode, encode

    service, child, started, finish = steer_lifecycle
    request = SubagentLaunchRequest(goal="review", allowed_toolsets=("file",))
    handle = service.launch(decode(encode(request)) if wire else request)
    assert started.wait(2)
    handle = decode(encode(handle)) if wire else handle
    assert service.status(handle).state is SubagentState.RUNNING
    assert service.reconnect(handle).connected
    assert service.steer(handle, "focus on correctness") is True
    assert child.steering == ["focus on correctness"]
    assert service.steer(handle, "") is False
    assert service.steer(handle, "   ") is False
    other = SubagentLifecycleService(lambda: SimpleNamespace(session_id="foreign"))
    assert other.steer(handle, "foreign") is False
    forged = {**(dict(handle) if wire else handle.to_dict()), "capability": "forged"}
    assert service.steer(forged, "forged") is False
    finish.set()
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert service.result(handle).summary == "done"
    assert service.steer(handle, "too late") is False
    assert child.steering == ["focus on correctness"]


@pytest.mark.parametrize("launch_request, message", [
    ({"goal": "review", "future_field": True}, "future_field"),
    (42, "request must be a SubagentLaunchRequest"),
])
def test_mapping_request_errors_are_explicit(lifecycle, launch_request, message):
    with pytest.raises(SubagentLifecycleError, match=message):
        lifecycle.launch(launch_request)


@pytest.fixture
def routing_lifecycle(lifecycle, monkeypatch):
    from tools.delegate_tool import _build_child_agent

    build = Mock(wraps=_build_child_agent)
    config = Mock(return_value={})
    resolve = Mock(return_value={
        "provider": "openrouter", "model": "runtime-model",
        "base_url": "https://route.example/v1", "api_key": "fixture-key",
        "api_mode": "chat_completions", "request_overrides": {"temperature": 0.2},
    })
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._load_config", config)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    return lifecycle, build, config, resolve


def test_routing_delegation_provider_bundle(routing_lifecycle):
    service, build, config, resolve = routing_lifecycle
    config.return_value = {"provider": "openrouter", "model": "delegated-model"}
    handle = service.launch(SubagentLaunchRequest(goal="route"))
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    kwargs = build.call_args.kwargs
    assert kwargs["model"] == "delegated-model"
    assert kwargs["override_provider"] == "openrouter"
    assert kwargs["override_base_url"] == "https://route.example/v1"
    assert kwargs["override_api_key"] == "fixture-key"
    assert kwargs["override_api_mode"] == "chat_completions"
    assert kwargs["override_request_overrides"] == {"temperature": 0.2}
    assert kwargs["override_acp_command"] is None
    assert kwargs["override_acp_args"] == []
    resolve.assert_called_once_with(requested="openrouter", target_model="delegated-model")
    handle = service.launch(SubagentLaunchRequest(goal="route", model="request-model"))
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert build.call_args.kwargs["model"] == "request-model"


@pytest.mark.parametrize("wire", [False, True])
def test_reasoning_effort_is_a_launch_local_override(routing_lifecycle, wire):
    from hermes_cli.plugin_host_wire import decode, encode

    service, build, config, _resolve = routing_lifecycle
    config.return_value = {"reasoning_effort": "low"}
    request = {"goal": "route", "reasoning_effort": "high"}
    request = decode(encode(request)) if wire else request

    explicit_handle = service.launch(request)
    inherited_handle = service.launch({"goal": "route"})
    # Check both calls before either child finishes: per-launch data stays
    # independent even when launches overlap in the shared service.
    assert [call.kwargs["reasoning_config_override"] for call in build.call_args_list] == [
        {"enabled": True, "effort": "high"}, None,
    ]
    assert service.wait(explicit_handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert service.wait(inherited_handle, timeout_seconds=2).state is SubagentState.SUCCEEDED


@pytest.mark.parametrize("value", [False, "none"])
def test_reasoning_effort_explicit_disable_is_not_omission(routing_lifecycle, value):
    service, build, config, _resolve = routing_lifecycle
    config.return_value = {"reasoning_effort": "high"}

    handle = service.launch({"goal": "route", "reasoning_effort": value})
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert build.call_args.kwargs["reasoning_config_override"] == {"enabled": False}


def test_invalid_reasoning_effort_fails_before_child_build(routing_lifecycle):
    service, build, _config, _resolve = routing_lifecycle

    with pytest.raises(SubagentLifecycleError, match="reasoning_effort"):
        service.launch({"goal": "route", "reasoning_effort": "not-a-real-effort"})

    build.assert_not_called()


@pytest.mark.parametrize("wire", [False, True])
def test_routing_request_provider_wins_without_mixing_bundles(routing_lifecycle, wire):
    from hermes_cli.plugin_host_wire import decode, encode

    service, build, config, resolve = routing_lifecycle
    config.return_value = {
        "provider": "nous", "model": "delegated-model",
        "base_url": "https://delegation.example/v1", "api_key": "delegation-fixture-key",
        "api_mode": "anthropic_messages", "command": "delegation-command",
        "request_overrides": {"temperature": 0.9},
    }
    request = SubagentLaunchRequest(goal="route", provider="openrouter", model="m")
    request = decode(encode(request)) if wire else request
    if wire:
        assert request["provider"] == "openrouter"
    handle = service.launch(request)
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    kwargs = build.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["override_provider"] == "openrouter"
    assert kwargs["override_base_url"] == "https://route.example/v1"
    assert kwargs["override_api_key"] == "fixture-key"
    assert kwargs["override_api_mode"] == "chat_completions"
    assert kwargs["override_request_overrides"] == {"temperature": 0.2}
    assert kwargs["override_acp_command"] is None
    assert kwargs["override_acp_args"] == []
    assert kwargs["routing_cfg"] == {"provider": "openrouter", "model": "m"}
    resolve.assert_called_once_with(requested="openrouter", target_model="m")


@pytest.mark.parametrize("per_launch", [False, True])
def test_routing_unknown_provider_refuses_before_registration(routing_lifecycle, per_launch):
    from agent.subagent_lifecycle import _REGISTRY

    service, build, config, resolve = routing_lifecycle
    resolve.side_effect = ValueError("Unknown provider fixture-missing")
    config.return_value = {} if per_launch else {"provider": "fixture-missing"}
    request = {"goal": "route", "correlation_id": "refused-route"}
    if per_launch:
        request["provider"] = "fixture-missing"
    records, correlations = dict(_REGISTRY.records), dict(_REGISTRY.correlations)
    with pytest.raises(SubagentLifecycleError, match="Unknown provider fixture-missing"):
        service.launch(request)
    assert _REGISTRY.records == records
    assert _REGISTRY.correlations == correlations
    build.assert_not_called()
    resolve.assert_called_once_with(requested="fixture-missing", target_model=None)


def test_plugin_toolsets_are_known_toolsets(lifecycle, monkeypatch):
    monkeypatch.setattr("toolsets._get_plugin_toolset_names", lambda: {"plugin-tools"})
    parent = SimpleNamespace(session_id="parent-plugin", enabled_toolsets=["file", "plugin-tools"])
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="use the plugin", allowed_toolsets=("plugin-tools",)))
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    with pytest.raises(SubagentLifecycleError, match="Unknown toolsets: nope"):
        service.launch(SubagentLaunchRequest(goal="use the plugin", allowed_toolsets=("nope",)))
    narrow = SubagentLifecycleService(lambda: SimpleNamespace(session_id="parent-narrow", enabled_toolsets=["file"]))
    with pytest.raises(SubagentLifecycleError, match="broaden"):
        narrow.launch(SubagentLaunchRequest(goal="use the plugin", allowed_toolsets=("plugin-tools",)))


def test_provider_is_plugin_facing_only():
    """``provider`` lives on the plugin request; no model-facing delegate_task field can carry a route."""
    import dataclasses
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    assert "provider" in {field.name for field in dataclasses.fields(SubagentLaunchRequest)}
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    exposed = set(props) | set(props["tasks"]["items"]["properties"])
    assert not exposed & {"provider", "model", "base_url", "api_key"}
