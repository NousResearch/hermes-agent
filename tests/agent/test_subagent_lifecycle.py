"""Contract tests for the public plugin subagent lifecycle API."""

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

    def interrupt(self, _reason):
        self.interrupted = True


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


def test_launch_wait_result_and_handle_round_trip(lifecycle):
    handle = lifecycle.launch(
        SubagentLaunchRequest(goal="x", allowed_toolsets=("file",))
    )
    assert handle.from_dict(handle.to_dict()) == handle
    assert lifecycle.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED
    first = lifecycle.result(handle)
    assert first.ready and first.summary == "safe summary" and first.result_hash
    assert lifecycle.result(handle) == first


def test_duplicate_correlation_and_permission_validation(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x", correlation_id="same"))
    with pytest.raises(SubagentLifecycleError, match="Duplicate"):
        lifecycle.launch(SubagentLaunchRequest(goal="x", correlation_id="same"))
    with pytest.raises(SubagentLifecycleError, match="broaden"):
        lifecycle.launch(
            SubagentLaunchRequest(goal="x", allowed_toolsets=("terminal",))
        )
    with pytest.raises(SubagentLifecycleError, match="working_directory"):
        lifecycle.launch(SubagentLaunchRequest(goal="x", working_directory="C:/"))
    lifecycle.wait(handle, timeout_seconds=1)


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


def test_simultaneous_launches_are_distinct_and_reconnect_is_in_process(lifecycle):
    handles = [lifecycle.launch(SubagentLaunchRequest(goal="x")) for _ in range(10)]
    assert len({h.subagent_id for h in handles}) == 10
    assert lifecycle.reconnect(handles[0]).connected
    for handle in handles:
        lifecycle.wait(handle, timeout_seconds=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability", []),
        ("contract_version", True),
        ("subagent_id", None),
        ("parent_session_id", []),
        ("correlation_id", []),
        ("created_at", "yesterday"),
        ("provider", []),
        ("model", []),
        ("role", []),
        ("depth", "one"),
    ],
)
def test_malformed_deserialized_handle_is_unknown(lifecycle, field, value):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    malformed = handle.from_dict({**handle.to_dict(), field: value})

    assert lifecycle.status(malformed).state is SubagentState.UNKNOWN
    assert lifecycle.result(malformed).error_classification == "UNKNOWN_HANDLE"
    lifecycle.wait(handle, timeout_seconds=1)


def test_launch_preserves_parent_tool_resolution(monkeypatch):
    import model_tools

    parent = SimpleNamespace(session_id="parent-tools", enabled_toolsets=["file"])
    model_tools._last_resolved_tool_names = ["parent_tool"]

    def build(**_kwargs):
        model_tools._last_resolved_tool_names = ["child_tool"]
        return FakeChild("sa-tools")

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="x"))

    assert model_tools._last_resolved_tool_names == ["parent_tool"]
    assert handle.subagent_id == "sa-tools"
    service.wait(handle, timeout_seconds=1)


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
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"


def test_plugin_context_uses_turn_scoped_parent(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManifest

    parent = SimpleNamespace(session_id="gateway-parent", enabled_toolsets=["file"])
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", lambda **_kwargs: FakeChild("sa-gateway")
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    manager = SimpleNamespace(_cli_ref=None)
    ctx = PluginContext(PluginManifest(name="test", source="test"), manager)

    with bind_subagent_parent(parent):
        handle = ctx.subagent_lifecycle.launch(SubagentLaunchRequest(goal="x"))
        ctx.subagent_lifecycle.wait(handle, timeout_seconds=1)

    assert handle.parent_session_id == "gateway-parent"


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
