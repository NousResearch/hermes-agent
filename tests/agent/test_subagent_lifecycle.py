"""Contract tests for the public plugin subagent lifecycle API."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def test_duplicate_correlation_is_reserved_during_child_construction(
    lifecycle, monkeypatch
):
    first_build_started = threading.Event()
    release_first_build = threading.Event()
    build_count = iter(range(2))

    def slow_build(**_kwargs):
        ident = next(build_count)
        if ident == 0:
            first_build_started.set()
            assert release_first_build.wait(timeout=1)
        return FakeChild(f"sa-race-{ident}")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", slow_build
    )
    request = SubagentLaunchRequest(goal="x", correlation_id="same-concurrent")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lifecycle.launch, request)
        assert first_build_started.wait(timeout=1)
        second = executor.submit(lifecycle.launch, request)
        try:
            with pytest.raises(SubagentLifecycleError, match="Duplicate"):
                second.result(timeout=1)
        finally:
            release_first_build.set()
        handle = first.result(timeout=1)

    assert handle.correlation_id == "same-concurrent"
    lifecycle.wait(handle, timeout_seconds=1)


@pytest.mark.parametrize("failure", ["build", "identity"])
def test_failed_child_construction_releases_correlation_reservation(
    lifecycle, monkeypatch, failure
):
    attempts = iter(range(2))

    def build(**_kwargs):
        if next(attempts) == 0:
            if failure == "build":
                raise RuntimeError("build failed")
            return SimpleNamespace(_subagent_id="")
        return FakeChild(f"sa-retry-{failure}")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    request = SubagentLaunchRequest(goal="x", correlation_id=f"retry-{failure}")

    expected = RuntimeError if failure == "build" else SubagentLifecycleError
    with pytest.raises(expected):
        lifecycle.launch(request)
    handle = lifecycle.launch(request)

    assert handle.subagent_id == f"sa-retry-{failure}"
    lifecycle.wait(handle, timeout_seconds=1)


def test_failed_executor_submission_rolls_back_record_and_correlation(
    lifecycle, monkeypatch
):
    import agent.subagent_lifecycle as lifecycle_module

    executor = lifecycle_module._EXECUTOR
    real_submit = executor.submit

    def reject_submission(*_args, **_kwargs):
        raise RuntimeError("executor rejected")

    monkeypatch.setattr(executor, "submit", reject_submission)
    request = SubagentLaunchRequest(goal="x", correlation_id="retry-after-reject")

    with pytest.raises(RuntimeError, match="executor rejected"):
        lifecycle.launch(request)
    monkeypatch.setattr(executor, "submit", real_submit)
    handle = lifecycle.launch(request)

    assert handle.correlation_id == "retry-after-reject"
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
