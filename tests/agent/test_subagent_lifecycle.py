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
    import agent.subagent_lifecycle as lifecycle_module
    monkeypatch.setattr(lifecycle_module, "_REGISTRY", lifecycle_module._Registry())
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


@pytest.mark.parametrize("other_parent,other_correlation", [
    (False, "same-concurrent"), (True, "same-concurrent"), (False, "different"), (False, None),
])
def test_duplicate_correlation_is_reserved_during_child_construction(
    lifecycle, monkeypatch, other_parent, other_correlation,
):
    first_build_started = threading.Event()
    release_first_build = threading.Event()
    built = []

    def slow_build(**_kwargs):
        child = FakeChild(f"sa-race-{len(built)}")
        built.append(child)
        if len(built) == 1:
            first_build_started.set()
            assert release_first_build.wait(timeout=10)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", slow_build)
    request = SubagentLaunchRequest(goal="x", correlation_id="same-concurrent" if other_correlation else None)
    second_service = (SubagentLifecycleService(lambda: SimpleNamespace(session_id="other-parent"))
                      if other_parent else lifecycle)
    duplicate = bool(request.correlation_id) and not other_parent and other_correlation == request.correlation_id
    handles = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lifecycle.launch, request)
        try:
            assert first_build_started.wait(timeout=5)
            second = executor.submit(second_service.launch,
                                     SubagentLaunchRequest(goal="y", correlation_id=other_correlation))
            if duplicate:
                with pytest.raises(SubagentLifecycleError, match="Duplicate"):
                    second.result(timeout=5)
            else:
                handles.append((second_service, second.result(timeout=5)))
        finally:
            release_first_build.set()
            handles.append((lifecycle, first.result(timeout=5)))
            for service, handle in handles:
                assert service.wait(handle, timeout_seconds=5).state is SubagentState.SUCCEEDED

    assert len(built) == (1 if duplicate else 2)
    # A completed child remains correlated during terminal-result retention.
    if request.correlation_id:
        with pytest.raises(SubagentLifecycleError, match="Duplicate"):
            lifecycle.launch(request)


@pytest.mark.parametrize("failure", ["build", "interrupt", "identity", "handle", "submit", "queued-submit"])
def test_failed_launch_releases_resources_and_allows_retry(lifecycle, monkeypatch, tmp_path, failure):
    import sqlite3
    import agent.subagent_lifecycle as lifecycle_module
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.daemon_pool import DaemonThreadPoolExecutor
    from tools.delegate_tool_child_run import _attach_child, _detach_child

    parent = lifecycle._parent_agent_resolver()
    parent._active_children, parent._active_children_lock = [], threading.RLock()
    built, executed = [], []
    attempts = iter(range(2))

    def build(**_kwargs):
        attempt = next(attempts)
        if attempt == 0 and failure in {"build", "interrupt"}:
            raise (KeyboardInterrupt("build interrupted") if failure == "interrupt" else RuntimeError("build failed"))
        # Real close()/SessionDB ownership, without opening a model client.
        child = AIAgent.__new__(AIAgent)
        child._active_children, child._active_children_lock = [], threading.RLock()
        child.session_id = f"child-{attempt}"
        child._subagent_id = "" if attempt == 0 and failure == "identity" else child.session_id
        child._delegate_depth = "invalid" if attempt == 0 and failure == "handle" else 1
        child._session_db = SessionDB(db_path=tmp_path / f"child-{attempt}.db")
        child._owns_session_db = True
        child.close = Mock(wraps=child.close)
        built.append((child, child._session_db._conn))
        _attach_child(parent, child)
        return child

    def run(_index, _goal, child, _parent):
        executed.append(child.session_id)
        _detach_child(parent, child)
        child.close()
        return {"status": "completed", "summary": "done"}

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    request = SubagentLaunchRequest(goal="x", correlation_id="retry-failed-launch")
    with DaemonThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(lifecycle_module, "_EXECUTOR", executor)
        real_submit, real_adjust = executor.submit, executor._adjust_thread_count

        def reject(*_args, **_kwargs):
            raise RuntimeError("executor rejected")

        if failure == "submit":
            monkeypatch.setattr(executor, "submit", reject)
        elif failure == "queued-submit":
            # The real submit enqueues its WorkItem before attempting worker startup.
            monkeypatch.setattr(executor, "_adjust_thread_count", reject)
        expected = {"interrupt": KeyboardInterrupt, "identity": SubagentLifecycleError,
                    "handle": ValueError}.get(failure, RuntimeError)
        try:
            with pytest.raises(expected):
                lifecycle.launch(request)
            assert parent._active_children == []
            assert not lifecycle_module._REGISTRY.records
            assert not lifecycle_module._REGISTRY.correlations
            if built:
                child, connection = built[0]
                child.close.assert_called_once()
                with pytest.raises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

            monkeypatch.setattr(executor, "submit", real_submit)
            monkeypatch.setattr(executor, "_adjust_thread_count", real_adjust)
            handle = lifecycle.launch(request)
            assert lifecycle.wait(handle, timeout_seconds=5).state is SubagentState.SUCCEEDED
            assert executed == [handle.subagent_id]
        finally:
            for child, _connection in built:
                _detach_child(parent, child)
                child.close()


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
