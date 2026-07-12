"""Contract tests for the public plugin subagent lifecycle API."""

import time
from types import SimpleNamespace

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
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
    lifecycle.launch(SubagentLaunchRequest(goal="x", correlation_id="same"))
    with pytest.raises(SubagentLifecycleError, match="Duplicate"):
        lifecycle.launch(SubagentLaunchRequest(goal="x", correlation_id="same"))
    with pytest.raises(SubagentLifecycleError, match="broaden"):
        lifecycle.launch(
            SubagentLaunchRequest(goal="x", allowed_toolsets=("terminal",))
        )
    with pytest.raises(SubagentLifecycleError, match="working_directory"):
        lifecycle.launch(SubagentLaunchRequest(goal="x", working_directory="C:/"))


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
