"""Terminal retention applies to readers even when no new child is launched."""
from types import SimpleNamespace

import pytest

from agent import subagent_lifecycle as lifecycle


@pytest.mark.parametrize("operation", ["status", "wait", "result", "reconnect", "cancel"])
def test_expired_terminal_handle_is_unavailable_without_another_launch(monkeypatch, operation):
    service = lifecycle.SubagentLifecycleService(lambda: SimpleNamespace(session_id="parent"))
    clock = 10_000.0
    monkeypatch.setattr(lifecycle.time, "time", lambda: clock)
    registry = lifecycle._Registry()
    monkeypatch.setattr(lifecycle, "_REGISTRY", registry)
    for name, terminal in [("finished", True), ("running", False)]:
        handle = lifecycle.SubagentHandle(
            1, name, "parent", name, clock, None, None, "leaf", 1,
            service._capability(name, "parent", clock),
        )
        state = lifecycle.SubagentState.SUCCEEDED if terminal else lifecycle.SubagentState.RUNNING
        result = lifecycle.SubagentResult(handle, state, True) if terminal else None
        registry.records[name] = lifecycle._Record(
            handle, state, clock, result=result, completed_at=clock if terminal else None,
        )
        registry.correlations[("parent", name)] = name
    finished = registry.records["finished"].handle
    running = registry.records["running"].handle
    assert service.result(finished).ready
    clock += lifecycle._TERMINAL_RETENTION_SECONDS + 1

    observed = getattr(service, operation)(finished, **({"reason": "done"} if operation == "cancel" else {}))

    state = observed.terminal_state if operation == "result" else observed.state
    assert state is lifecycle.SubagentState.UNKNOWN
    assert "finished" not in registry.records
    assert ("parent", "finished") not in registry.correlations
    assert service.status(running).state is lifecycle.SubagentState.RUNNING
