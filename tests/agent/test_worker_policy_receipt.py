"""Execution receipts preserve explicit empty authority through durable completion."""

from types import SimpleNamespace

from agent.subagent_lifecycle import (
    SubagentLifecycleService,
    _REGISTRY,
    _profile_policy_snapshot,
)
from agent.worker_store import WorkerStore
from hermes_state import SessionDB


def test_empty_authority_survives_adoption_and_terminal_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="empty-authority-owner", _session_db=db)
    child = SimpleNamespace(
        _subagent_id="empty-authority-child", session_id="empty-authority-session",
        provider="openrouter", model="fixture/model", _delegate_role="leaf",
        _delegate_depth=1, _worker_route_receipt={},
        _worker_effective_tool_names=frozenset(),
        _executable_tool_names={"delegate_task"}, valid_tool_names=set(),
    )
    service = SubagentLifecycleService(lambda: parent)
    try:
        handle = service.adopt_delegate_child(
            child, goal="synthetic receipt", context=None, profile=None,
            creds={}, cfg={"max_iterations": 2},
        )
        store = WorkerStore(db)
        assert store.get_worker(handle.worker_id, parent.session_id)["policy"]["effective_tools"] == []
        assert child._worker_route_receipt["effective_tools"] == []
        service.complete_adopted_child(child, {"status": "completed", "summary": "done"})
        run = store.get_run(handle.run_id, parent.session_id)
        assert run["status"] == "SUCCEEDED"
        assert run["result"]["effective_tools"] == []
        assert run["result"]["route"]["effective_tools"] == []
    finally:
        record = getattr(child, "_worker_lifecycle_record", None)
        if record is not None and record.lease_stop is not None:
            record.lease_stop.set()
        with _REGISTRY.lock:
            _REGISTRY.records.pop(child._subagent_id, None)
        db.close()


def test_only_absent_catalog_uses_legacy_visible_tools():
    empty_catalog = SimpleNamespace(_executable_tool_names=set(), valid_tool_names={"delegate_task"})
    legacy = SimpleNamespace(valid_tool_names={"read_file"})
    assert _profile_policy_snapshot({}, None, {}, child=empty_catalog)[1]["effective_tools"] == []
    assert _profile_policy_snapshot({}, None, {}, child=legacy)[1]["effective_tools"] == ["read_file"]
