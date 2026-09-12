"""Acceptance regressions for catalog authority and just-in-time worker admission."""

import dataclasses
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent.delegation_model_routing import ExecutionLimits, ToolPolicy, WorkspaceContextPolicy
from agent.subagent_lifecycle import (
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    before_worker_tool,
    checkpoint_worker_tool_result,
)
from agent.worker_store import WorkerStore
from hermes_state import SessionDB


class Child:
    def __init__(self, ident, tools):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "fixture"
        self.model = "fixture-model"
        self.ephemeral_system_prompt = "frozen worker prompt"
        self.valid_tool_names = set(tools)
        self._executable_tool_names = set(tools)
        self.tools = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in sorted(tools)
        ]
        self._worker_route_receipt = {}
        self.interrupted = False

    def close(self):
        return None

    def hard_interrupt(self, _reason, *, tool_reason=None):
        self.interrupted = True
        self.tool_reason = tool_reason
        return True


def _credentials(profile="cold"):
    return {
        "provider": "fixture", "model": "fixture-model", "base_url": None,
        "api_key": None, "api_mode": "chat_completions", "request_overrides": {},
        "command": None, "args": [], "requested_profile": profile,
        "requested_provider": None, "requested_model": None,
        "requested_reasoning_effort": None, "resolved_provider": "fixture",
        "resolved_model": "fixture-model", "resolved_reasoning_effort": None,
        "route_provenance": f"delegation.profiles.{profile}", "normalization_events": [],
        "transmitted_model": None, "provider_reported_model": None,
        "fallback_model": [], "reasoning_config": None, "max_iterations": 4,
        "supports_tools": True, "tool_policy": ToolPolicy(allowed_toolsets=("file",)),
        "workspace_context": WorkspaceContextPolicy(mode="none"),
        "execution_limits": ExecutionLimits(max_followups=8, max_tool_calls=2),
        "profile_instructions": "stable contract",
    }


def _install_runtime(monkeypatch, parent, *, run=None, credential_error=None):
    cfg = {
        "max_concurrent_children": 2,
        "max_iterations": 8,
        "profiles": {"cold": {
            "instructions": "stable contract", "provider": "fixture",
            "model": "fixture-model", "tool_policy": {"allowed_toolsets": ["file"]},
            "execution_limits": {"max_followups": 8, "max_tool_calls": 2},
        }},
    }
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)

    def resolve(*_args, **_kwargs):
        if credential_error:
            raise RuntimeError(credential_error)
        return _credentials()

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", resolve)
    built = []

    def build(**kwargs):
        child = Child(f"child-{len(built)}", parent.valid_tool_names)
        child._requested_worker_interface_contract = kwargs.get("worker_interface_contract")
        child._delegate_depth = kwargs.get("retained_child_depth") or 1
        child.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or child.ephemeral_system_prompt
        built.append(child)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    if run is not None:
        monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    return cfg, built


def _clear_worker_records(worker_id):
    from agent import subagent_lifecycle as module

    with module._REGISTRY.lock:
        for key, value in list(module._REGISTRY.records.items()):
            if value.worker_id == worker_id:
                module._REGISTRY.records.pop(key, None)


def test_public_resume_rehydrates_exact_cold_fifo_before_new_turn(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(
        session_id="cold-owner", enabled_toolsets=["file"],
        valid_tool_names={"read_file"}, _session_db=db,
    )
    ran = []

    def run(_index, goal, child, _parent):
        ran.append(goal)
        history = list(getattr(child, "_worker_resume_history", []) or [])
        history.extend([
            {"role": "user", "content": goal},
            {"role": "assistant", "content": f"done:{goal}"},
        ])
        child._worker_last_history = history
        return {"status": "completed", "summary": f"done:{goal}", "api_calls": 1}

    _cfg, built = _install_runtime(monkeypatch, parent, run=run)
    service = SubagentLifecycleService(lambda: parent)
    first = service.launch(SubagentLaunchRequest(goal="first", profile="cold"))
    assert service.wait(first, timeout_seconds=2).state is SubagentState.SUCCEEDED
    store = WorkerStore(db)
    _clear_worker_records(first.worker_id)

    created = time.time()
    subagent_id = "cold-pending-handle"
    capability = service._capability(subagent_id, parent.session_id, created)
    second_run = store.enqueue_run(
        first.worker_id, parent.session_id, goal="second",
        previous_run_id=first.run_id,
        capability_digest=hashlib.sha256(capability.encode()).hexdigest(),
    )
    second = dataclasses.replace(
        first, subagent_id=subagent_id, created_at=created,
        capability=capability, run_id=second_run["run_id"],
    )

    third = service.resume(second, "third")
    assert third.run_id != second.run_id
    assert service.wait(third, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert ran == ["first", "second", "third"]
    runs = store.list_runs(first.worker_id, parent.session_id)
    assert [item["run_id"] for item in runs] == [first.run_id, second.run_id, third.run_id]
    assert [item["status"] for item in runs] == ["SUCCEEDED", "SUCCEEDED", "SUCCEEDED"]
    assert built[0]._requested_worker_interface_contract is None
    assert all(
        child._requested_worker_interface_contract["name"] == "hermes"
        for child in built[1:]
    )
    db.close()


@pytest.mark.parametrize("revocation", ["credential", "tool", "profile"])
def test_warm_queued_run_revalidates_current_authority_before_launch(
    tmp_path, monkeypatch, revocation,
):
    db = SessionDB(tmp_path / f"{revocation}.db")
    parent = SimpleNamespace(
        session_id=f"warm-{revocation}", enabled_toolsets=["file"],
        valid_tool_names={"read_file"}, _session_db=db,
    )
    first_started = threading.Event()
    release_first = threading.Event()
    ran = []

    def run(_index, goal, child, _parent):
        ran.append(goal)
        if goal == "first":
            first_started.set()
            assert release_first.wait(3)
        child._worker_last_history = [{"role": "assistant", "content": goal}]
        return {"status": "completed", "summary": goal, "api_calls": 1}

    cfg, _built = _install_runtime(monkeypatch, parent, run=run)
    service = SubagentLifecycleService(lambda: parent)
    first = service.launch(SubagentLaunchRequest(goal="first", profile="cold"))
    assert first_started.wait(1)
    queued = service.resume(first, "queued")
    record = service._record(queued)
    assert record is not None and record.future is None and record.agent is None

    if revocation == "credential":
        monkeypatch.setattr(
            "tools.delegate_tool_config._resolve_delegation_credentials",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic credential unavailable")),
        )
    elif revocation == "tool":
        parent.valid_tool_names = set()
    else:
        cfg["profiles"]["cold"]["instructions"] = "changed contract"
    release_first.set()
    result = service.wait(queued, timeout_seconds=2)
    assert result.state is SubagentState.FAILED
    failed = WorkerStore(db).get_run(queued.run_id, parent.session_id)
    assert failed["result"]["error_classification"] == "AUTHORITY_REVALIDATION_FAILED"
    assert ran == ["first"]
    db.close()


def test_actor_authorization_precedes_queue_claim_and_uses_retained_parent(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    root = store.create_worker("tree-owner", worker_id="root")
    actor_worker = store.create_worker("tree-owner", parent_worker_id=root["worker_id"], worker_id="actor")
    sibling = store.create_worker("tree-owner", parent_worker_id=root["worker_id"], worker_id="sibling")
    grandchild = store.create_worker("tree-owner", parent_worker_id=actor_worker["worker_id"], worker_id="grandchild")
    sibling_run = store.enqueue_run(sibling["worker_id"], "tree-owner", goal="sibling queued")
    grandchild_run = store.enqueue_run(grandchild["worker_id"], "tree-owner", goal="needs actor authority")
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {"max_concurrent_children": 2})

    actor = SimpleNamespace(
        session_id="actor-session", _worker_owner_session_id="tree-owner",
        _worker_id=actor_worker["worker_id"], _session_db=db,
    )
    actor_service = SubagentLifecycleService(lambda: actor)
    with pytest.raises(PermissionError):
        actor_service.control("status", worker_id=sibling["worker_id"])
    assert store.get_run(sibling_run["run_id"], "tree-owner")["status"] == "PENDING"

    root_agent = SimpleNamespace(session_id="tree-owner", _session_db=db)
    root_service = SubagentLifecycleService(lambda: root_agent)
    pending = root_service.control(
        "wait", worker_id=grandchild["worker_id"],
        run_id=grandchild_run["run_id"], timeout_seconds=0,
    )
    assert pending["status"] == "PENDING"
    assert store.get_run(grandchild_run["run_id"], "tree-owner")["status"] == "PENDING"
    db.close()


def test_active_subtree_cancel_and_profile_tree_tool_limits(tmp_path, monkeypatch):
    import tools.delegate_tool as delegate_module

    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="cancel-owner", _session_db=db)
    child = Child("active-child", {"read_file", "delegate_task"})
    service = SubagentLifecycleService(lambda: parent)
    handle = service.adopt_delegate_child(
        child, goal="active", context=None, profile=None,
        creds={"execution_limits": ExecutionLimits(max_tool_calls=1)}, cfg={"max_concurrent_children": 2},
    )
    store = WorkerStore(db)
    descendant = store.create_worker(
        parent.session_id, parent_worker_id=handle.worker_id, worker_id="active-grandchild")
    queued = store.enqueue_run(descendant["worker_id"], parent.session_id, goal="queued descendant")

    before_worker_tool(child, "one")
    checkpoint_worker_tool_result(child, [], tool_call_id="one", settled=True)
    with pytest.raises(SubagentLifecycleError, match="tool-call limit"):
        before_worker_tool(child, "two")

    cancelled = service.control("cancel", worker_id=handle.worker_id, run_id=handle.run_id)
    assert set(cancelled["workers_targeted"]) == {handle.worker_id, descendant["worker_id"]}
    assert child.interrupted is True
    assert store.get_run(queued["run_id"], parent.session_id)["status"] == "CANCELLED"

    actor = SimpleNamespace(
        _delegate_depth=1, _delegate_spawn_allowed=True,
        _delegate_profile_max_spawn_depth=0,
        _delegate_profile_max_concurrent_children=1,
    )
    monkeypatch.setattr(delegate_module, "_get_max_spawn_depth", lambda: 4)
    monkeypatch.setattr(delegate_module, "_get_max_concurrent_children", lambda: 4)
    with pytest.raises(ValueError, match="does not permit spawning descendants"):
        delegate_module._validate_spawn_admission(actor, 1)
    actor._delegate_profile_max_spawn_depth = 2
    with pytest.raises(ValueError, match="concurrency limit 1"):
        delegate_module._validate_spawn_admission(actor, 2)
    db.close()


def test_explicit_inspect_returns_visible_conversation_without_hidden_reasoning(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="inspect-owner", _session_db=db)
    store = WorkerStore(db)
    store.ensure_schema()
    worker = store.create_worker(parent.session_id, worker_id="inspect-worker")
    queued = store.enqueue_run(worker["worker_id"], parent.session_id, goal="inspect")
    active = store.claim_run(queued["run_id"], parent.session_id)
    history = [
        {"role": "system", "content": "private frozen prompt"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": [
            {"type": "reasoning", "text": "hidden chain"},
            {"type": "output_text", "text": "visible answer"},
        ], "reasoning": "hidden field"},
        {"role": "tool", "name": "read_file", "tool_call_id": "call-1", "content": "visible result"},
        {"role": "reasoning", "content": "hidden role"},
    ]
    store.finish_run(
        active["run_id"], parent.session_id, active["lease_token"],
        status="SUCCEEDED", result={"summary": "visible answer"}, history=history,
    )
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {})
    inspected = SubagentLifecycleService(lambda: parent).control(
        "inspect", worker_id=worker["worker_id"], run_id=active["run_id"])
    assert inspected["conversation"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "visible answer"},
        {"role": "tool", "content": "visible result", "name": "read_file", "tool_call_id": "call-1"},
    ]
    assert "private frozen prompt" not in json.dumps(inspected)
    assert "hidden" not in json.dumps(inspected)
    db.close()
