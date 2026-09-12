"""Contract tests for the public plugin subagent lifecycle API."""

import dataclasses
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
from agent.worker_store import WorkerStore
from hermes_state import SessionDB


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


def test_public_launch_rejects_unenforced_filesystem_guarantee(lifecycle):
    with pytest.raises(SubagentLifecycleError, match="working_directory is not supported"):
        lifecycle.launch(SubagentLaunchRequest(goal="x", working_directory="/tmp/promised-sandbox"))


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


def test_durable_active_followup_runs_fifo_with_frozen_prompt_and_history(tmp_path, monkeypatch):
    from agent.delegation_model_routing import ExecutionLimits, ToolPolicy, WorkspaceContextPolicy
    from agent import subagent_lifecycle as lifecycle_module

    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="owner-durable", enabled_toolsets=["file"], _session_db=db)
    cfg = {
        "max_iterations": 20,
        "max_concurrent_children": 2,
        "profiles": {"review": {
            "provider": "fixture",
            "model": "review-model",
            "execution_limits": {"max_followups": 2, "max_tool_calls": 3},
        }},
    }
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)

    def credentials(*_args, **_kwargs):
        return {
            "provider": "fixture", "model": "review-model", "base_url": None,
            "api_key": None, "api_mode": "chat_completions", "request_overrides": {},
            "command": None, "args": [], "requested_profile": "review",
            "requested_provider": None, "requested_model": None, "requested_reasoning_effort": None,
            "resolved_provider": "fixture", "resolved_model": "review-model",
            "resolved_reasoning_effort": None, "route_provenance": "delegation.profiles.review",
            "normalization_events": [], "transmitted_model": None, "provider_reported_model": None,
            "fallback_model": [], "reasoning_config": None, "max_iterations": 12,
            "supports_tools": True, "tool_policy": ToolPolicy(allowed_toolsets=("file",)),
            "workspace_context": WorkspaceContextPolicy(mode="none"),
            "execution_limits": ExecutionLimits(max_followups=2, max_tool_calls=3),
            "profile_instructions": "",
        }

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", credentials)
    built = []

    def build(**kwargs):
        child = FakeChild(f"sa-durable-{len(built)}")
        child.session_id = f"child-{len(built)}"
        child.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or "frozen-root-prompt"
        child.valid_tool_names = {"read_file"}
        child.tools = [{"type": "function", "function": {"name": "read_file"}}]
        child._worker_route_receipt = dict(kwargs.get("profile_route_receipt") or {})
        child.steered = []
        child.steer = lambda text: child.steered.append(text) or True
        child.close = lambda: None
        built.append((child, kwargs))
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    first_started = threading.Event()
    release_first = threading.Event()
    goals = []

    def run(_index, goal, child, _parent):
        goals.append((goal, list(getattr(child, "_worker_resume_history", None) or [])))
        if len(goals) == 1:
            first_started.set()
            assert release_first.wait(2)
        history = list(getattr(child, "_worker_resume_history", None) or [])
        history.extend([
            {"role": "user", "content": goal},
            {"role": "assistant", "content": f"done-{len(goals)}"},
        ])
        child._worker_last_history = history
        return {
            "status": "completed", "summary": history[-1]["content"], "exit_reason": "completed",
            "api_calls": 1, "duration_seconds": 0.01, "tokens": {"input": 1, "output": 1},
            "cost_status": "unknown", "cost_usd": 0.0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)
    first = service.launch(SubagentLaunchRequest(goal="first turn", profile="review"))
    assert first_started.wait(1)
    record = service._record(first)
    assert record is not None
    assert record.completion_owner == "service"
    SubagentLifecycleService.complete_adopted_child(
        record.agent,
        {"status": "completed", "summary": "inner completion must not publish"},
    )
    store_run = WorkerStore(db).get_run(first.run_id, parent.session_id)
    assert store_run["status"] == "RUNNING"
    assert record.result is None
    live = service.control(
        "message", worker_id=first.worker_id, message="steer the active run")
    assert live["delivery"] == "RUNNING_STEER_PENDING_CHECKPOINT"
    from agent.subagent_lifecycle import (
        before_worker_tool,
        checkpoint_worker_tool_result,
        worker_tool_calls_remaining,
    )
    for index, remaining in enumerate((2, 1, 0)):
        call_id = f"tool-{index}"
        before_worker_tool(record.agent, call_id)
        checkpoint_worker_tool_result(
            record.agent, [], tool_call_id=call_id, settled=True)
        assert worker_tool_calls_remaining(record.agent) == remaining
    with pytest.raises(SubagentLifecycleError, match="tool-call limit"):
        before_worker_tool(record.agent, "tool-over-limit")
    assert record.agent.steered == ["steer the active run"]
    live_message = next(
        item for item in WorkerStore(db).list_messages(first.worker_id, parent.session_id)
        if item["message_id"] == live["message_id"]
    )
    assert live_message["status"] == "DELIVERED"
    service.message(first, "queued message")
    followup = service.resume(first, "second turn")
    assert followup.worker_id == first.worker_id
    assert followup.run_id != first.run_id
    substituted = dataclasses.replace(first, run_id=followup.run_id)
    assert service.reconnect(substituted).connected is False
    assert service.status(followup).state is SubagentState.PENDING
    release_first.set()
    assert service.wait(followup, timeout_seconds=2).state is SubagentState.SUCCEEDED

    store = WorkerStore(db)
    runs = store.list_runs(first.worker_id, parent.session_id)
    assert [item["status"] for item in runs] == ["SUCCEEDED", "SUCCEEDED"]
    assert runs[1]["previous_run_id"] == runs[0]["run_id"]
    assert built[1][1]["frozen_system_prompt"] == "frozen-root-prompt"
    assert goals[1][1][-1] == {"role": "assistant", "content": "done-1"}
    assert "queued message" in goals[1][0]
    assert store.list_messages(first.worker_id, parent.session_id)[0]["status"] == "DELIVERED"

    with lifecycle_module._REGISTRY.lock:
        lifecycle_module._REGISTRY.records.pop(followup.subagent_id, None)
    assert service.reconnect(followup).connected is True
    restarted = service.resume(followup, "third turn")
    assert restarted.worker_id == first.worker_id and restarted.run_id != followup.run_id
    assert service.wait(restarted, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert goals[2][1][-1] == {"role": "assistant", "content": "done-2"}
    inspected = service.control("inspect", worker_id=first.worker_id, run_id=restarted.run_id)
    assert inspected["run"]["result"]["route"]["resolved_model"] == "review-model"
    assert inspected["run"]["result"]["lineage"]["worker_id"] == first.worker_id
    assert inspected["run"]["result"]["cost"]["status"] == "unknown"
    assert service.control(
        "ack", worker_id=first.worker_id, run_id=restarted.run_id)["acknowledged"] is True
    with pytest.raises(SubagentLifecycleError, match="followup limit"):
        service.resume(restarted, "fourth turn")
    forged = followup.__class__(**{**followup.to_dict(), "capability": "forged"})
    assert service.reconnect(forged).connected is False
    db.close()


def test_nested_actor_authority_sibling_policy_and_recursive_cancel(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    root = store.create_worker("tree-owner", worker_id="worker-root")
    child = store.create_worker("tree-owner", parent_worker_id=root["worker_id"], worker_id="worker-child")
    sibling = store.create_worker("tree-owner", parent_worker_id=root["worker_id"], worker_id="worker-sibling")
    grandchild = store.create_worker(
        "tree-owner", parent_worker_id=child["worker_id"], worker_id="worker-grandchild")
    cfg = {"allow_sibling_messaging": False}
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    actor = SimpleNamespace(
        session_id="actor-session",
        _worker_owner_session_id="tree-owner",
        _worker_id=child["worker_id"],
        _session_db=db,
    )
    service = SubagentLifecycleService(lambda: actor)
    assert service.control("status", worker_id=grandchild["worker_id"])["worker_id"] == grandchild["worker_id"]
    service.control("message", worker_id=root["worker_id"], message="child to parent")
    with pytest.raises(PermissionError):
        service.control("status", worker_id=sibling["worker_id"])
    with pytest.raises(PermissionError):
        service.control("message", worker_id=sibling["worker_id"], message="blocked sibling")
    cfg["allow_sibling_messaging"] = True
    assert service.control(
        "message", worker_id=sibling["worker_id"], message="allowed sibling")["status"] == "PENDING"
    store.enqueue_run(child["worker_id"], "tree-owner", goal="queued child")
    store.enqueue_run(grandchild["worker_id"], "tree-owner", goal="queued grandchild")
    cancelled = service.control("cancel", worker_id=child["worker_id"])
    assert set(cancelled["workers_targeted"]) == {child["worker_id"], grandchild["worker_id"]}
    assert cancelled["queued_runs_cancelled"] == 2
    assert store.get_run(store.list_runs(child["worker_id"], "tree-owner")[0]["run_id"], "tree-owner")["status"] == "CANCELLED"
    db.close()


def test_failure_publication_preserves_newer_tool_checkpoint(tmp_path):
    from agent.subagent_lifecycle import (
        before_worker_tool,
        checkpoint_worker_tool_result,
    )

    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="checkpoint-owner", _session_db=db)
    child = FakeChild("checkpoint-child")
    child.session_id = "checkpoint-child-session"
    child.valid_tool_names = {"read_file"}
    child.tools = [{"type": "function", "function": {"name": "read_file"}}]
    child.ephemeral_system_prompt = "frozen"
    child._worker_route_receipt = {}
    service = SubagentLifecycleService(lambda: parent)
    handle = service.adopt_delegate_child(
        child, goal="checkpoint", context=None, profile=None, creds={}, cfg={})
    record = service._record(handle)
    history = [
        {"role": "user", "content": "checkpoint"},
        {"role": "assistant", "content": "calling tool"},
        {"role": "tool", "tool_call_id": "read-1", "content": "durable result"},
    ]
    before_worker_tool(child, "read-1")
    checkpoint_worker_tool_result(
        child, history, tool_call_id="read-1", settled=True)
    child._worker_last_history = []

    service.complete_adopted_child(
        child,
        {
            "status": "error",
            "error": "later model failure",
            "summary": None,
            "api_calls": 1,
            "duration_seconds": 0.01,
        },
    )

    assert record.result.terminal_state is SubagentState.FAILED
    assert WorkerStore(db).get_worker(handle.worker_id, parent.session_id)["history"] == history
    db.close()


def test_cold_pending_run_rehydrates_fifo_and_revalidates_authority(tmp_path, monkeypatch):
    from agent.delegation_model_routing import ExecutionLimits, ToolPolicy, WorkspaceContextPolicy
    from agent import subagent_lifecycle as lifecycle_module

    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(
        session_id="cold-owner", enabled_toolsets=["file"],
        valid_tool_names={"read_file"}, _session_db=db)
    cfg = {
        "max_concurrent_children": 2,
        "profiles": {"cold": {
            "instructions": "frozen contract",
            "provider": "fixture",
            "model": "cold-model",
            "tool_policy": {"allowed_toolsets": ["file"]},
        }},
    }
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)

    def credentials(*_args, **_kwargs):
        return {
            "provider": "fixture", "model": "cold-model", "base_url": None,
            "api_key": None, "api_mode": "chat_completions", "request_overrides": {},
            "command": None, "args": [], "requested_profile": "cold",
            "requested_provider": None, "requested_model": None,
            "requested_reasoning_effort": None, "resolved_provider": "fixture",
            "resolved_model": "cold-model", "resolved_reasoning_effort": None,
            "route_provenance": "delegation.profiles.cold", "normalization_events": [],
            "transmitted_model": None, "provider_reported_model": None,
            "fallback_model": [], "reasoning_config": None, "max_iterations": 8,
            "supports_tools": True, "tool_policy": ToolPolicy(allowed_toolsets=("file",)),
            "workspace_context": WorkspaceContextPolicy(mode="none"),
            "execution_limits": ExecutionLimits(max_followups=3),
            "profile_instructions": cfg["profiles"]["cold"]["instructions"],
        }

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", credentials)
    built = []

    def build(**kwargs):
        child = FakeChild(f"cold-child-{len(built)}")
        child.session_id = f"cold-session-{len(built)}"
        child.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or "frozen prompt"
        child.valid_tool_names = set(parent.valid_tool_names)
        child.tools = [
            {"type": "function", "function": {"name": name}}
            for name in sorted(child.valid_tool_names)
        ]
        child._worker_route_receipt = {}
        child.close = lambda: None
        built.append(child)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    goals = []

    def run(_index, goal, child, _parent):
        goals.append(goal)
        child._worker_last_history = [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": "done"},
        ]
        return {
            "status": "completed", "summary": "done", "exit_reason": "completed",
            "api_calls": 1, "duration_seconds": 0.01, "cost_status": "unknown",
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)
    first = service.launch(SubagentLaunchRequest(goal="first", profile="cold"))
    assert service.wait(first, timeout_seconds=2).state is SubagentState.SUCCEEDED
    store = WorkerStore(db)
    second = store.enqueue_run(
        first.worker_id, parent.session_id, goal="cold queued second",
        previous_run_id=first.run_id, capability_digest="owner-control-only")
    with lifecycle_module._REGISTRY.lock:
        lifecycle_module._REGISTRY.records.pop(first.subagent_id, None)

    waited = service.control(
        "wait", worker_id=first.worker_id, run_id=second["run_id"], timeout_seconds=2)
    assert waited["status"] == "SUCCEEDED"
    assert goals == ["first", "cold queued second"]
    assert [item["run_id"] for item in store.list_runs(first.worker_id, parent.session_id)] == [
        first.run_id, second["run_id"]]

    third = store.enqueue_run(
        first.worker_id, parent.session_id, goal="must revalidate",
        previous_run_id=second["run_id"])
    parent.valid_tool_names = set()
    with lifecycle_module._REGISTRY.lock:
        for key, value in list(lifecycle_module._REGISTRY.records.items()):
            if value.worker_id == first.worker_id:
                lifecycle_module._REGISTRY.records.pop(key, None)
    rejected = service.control(
        "wait", worker_id=first.worker_id, run_id=third["run_id"], timeout_seconds=2)
    assert rejected["status"] == "FAILED"
    failed = store.get_run(third["run_id"], parent.session_id)
    assert failed["result"]["error_classification"] == "AUTHORITY_REVALIDATION_FAILED"
    assert goals == ["first", "cold queued second"]
    db.close()


def test_public_nested_launch_uses_shared_tree_admission_and_lineage(tmp_path, monkeypatch):
    from agent.delegation_model_routing import ExecutionLimits, ToolPolicy, WorkspaceContextPolicy

    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    root_worker = store.create_worker("tree-public", worker_id="tree-public-root")
    root_run = store.enqueue_run(
        root_worker["worker_id"], "tree-public", goal="root",
        budget_limits={"max_iterations": 5, "max_tool_calls": None, "timeout_seconds": None},
    )
    root_run = store.claim_next_run(root_worker["worker_id"], "tree-public")
    parent = SimpleNamespace(
        session_id="child-session",
        _worker_owner_session_id="tree-public",
        _worker_id=root_worker["worker_id"],
        _delegate_depth=1,
        _delegate_spawn_allowed=True,
        enabled_toolsets=["file"],
        valid_tool_names={"read_file", "delegate_task"},
        _session_db=db,
    )
    from agent.subagent_lifecycle import _Record
    parent._worker_lifecycle_record = _Record(
        None, SubagentState.RUNNING, time.time(), store=store,
        owner_session_id="tree-public", worker_id=root_worker["worker_id"],
        run_id=root_run["run_id"], lease_token=root_run["lease_token"],
        budget_epoch_id=root_run["budget_epoch_id"],
    )
    cfg = {
        "max_concurrent_children": 2,
        "max_spawn_depth": 3,
        "profiles": {"nested": {"provider": "fixture", "model": "nested-model"}},
    }
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    monkeypatch.setattr(
        "tools.delegate_tool_config._resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "provider": "fixture", "model": "nested-model", "base_url": None,
            "api_key": None, "api_mode": "chat_completions", "request_overrides": {},
            "command": None, "args": [], "requested_profile": "nested",
            "requested_provider": None, "requested_model": None,
            "requested_reasoning_effort": None, "resolved_provider": "fixture",
            "resolved_model": "nested-model", "resolved_reasoning_effort": None,
            "route_provenance": "delegation.profiles.nested", "normalization_events": [],
            "transmitted_model": None, "provider_reported_model": None,
            "fallback_model": [], "reasoning_config": None, "max_iterations": 5,
            "supports_tools": True, "tool_policy": ToolPolicy(),
            "workspace_context": WorkspaceContextPolicy(mode="none"),
            "execution_limits": ExecutionLimits(max_spawn_depth=1),
            "profile_instructions": "",
        },
    )

    def build(**_kwargs):
        child = FakeChild("nested-child")
        child.session_id = "nested-child-session"
        child.ephemeral_system_prompt = "frozen nested"
        child.valid_tool_names = {"read_file", "delegate_task"}
        child.tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "delegate_task"}},
        ]
        child._worker_route_receipt = {}
        child.close = lambda: None
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "nested done", "exit_reason": "completed",
            "api_calls": 1, "duration_seconds": 0.01, "cost_status": "unknown",
        },
    )
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="nested", profile="nested"))
    assert service.wait(handle, timeout_seconds=2).state is SubagentState.SUCCEEDED
    durable = store.get_worker(handle.worker_id, "tree-public")
    assert durable["parent_worker_id"] == root_worker["worker_id"]
    assert durable["root_worker_id"] == root_worker["worker_id"]
    assert durable["depth"] == 2

    parent._delegate_spawn_allowed = False
    with pytest.raises(SubagentLifecycleError, match="does not permit spawning"):
        service.launch(SubagentLaunchRequest(goal="forbidden", profile="nested"))
    db.close()


def test_dynamic_retained_route_revalidates_original_request_and_revocation(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(
        session_id="dynamic-owner", enabled_toolsets=[], valid_tool_names=set(),
        model="parent", provider="fixture", base_url=None, request_overrides={},
        api_key=None, _session_db=db,
    )
    cfg = {
        "routing_mode": "dynamic",
        "max_concurrent_children": 1,
        "profiles": {"dynamic": {
            "provider": "fixture", "model": "model-a",
            "enabled_routes": [{"provider": "fixture", "model": "model-b"}],
            "execution_limits": {"max_followups": 2},
        }},
    }
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None, **_kwargs: {
            "provider": requested, "model": target_model, "api_key": None,
            "base_url": None, "api_mode": "chat_completions",
        },
    )
    built = []

    def build(**kwargs):
        child = FakeChild(f"dynamic-{len(built)}")
        child.session_id = f"dynamic-session-{len(built)}"
        child.provider = kwargs.get("override_provider")
        child.model = kwargs.get("model")
        child.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or "frozen"
        child.valid_tool_names = set()
        child.tools = []
        child._worker_route_receipt = dict(kwargs.get("profile_route_receipt") or {})
        child.close = lambda: None
        built.append(child)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "exit_reason": "completed",
            "api_calls": 1, "duration_seconds": 0.01, "cost_status": "unknown",
        },
    )
    service = SubagentLifecycleService(lambda: parent)
    first = service.launch(SubagentLaunchRequest(
        goal="first", profile="dynamic", model="model-b"))
    assert service.wait(first, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert first.model == "model-b"

    second = service.resume(first, "second")
    assert service.wait(second, timeout_seconds=2).state is SubagentState.SUCCEEDED
    assert second.model == "model-b"

    cfg["profiles"]["dynamic"]["enabled_routes"] = []
    with pytest.raises(ValueError, match="not enabled"):
        service.resume(second, "route was revoked")
    assert len(WorkerStore(db).list_runs(first.worker_id, parent.session_id)) == 2
    db.close()
