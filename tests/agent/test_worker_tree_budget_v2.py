"""Integrated acceptance for admission-v2 FIFO, catalog refresh, and durable tree budgets."""

import time
from types import SimpleNamespace

import pytest

from agent.delegation_model_routing import ExecutionLimits
from agent.subagent_lifecycle import (
    SubagentLifecycleError,
    SubagentLifecycleService,
    before_worker_provider_attempt,
    checkpoint_worker_tool_result,
    queue_worker_parent_message,
)
from agent.worker_store import WorkerBudgetExceeded, WorkerStore
from hermes_state import SessionDB


class Child:
    def __init__(self, ident, tools=("delegate_task", "read_file"), depth=1):
        self._subagent_id = ident
        self._delegate_role = "orchestrator"
        self._delegate_depth = depth
        self.provider = "fixture"
        self.model = "fixture-model"
        self.ephemeral_system_prompt = "fixed prompt"
        self.valid_tool_names = set(tools)
        self._executable_tool_names = set(tools)
        self.tools = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in tools
        ]
        self._worker_route_receipt = {}
        self.interrupted = False

    def hard_interrupt(self, _message=None, **_kwargs):
        self.interrupted = True
        return True

    def close(self):
        return None


def _creds(*, iterations=3, tools=3, timeout=60):
    return {
        "requested_profile": None,
        "resolved_provider": "fixture",
        "resolved_model": "fixture-model",
        "resolved_reasoning_effort": None,
        "max_iterations": iterations,
        "execution_limits": ExecutionLimits(
            max_iterations=iterations, max_tool_calls=tools, timeout_seconds=timeout),
    }


def _finish_adopted(child, status="completed"):
    child._worker_last_history = [{"role": "assistant", "content": "done"}]
    SubagentLifecycleService.complete_adopted_child(child, {
        "status": status,
        "summary": "done",
        "exit_reason": "completed" if status == "completed" else "interrupted",
        "api_calls": 0,
    })


def test_existing_rows_migrate_and_nested_budget_spending_survives_reopen(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    db = SessionDB(path)

    def old_schema(conn):
        conn.execute("""CREATE TABLE orchestration_workers (
            worker_id TEXT PRIMARY KEY, owner_session_id TEXT NOT NULL, parent_worker_id TEXT,
            root_worker_id TEXT NOT NULL, depth INTEGER NOT NULL, profile TEXT,
            config_revision TEXT NOT NULL, policy TEXT NOT NULL, frozen_prompt TEXT NOT NULL,
            frozen_prompt_hash TEXT NOT NULL, history TEXT NOT NULL DEFAULT '[]',
            uncertain_side_effect INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        conn.execute("""CREATE TABLE orchestration_runs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL,
            worker_id TEXT NOT NULL, request_id TEXT NOT NULL, previous_run_id TEXT,
            goal TEXT NOT NULL, context TEXT NOT NULL, status TEXT NOT NULL,
            capability_digest TEXT NOT NULL DEFAULT '', lease_token TEXT, lease_expires_at REAL,
            tool_inflight INTEGER NOT NULL DEFAULT 0, tool_inflight_count INTEGER NOT NULL DEFAULT 0,
            uncertain_side_effect INTEGER NOT NULL DEFAULT 0, result TEXT,
            completion_ack INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            UNIQUE(worker_id,request_id))""")
        now = time.time()
        conn.execute("""INSERT INTO orchestration_workers VALUES
            ('legacy','owner',NULL,'legacy',1,NULL,'','{}','prompt','hash','[]',0,?,?)""", (now, now))
        conn.execute("""INSERT INTO orchestration_runs
            (run_id,worker_id,request_id,goal,context,status,created_at,updated_at)
            VALUES ('legacy-run','legacy','legacy-request','old','', 'SUCCEEDED',?,?)""", (now, now))

    db._execute_write(old_schema)
    store = WorkerStore(db)
    store.ensure_schema()
    assert store.get_run("legacy-run", "owner")["budget_epoch_id"] is None

    root = store.create_worker("owner", worker_id="root")
    root_run = store.enqueue_run(
        root["worker_id"], "owner", goal="root", budget_limits={
            "max_iterations": 2, "max_tool_calls": 2, "timeout_seconds": 60})
    root_active = store.claim_run(root_run["run_id"], "owner", max_concurrent=2)
    child = store.create_worker("owner", parent_worker_id=root["worker_id"], worker_id="child")
    child_run = store.enqueue_run(
        child["worker_id"], "owner", goal="child",
        budget_epoch_id=root_active["budget_epoch_id"],
        budget_limits={"max_iterations": 2, "max_tool_calls": 2, "timeout_seconds": 60})
    child_active = store.claim_run(child_run["run_id"], "owner", max_concurrent=2)
    store.reserve_iteration(root_active["run_id"], "owner", root_active["lease_token"])
    store.reserve_iteration(child_active["run_id"], "owner", child_active["lease_token"])
    with pytest.raises(WorkerBudgetExceeded, match="tree iteration budget exhausted"):
        store.reserve_iteration(child_active["run_id"], "owner", child_active["lease_token"])
    store.mark_tool_boundary(root_active["run_id"], "owner", root_active["lease_token"], tool_call_id="root-tool")
    store.checkpoint_tool_result(root_active["run_id"], "owner", root_active["lease_token"],
                                 history=[], tool_call_id="root-tool")
    store.mark_tool_boundary(child_active["run_id"], "owner", child_active["lease_token"], tool_call_id="child-tool")
    with pytest.raises(WorkerBudgetExceeded, match="tree tool call budget exhausted"):
        store.mark_tool_boundary(child_active["run_id"], "owner", child_active["lease_token"], tool_call_id="extra")
    epoch = root_active["budget_epoch_id"]
    db.close()

    reopened = SessionDB(path)
    try:
        snapshot = WorkerStore(reopened).budget_snapshot(child_active["run_id"], "owner")
        assert snapshot["budget_epoch_id"] == epoch
        assert snapshot["used_iterations"] == snapshot["used_tool_calls"] == 2
        assert snapshot["deadline_at"] is not None
    finally:
        reopened.close()


def test_provider_attempt_reservation_is_at_real_transport_boundary(tmp_path, monkeypatch):
    from agent.turn_api_call import perform_api_call
    from agent import relay_llm
    from hermes_cli import middleware

    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="owner", _session_db=db)
    child = Child("transport")
    service = SubagentLifecycleService(lambda: parent)
    service.adopt_delegate_child(child, goal="bounded", context=None, profile=None,
                                 creds=_creds(iterations=2), cfg={"max_iterations": 2})
    wire = []
    child.api_mode = "chat_completions"
    child.session_id = "child-session"
    child.platform = "subagent"
    child.base_url = None
    child._disable_streaming = True
    child._has_pending_redirect = lambda: False
    child._interruptible_api_call = lambda kwargs: wire.append(kwargs) or SimpleNamespace(model="fixture")
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda kwargs, send, **_kw: send(kwargs))
    monkeypatch.setattr(relay_llm, "execute", lambda kwargs, send, **_kw: send(kwargs))

    def send():
        return perform_api_call(
            child, api_kwargs={"model": "fixture-model"}, _original_api_kwargs={},
            _llm_middleware_trace=[], _moa_prepared_request=None, _retry=SimpleNamespace(),
            thinking_spinner=None, retry_count=0, api_call_count=0, api_request_id="request",
            effective_task_id="task", turn_id="turn", interrupted=False)

    send()
    send()
    with pytest.raises(InterruptedError, match="tree_iteration_budget_exhausted"):
        send()
    assert len(wire) == 2
    assert WorkerStore(db).budget_snapshot(child._worker_run_id, "owner")["used_iterations"] == 2
    _finish_adopted(child, "interrupted")
    db.close()


def test_wait_for_later_run_admits_same_worker_predecessor_then_target(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    worker = store.create_worker("owner", worker_id="worker", policy={"role": "leaf"})
    first = store.enqueue_run(worker["worker_id"], "owner", goal="A")
    active = store.claim_run(first["run_id"], "owner")
    store.finish_run(first["run_id"], "owner", active["lease_token"], status="SUCCEEDED", result={})
    second = store.enqueue_run(worker["worker_id"], "owner", goal="B", previous_run_id=first["run_id"])
    third = store.enqueue_run(worker["worker_id"], "owner", goal="C", previous_run_id=second["run_id"])
    parent = SimpleNamespace(session_id="owner", _session_db=db)
    ran = []

    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: {"max_concurrent_children": 1})
    monkeypatch.setattr(SubagentLifecycleService, "_build_revalidated_child",
                        lambda self, item, *, goal, role: (Child(goal, depth=item["depth"]), _creds(), {}, item["policy"]))
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle",
                        lambda _index, goal, child, _parent: ran.append(goal) or {
                            "status": "completed", "summary": goal, "api_calls": 0})
    result = SubagentLifecycleService(lambda: parent).control(
        "wait", worker_id=worker["worker_id"], run_id=third["run_id"], timeout_seconds=2)
    assert result["status"] == "SUCCEEDED"
    assert ran == ["B", "C"]
    assert [item["status"] for item in store.list_runs(worker["worker_id"], "owner")] == [
        "SUCCEEDED", "SUCCEEDED", "SUCCEEDED"]
    db.close()


def test_refresh_preserves_authorized_deferred_bridges(monkeypatch):
    import model_tools
    from tools import mcp_tool_agent, tool_search

    def tool(name):
        return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}

    allowed, denied = "mcp__fixture__allowed", "mcp__fixture__denied"
    bridges = [tool(name) for name in sorted(tool_search.BRIDGE_TOOL_NAMES)]
    agent = SimpleNamespace(
        tools=bridges, valid_tool_names=set(tool_search.BRIDGE_TOOL_NAMES),
        _executable_tool_names={allowed}, _worker_effective_tool_names=frozenset({allowed}),
        enabled_toolsets=None, disabled_toolsets=None,
    )
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs:
                        [tool(allowed), tool(denied)] if kwargs.get("skip_tool_search_assembly") else bridges)
    monkeypatch.setattr(tool_search, "is_deferrable_tool_name", lambda name, _deferred=None: name == allowed)
    mcp_tool_agent.refresh_agent_mcp_tools(agent, content_aware=True)
    assert agent._executable_tool_names == {allowed}
    assert agent.valid_tool_names == set(tool_search.BRIDGE_TOOL_NAMES)
    assert denied not in agent.valid_tool_names


def test_concurrency_cancellation_and_parent_outbox_use_shared_service(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    db = SessionDB(tmp_path / "state.db")
    root_parent = SimpleNamespace(session_id="owner", _session_db=db)
    service = SubagentLifecycleService(lambda: root_parent)
    root_child = Child("root-child", depth=1)
    root_handle = service.adopt_delegate_child(
        root_child, goal="root", context=None, profile=None, creds=_creds(),
        cfg={"max_iterations": 3, "max_concurrent_children": 2})
    nested_service = SubagentLifecycleService(lambda: root_child)
    nested = Child("nested", depth=2)
    nested_handle = nested_service.adopt_delegate_child(
        nested, goal="nested", context=None, profile=None, creds=_creds(),
        cfg={"max_iterations": 3, "max_concurrent_children": 2})
    blocked = Child("blocked", depth=2)
    with pytest.raises(SubagentLifecycleError, match="execution slot"):
        nested_service.adopt_delegate_child(
            blocked, goal="blocked", context=None, profile=None, creds=_creds(),
            cfg={"max_iterations": 3, "max_concurrent_children": 2})

    queued = queue_worker_parent_message(root_child, "durable progress")
    assert queued["status"] == "QUEUED"
    _finish_adopted(root_child)
    completions = service.control("completions")["completions"]
    published = next(item for item in completions if item["run_id"] == root_handle.run_id)
    assert published["messages_to_parent"] == [{
        "message_id": queued["message_id"], "content": "durable progress", "status": "PUBLISHED"}]
    service.control("ack", worker_id=root_handle.worker_id, run_id=root_handle.run_id)
    assert WorkerStore(db).list_parent_messages(root_handle.run_id, "owner")[0]["status"] == "ACKNOWLEDGED"

    # A process loss before normal publication retains the payload in the
    # same durable completion outbox; recovery publishes, but does not claim
    # that the parent consumed it.
    store = WorkerStore(db)
    crashed_worker = store.create_worker("owner", worker_id="crashed")
    crashed_run = store.enqueue_run(crashed_worker["worker_id"], "owner", goal="crash")
    crashed_active = store.claim_run(crashed_run["run_id"], "owner", lease_seconds=0.01, max_concurrent=4)
    crashed_message = store.enqueue_parent_message(
        crashed_active["run_id"], "owner", "survives restart")
    time.sleep(0.02)
    store.recover_expired_runs("owner", [crashed_worker["worker_id"]])
    recovered = next(
        item for item in service.control("completions")["completions"]
        if item["run_id"] == crashed_run["run_id"])
    assert recovered["messages_to_parent"] == [{
        "message_id": crashed_message["message_id"],
        "content": "survives restart",
        "status": "PUBLISHED",
    }]

    cancelled = service.control("cancel", worker_id=nested_handle.worker_id, run_id=nested_handle.run_id)
    assert nested.interrupted is True
    assert cancelled["live_interrupts"] == 1
    _finish_adopted(nested, "interrupted")
    db.close()
