"""Focused saved-workflow proof over real Kanban and WorkerStore services."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from types import SimpleNamespace
import threading
import time

import pytest

from agent.shared_discovery import build_local_discovery_scope
from agent.subagent_lifecycle import SubagentLifecycleService, SubagentState
from agent.team_orchestration import TeamOrchestrationService
from agent.worker_interfaces import (
    InterfaceSelection,
    bind_worker_interface,
    dispatch_worker_interface_call,
    project_worker_tool_definitions,
)
from agent.worker_store import WorkerStore
from hermes_state import SessionDB


WORKFLOW_TOOLS = {
    "kanban_team",
    "delegate_task",
    "kanban_create",
    "kanban_list",
    "kanban_show",
    "kanban_heartbeat",
    "kanban_request_review",
    "kanban_request_changes",
    "kanban_complete",
    "kanban_block",
}

PARALLEL_WORKFLOW = {
    "name": "Compare and combine",
    "steps": [
        {
            "key": "left",
            "title": "Inspect the left input",
            "profile": "alpha",
            "reviewer": "checker",
            "max_corrections": 1,
        },
        {
            "key": "right",
            "title": "Inspect the right input",
            "profile": "beta",
            "reviewer": "checker",
            "max_corrections": 1,
        },
        {
            "key": "join",
            "title": "Combine accepted results",
            "profile": "alpha",
            "reviewer": "checker",
            "depends_on": ["left", "right"],
            "max_corrections": 1,
        },
    ],
}


class ControlledExecution:
    """Use the real durable lifecycle while holding provider execution."""

    def __init__(self, db, records, builds):
        self.store = WorkerStore(db)
        self.records = records
        self.builds = builds

    def finish(self, run_ref, status="SUCCEEDED"):
        run_id = run_ref.partition(":")[2]
        record = self.records[run_id]
        self.store.finish_run(
            run_id,
            record.owner_session_id,
            record.lease_token,
            status=status,
            result={
                "summary": f"fixture {status.lower()} evidence",
                "termination": {"status": status},
            },
            history=[],
        )
        record.state = getattr(SubagentState, status)

    def runs_for(self, worker_ref):
        worker_id = worker_ref.partition(":")[2]
        return self.store.list_runs(worker_id, "owner-synthetic")


def _agent(scope, *, session_id="owner-synthetic", tools=WORKFLOW_TOOLS):
    return SimpleNamespace(
        session_id=session_id,
        _shared_discovery_scope=scope,
        _worker_effective_tool_names=set(tools),
        _executable_tool_names=set(tools),
        valid_tool_names=set(tools),
        tools=[],
        provider="fixture",
        model="fixture",
    )


def _service(tmp_path, monkeypatch, *, session_id="owner-synthetic", tools=WORKFLOW_TOOLS):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    board_db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    state_db = SessionDB(home / "state.db")
    scope = build_local_discovery_scope()
    agent = _agent(scope, session_id=session_id, tools=tools)
    agent._session_db = state_db
    service = TeamOrchestrationService(agent)
    cfg = {"max_iterations": 10, "max_concurrent_children": 3, "profiles": {}}
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    monkeypatch.setattr("tools.delegate_tool._validate_spawn_admission", lambda *_args: None)

    def credentials(_cfg, _parent, profile=None, **_kwargs):
        return {
            "provider": "fixture",
            "model": f"{profile or 'default'}-model",
            "base_url": None,
            "api_key": None,
            "api_mode": "chat_completions",
            "request_overrides": {},
            "command": None,
            "args": [],
            "requested_profile": profile,
            "requested_provider": None,
            "requested_model": None,
            "requested_reasoning_effort": None,
            "resolved_provider": "fixture",
            "resolved_model": f"{profile or 'default'}-model",
            "resolved_reasoning_effort": None,
            "route_provenance": "fixture",
            "normalization_events": [],
            "reasoning_config": None,
            "max_iterations": 10,
            "execution_limits": SimpleNamespace(
                max_followups=8,
                max_tool_calls=20,
                timeout_seconds=None,
            ),
        }

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", credentials)
    builds = []

    class Child:
        def __init__(self, kwargs):
            self._subagent_id = f"fixture-child-{len(builds)}"
            self._delegate_role = kwargs.get("role", "leaf")
            self._delegate_depth = 1
            self.provider = "fixture"
            self.model = kwargs.get("model")
            self.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or "fixture prompt"
            self._worker_effective_tool_names = {"read_file"}
            self._executable_tool_names = {"read_file"}
            self.valid_tool_names = {"read_file"}
            self.interruptions = 0

        def close(self):
            return None

        def steer(self, _message):
            return True

        def hard_interrupt(self, _reason, **_kwargs):
            self.interruptions += 1
            return True

    def build(**kwargs):
        child = Child(kwargs)
        builds.append((child, kwargs))
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "agent.subagent_lifecycle._profile_policy_snapshot",
        lambda _cfg, profile, creds, child=None: (
            "fixture-policy-v1",
            {
                "profile_contract": {"name": profile},
                "effective_tools": sorted(getattr(child, "_worker_effective_tool_names", ())),
                "worker_interface": {"version": 1, "style": "hermes", "aliases": {}},
                "route": {
                    "requested_profile": profile,
                    "requested_provider": None,
                    "requested_model": None,
                    "requested_reasoning_effort": None,
                    "resolved_provider": creds["resolved_provider"],
                    "resolved_model": creds["resolved_model"],
                    "resolved_reasoning_effort": None,
                },
            },
        ),
    )
    records = {}

    def hold(_cls, record):
        record.state = SubagentState.RUNNING
        record.conversation_history = list(
            record.store.get_worker(record.worker_id, record.owner_session_id).get("history") or []
        )
        record.agent._worker_lifecycle_record = record
        records[record.run_id] = record

    monkeypatch.setattr(SubagentLifecycleService, "_start_record", classmethod(hold))
    service._start_monitor = lambda *_args, **_kwargs: None
    return service, ControlledExecution(state_db, records, builds), board_db


def _save(service, definition=PARALLEL_WORKFLOW):
    result = service.dispatch({"action": "workflow_save", "definition": definition})
    assert "error" not in result
    return result


def _step_refs(result):
    detail = result.get("workflow") or result
    return {step["step_key"]: step["task_ref"] for step in detail["steps"]}


def _outcome_for(result, task_ref):
    return next(item for item in result["advancement"]["outcomes"] if item["task_ref"] == task_ref)


def _review_and_accept(service, execution, task_ref, implementation, *, reject_once=False):
    execution.finish(implementation["run_ref"])
    submitted = service.dispatch({
        "action": "submit_review",
        "task_ref": task_ref,
        "summary": "Ready for bounded review.",
        "reviewer": "checker",
    })
    assert submitted["status"] == "review"
    reviewer = service.dispatch({"action": "start", "task_ref": task_ref})
    execution.finish(reviewer["run_ref"])
    if reject_once:
        correction = service.dispatch({
            "action": "request_changes",
            "task_ref": task_ref,
            "message": "Correct the bounded fixture.",
        })
        assert correction["worker_ref"] == implementation["worker_ref"]
        execution.finish(correction["run_ref"])
        assert service.dispatch({
            "action": "submit_review",
            "task_ref": task_ref,
            "summary": "Correction ready.",
            "reviewer": "checker",
        })["status"] == "review"
        reviewer = service.dispatch({"action": "start", "task_ref": task_ref})
        execution.finish(reviewer["run_ref"])
        exhausted = service.dispatch({
            "action": "request_changes",
            "task_ref": task_ref,
            "message": "A second correction must be refused.",
        })
        assert "correction limit reached (1/1)" in exhausted["error"]
    accepted = service.dispatch({
        "action": "accept",
        "task_ref": task_ref,
        "summary": "Accepted fixture evidence.",
    })
    assert accepted["status"] == "done"
    return accepted


def test_atomic_identical_admission_conflict_and_partial_graph_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workflows as workflows

    conn = kbc.connect()
    saved = workflows.save_template(conn, PARALLEL_WORKFLOW, created_by="fixture")
    repeated_save = workflows.save_template(
        conn, PARALLEL_WORKFLOW, created_by="fixture", template_id=saved["template_id"],
    )
    assert repeated_save["template_ref"] == saved["template_ref"]
    advanced = workflows.save_template(
        conn,
        {**PARALLEL_WORKFLOW, "name": "Compare and combine v2"},
        created_by="fixture",
        template_id=saved["template_id"],
    )
    assert advanced["version"] == 2
    assert advanced["template_ref"] != saved["template_ref"]
    assert workflows.template_detail(
        conn, saved["template_id"], saved["version"],
    )["name"] == PARALLEL_WORKFLOW["name"]
    conn.close()

    barrier = threading.Barrier(2)

    def invoke_once():
        local = kbc.connect()
        try:
            barrier.wait()
            return workflows.invoke_workflow(
                local,
                saved["template_ref"],
                owner_session_id="owner-synthetic",
                admission_key="same-key",
                input_payload={"sample": 1},
                created_by="fixture",
            )
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: invoke_once(), range(2)))
    assert results[0]["workflow_ref"] == results[1]["workflow_ref"]
    assert results[0]["graph_hash"] == results[1]["graph_hash"]
    assert _step_refs(results[0]) == _step_refs(results[1])

    conn = kbc.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM workflow_invocations").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4
        with pytest.raises(ValueError, match="different immutable content"):
            workflows.invoke_workflow(
                conn,
                saved["template_ref"],
                owner_session_id="owner-synthetic",
                admission_key="same-key",
                input_payload={"sample": 2},
                created_by="fixture",
            )
        changed = workflows.invoke_workflow(
            conn,
            saved["template_ref"],
            owner_session_id="owner-synthetic",
            admission_key="changed-input-key",
            input_payload={"sample": 2},
            created_by="fixture",
        )
        assert changed["workflow_ref"] != results[0]["workflow_ref"]
        assert changed["graph_hash"] != results[0]["graph_hash"]
        before_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        original_create = kb.create_task
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic admission failure")
            return original_create(*args, **kwargs)

        monkeypatch.setattr(kb, "create_task", fail_second)
        with pytest.raises(RuntimeError, match="synthetic admission failure"):
            workflows.invoke_workflow(
                conn,
                saved["template_ref"],
                owner_session_id="owner-synthetic",
                admission_key="rollback-key",
                input_payload={"sample": 3},
                created_by="fixture",
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before_tasks
        assert conn.execute(
            "SELECT 1 FROM workflow_invocations WHERE admission_key='rollback-key'"
        ).fetchone() is None
    finally:
        conn.close()


def test_existing_board_adds_workflow_columns_before_their_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    legacy_path = tmp_path / "legacy.db"
    from hermes_cli import kanban_db_connect as kbc

    initialized = kbc.connect(db_path=legacy_path)
    initialized.close()
    raw = sqlite3.connect(legacy_path)
    try:
        raw.execute("DROP INDEX idx_tasks_workflow")
        raw.execute("ALTER TABLE tasks DROP COLUMN workflow_invocation_id")
        raw.execute("ALTER TABLE tasks DROP COLUMN workflow_template_version")
        raw.commit()
    finally:
        raw.close()

    kbc.init_db(legacy_path)
    migrated = sqlite3.connect(legacy_path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(tasks)")}
        indexes = {row[1] for row in migrated.execute("PRAGMA index_list(tasks)")}
        assert {"workflow_invocation_id", "workflow_template_version"} <= columns
        assert "idx_tasks_workflow" in indexes
    finally:
        migrated.close()


def test_parallel_review_correction_join_and_restart_safe_resume(tmp_path, monkeypatch):
    service, execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service)
    invoked = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "parallel-one",
        "input": {"left": "A", "right": "B"},
    })
    assert "error" not in invoked
    refs = _step_refs(invoked)
    left = _outcome_for(invoked, refs["left"])
    right = _outcome_for(invoked, refs["right"])
    assert left["worker_status"] == right["worker_status"] == "RUNNING"
    assert {step["status"] for step in invoked["advancement"]["workflow"]["steps"]} == {
        "running", "todo"
    }

    repeated = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "parallel-one",
        "input": {"right": "B", "left": "A"},
    })
    assert repeated["workflow_ref"] == invoked["workflow_ref"]
    assert _outcome_for(repeated, refs["left"])["status"] == "observed"
    assert len(execution.runs_for(left["worker_ref"])) == 1
    conflict = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "parallel-one",
        "input": {"left": "changed", "right": "B"},
    })
    assert "different immutable content" in conflict["error"]

    _review_and_accept(service, execution, refs["left"], left, reject_once=True)
    _review_and_accept(service, execution, refs["right"], right)
    restarted = TeamOrchestrationService(service.agent)
    restarted._start_monitor = lambda *_args, **_kwargs: None
    resumed = restarted.dispatch({
        "action": "workflow_resume",
        "workflow_ref": invoked["workflow_ref"],
        "expected_version": 1,
    })
    join = next(item for item in resumed["outcomes"] if item["task_ref"] == refs["join"])
    assert join["worker_status"] == "RUNNING"
    accepted = _review_and_accept(restarted, execution, refs["join"], join)
    assert accepted["workflow_completed"] is True
    inspected = restarted.dispatch({
        "action": "workflow_inspect", "workflow_ref": invoked["workflow_ref"],
    })
    assert inspected["completed"] is True
    assert {step["status"] for step in inspected["steps"]} == {"done"}


def test_native_request_changes_respects_workflow_correction_bound(tmp_path, monkeypatch):
    service, execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service, {
        "name": "No corrections",
        "steps": [{
            "key": "only", "title": "Only", "profile": "alpha",
            "reviewer": "checker", "max_corrections": 0,
        }],
    })
    invoked = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "native-correction-bound",
    })
    task_ref = _step_refs(invoked)["only"]
    implementation = _outcome_for(invoked, task_ref)
    execution.finish(implementation["run_ref"])
    assert service.dispatch({
        "action": "submit_review",
        "task_ref": task_ref,
        "summary": "Ready for review.",
        "reviewer": "checker",
    })["status"] == "review"
    reviewer = service.dispatch({"action": "start", "task_ref": task_ref})
    execution.finish(reviewer["run_ref"])

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        task_id = task_ref.partition(":")[2]
        review_run = kb.get_task(conn, task_id).current_run_id
        ok, reason = kb.request_changes(
            conn, task_id, reason="Native transition must honor the immutable bound.",
            expected_run_id=review_run,
        )
        assert ok is False
        assert reason == "Workflow correction limit reached (0/0)"
        assert kb.get_task(conn, task_id).status == "running"
    finally:
        conn.close()


def test_native_complete_cannot_bypass_required_workflow_review_or_success(tmp_path, monkeypatch):
    service, execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service, {
        "name": "Required review",
        "steps": [{
            "key": "only", "title": "Only", "profile": "alpha",
            "reviewer": "checker", "max_corrections": 1,
        }],
    })
    invoked = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "native-complete-boundary",
    })
    task_ref = _step_refs(invoked)["only"]
    implementation = _outcome_for(invoked, task_ref)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    task_id = task_ref.partition(":")[2]
    try:
        implementation_run = kb.get_task(conn, task_id).current_run_id
        assert kb.complete_task(
            conn, task_id, summary="Native implementation completion bypass.",
            expected_run_id=implementation_run,
        ) is False
    finally:
        conn.close()

    execution.finish(implementation["run_ref"])
    assert service.dispatch({
        "action": "submit_review",
        "task_ref": task_ref,
        "summary": "Ready for review.",
        "reviewer": "checker",
    })["status"] == "review"
    reviewer = service.dispatch({"action": "start", "task_ref": task_ref})
    conn = kbc.connect()
    try:
        review_run = kb.get_task(conn, task_id).current_run_id
        assert kb.complete_task(
            conn, task_id, summary="Native reviewer completion without worker evidence.",
            expected_run_id=review_run,
        ) is False
    finally:
        conn.close()

    execution.finish(reviewer["run_ref"])
    assert service.dispatch({
        "action": "accept", "task_ref": task_ref, "summary": "Accepted evidence.",
    })["status"] == "done"


def test_optional_workflow_reviewer_can_be_selected_and_accepted(tmp_path, monkeypatch):
    service, execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service, {
        "name": "Optional reviewer",
        "steps": [{"key": "only", "title": "Only", "profile": "alpha"}],
    })
    invoked = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "dynamic-reviewer",
    })
    task_ref = _step_refs(invoked)["only"]
    implementation = _outcome_for(invoked, task_ref)
    execution.finish(implementation["run_ref"])
    assert service.dispatch({
        "action": "submit_review",
        "task_ref": task_ref,
        "summary": "Ready for caller-selected review.",
        "reviewer": "checker",
    })["status"] == "review"
    reviewer = service.dispatch({"action": "start", "task_ref": task_ref})
    execution.finish(reviewer["run_ref"])
    accepted = service.dispatch({
        "action": "accept", "task_ref": task_ref, "summary": "Accepted dynamic review.",
    })
    assert accepted["status"] == "done"
    assert accepted["workflow_completed"] is True


def test_pause_and_claim_share_one_board_serialization_point(tmp_path, monkeypatch):
    service, _execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service, {
        "name": "One step",
        "steps": [{"key": "only", "title": "Only", "profile": "alpha"}],
    })
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workflows as workflows

    conn = kbc.connect()
    try:
        admitted = workflows.invoke_workflow(
            conn,
            saved["template_ref"],
            owner_session_id="owner-synthetic",
            admission_key="pause-race",
            input_payload={},
            created_by="fixture",
        )
        task_id = _step_refs(admitted)["only"].partition(":")[2]
        invocation_id = admitted["workflow_ref"].partition(":")[2]
    finally:
        conn.close()

    barrier = threading.Barrier(2)

    def pause():
        local = kbc.connect()
        try:
            barrier.wait()
            return workflows.set_control(
                local,
                invocation_id,
                owner_session_id="owner-synthetic",
                expected_version=1,
                action="pause",
            )
        finally:
            local.close()

    def claim():
        local = kbc.connect()
        try:
            barrier.wait()
            return kb.claim_task(
                local,
                task_id,
                claimer="workflow-race",
                expected_execution_mode="parent",
                expected_workflow_invocation_id=invocation_id,
            )
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        paused_future = pool.submit(pause)
        claimed_future = pool.submit(claim)
        paused = paused_future.result()
        claimed = claimed_future.result()
    assert paused["control_state"] == "paused"
    conn = kbc.connect()
    try:
        current = kb.get_task(conn, task_id)
        if claimed is None:
            assert current.status == "ready"
        else:
            assert current.status == "running"
            assert claimed.current_run_id == current.current_run_id
        assert kb.claim_task(
            conn,
            task_id,
            claimer="after-pause",
            expected_execution_mode="parent",
            expected_workflow_invocation_id=invocation_id,
        ) is None
    finally:
        conn.close()


def test_exact_cancel_is_sticky_and_does_not_replay_uncertain_interrupt(tmp_path, monkeypatch):
    service, execution, _board_db = _service(tmp_path, monkeypatch)
    saved = _save(service, {
        "name": "Cancel branches",
        "steps": [
            {"key": "keep", "title": "Keep", "profile": "alpha", "reviewer": "checker"},
            {"key": "stop", "title": "Stop", "profile": "beta", "reviewer": "checker"},
            {"key": "stale", "title": "Stale", "profile": "beta", "reviewer": "checker"},
            {
                "key": "done-live", "title": "Done with live worker", "profile": "beta",
                "depends_on": ["keep"],
            },
            {
                "key": "join", "title": "Join", "profile": "alpha",
                "depends_on": ["keep", "stop", "stale", "done-live"],
            },
        ],
    })
    invoked = service.dispatch({
        "action": "workflow_invoke",
        "template_ref": saved["template_ref"],
        "admission_key": "cancel-one",
    })
    refs = _step_refs(invoked)
    keep = _outcome_for(invoked, refs["keep"])
    stop = _outcome_for(invoked, refs["stop"])
    stale = _outcome_for(invoked, refs["stale"])
    _review_and_accept(service, execution, refs["keep"], keep)
    advanced = service.dispatch({
        "action": "workflow_resume",
        "workflow_ref": invoked["workflow_ref"],
        "expected_version": 1,
    })
    done_live = next(
        item for item in advanced["outcomes"] if item["task_ref"] == refs["done-live"]
    )

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    conn = kbc.connect()
    try:
        stop_id = refs["stop"].partition(":")[2]
        stale_id = refs["stale"].partition(":")[2]
        done_live_id = refs["done-live"].partition(":")[2]
        assert kb.complete_task(
            conn, done_live_id, summary="Optional review was not requested.",
            expected_run_id=kb.get_task(conn, done_live_id).current_run_id,
        ) is True
        assert kb.block_task(
            conn, stop_id, reason="Native block left the worker execution live.",
            expected_run_id=kb.get_task(conn, stop_id).current_run_id,
        ) is True
        conn.execute(
            "UPDATE tasks SET claim_expires=? WHERE id=?",
            (int(time.time()) - 1, stale_id),
        )
        conn.commit()
        assert kb.release_stale_claims(conn) == 1
        assert kb.get_task(conn, stale_id).status == "ready"
        inspected = service.dispatch({
            "action": "workflow_inspect", "workflow_ref": invoked["workflow_ref"],
        })
        coordinator_id = inspected["coordinator_ref"].partition(":")[2]
        external_id = kb.create_task(
            conn,
            title="External dependent",
            assignee="alpha",
            parents=[coordinator_id],
            session_id="owner-synthetic",
            execution_mode="parent",
        )
    finally:
        conn.close()

    stop_run_id = stop["run_ref"].partition(":")[2]
    stop_child = execution.records[stop_run_id].agent
    stale_child = execution.records[stale["run_ref"].partition(":")[2]].agent
    done_live_child = execution.records[done_live["run_ref"].partition(":")[2]].agent
    real_control = service.lifecycle.control
    lost_receipt = True

    def uncertain_control(action, **kwargs):
        nonlocal lost_receipt
        if action == "interrupt" and kwargs.get("run_id") == stop_run_id and lost_receipt:
            lost_receipt = False
            real_control(action, **kwargs)
            raise RuntimeError("synthetic interrupt receipt loss")
        return real_control(action, **kwargs)

    service.lifecycle.control = uncertain_control
    first = service.dispatch({
        "action": "workflow_cancel",
        "workflow_ref": invoked["workflow_ref"],
        "expected_version": 1,
    })
    assert first["task_disposition"] == "pending_terminal_worker_evidence"
    stop_outcome = next(item for item in first["outcomes"] if item["task_ref"] == refs["stop"])
    assert stop_outcome["status"] == "effect_uncertain"
    assert stop_child.interruptions == 1
    assert stale_child.interruptions == 1
    assert done_live_child.interruptions == 1

    execution.finish(stop["run_ref"], status="CANCELLED")
    execution.finish(stale["run_ref"], status="CANCELLED")
    execution.finish(done_live["run_ref"], status="CANCELLED")
    service.lifecycle.control = real_control
    second = service.dispatch({
        "action": "workflow_cancel",
        "workflow_ref": invoked["workflow_ref"],
        "expected_version": 2,
    })
    assert second["task_disposition"] == "cancelled"
    assert stop_child.interruptions == 1
    assert stale_child.interruptions == 1
    assert done_live_child.interruptions == 1

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, refs["keep"].partition(":")[2]).status == "done"
        assert kb.get_task(conn, refs["done-live"].partition(":")[2]).status == "done"
        assert kb.get_task(conn, refs["stop"].partition(":")[2]).status == "blocked"
        assert kb.get_task(conn, refs["stale"].partition(":")[2]).status == "blocked"
        assert kb.get_task(conn, refs["join"].partition(":")[2]).status == "blocked"
        assert kb.get_task(conn, coordinator_id).status == "blocked"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, external_id).status == "todo"
        assert kb.claim_task(
            conn, external_id, claimer="must-not-release", expected_execution_mode="parent",
        ) is None
    finally:
        conn.close()


def test_readonly_and_styled_paths_preserve_owner_board_and_capability(tmp_path, monkeypatch):
    service, _execution, board_db = _service(tmp_path, monkeypatch)
    listed = service.dispatch({"action": "workflow_list"})
    assert listed["templates"] == [] and listed["schema_present"] is False
    assert not board_db.exists()

    selection = bind_worker_interface(
        InterfaceSelection("codex", "explicit", "experimental_unqualified", "fixture", "fixture"),
        [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("delegate_task", "kanban_team")
        ],
    )
    projected = project_worker_tool_definitions(
        [{"type": "function", "function": {"name": "kanban_team", "parameters": {"type": "object"}}}],
        selection,
    )
    styled = next(item for item in projected if item["function"]["name"] == "team_task")
    assert "workflow_invoke" in styled["function"]["parameters"]["properties"]["action"]["enum"]
    service.agent._worker_interface_selection = selection

    class Routed:
        def __init__(self, _agent):
            pass

        def dispatch(self, arguments):
            return {"routed": arguments["action"]}

    monkeypatch.setattr("agent.team_orchestration.TeamOrchestrationService", Routed)
    payload = json.loads(dispatch_worker_interface_call(
        service.agent,
        "team_task",
        {"action": "workflow_list"},
        lambda _args: pytest.fail("styled workflow call reached worker dispatch"),
    ))
    assert payload["routed"] == "workflow_list"

    claude = bind_worker_interface(
        InterfaceSelection("claude", "explicit", "experimental_unqualified", "fixture", "fixture"),
        [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("delegate_task", "kanban_team")
        ],
    )
    claude_projected = project_worker_tool_definitions(
        [{"type": "function", "function": {"name": "kanban_team", "parameters": {"type": "object"}}}],
        claude,
    )
    claude_tool = next(item for item in claude_projected if item["function"]["name"] == "TeamTask")
    assert "workflow_invoke" in claude_tool["function"]["parameters"]["properties"]["action"]["enum"]
    service.agent._worker_interface_selection = claude
    claude_payload = json.loads(dispatch_worker_interface_call(
        service.agent,
        "TeamTask",
        {"action": "workflow_list"},
        lambda _args: pytest.fail("Claude-style workflow call reached worker dispatch"),
    ))
    assert claude_payload["routed"] == "workflow_list"

    monkeypatch.undo()
    service, _execution, board_db = _service(tmp_path / "scope", monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "dispatcher-task")
    denied = service.dispatch({"action": "workflow_save", "definition": PARALLEL_WORKFLOW})
    assert "orchestrator-only" in denied["error"]
    assert not board_db.exists()
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    saved = _save(service)
    service.agent._worker_interface_selection = claude
    invoked = json.loads(dispatch_worker_interface_call(
        service.agent,
        "TeamTask",
        {
            "action": "workflow_invoke",
            "template_ref": saved["template_ref"],
            "admission_key": "owner-only",
        },
        lambda _args: pytest.fail("Claude workflow invoke reached worker dispatch"),
    ))
    assert "error" not in invoked
    assert invoked["action"] == "workflow_invoke"
    assert invoked["advancement"]["outcomes"]
    foreign = TeamOrchestrationService(_agent(service.agent._shared_discovery_scope, session_id="foreign"))
    foreign_result = foreign.dispatch({
        "action": "workflow_inspect", "workflow_ref": invoked["workflow_ref"],
    })
    assert "Unknown or unavailable" in foreign_result["error"]
    stale_path = tmp_path / "stale" / "other.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(stale_path))
    stale = service.dispatch({"action": "workflow_list"})
    assert "scope is stale" in stale["error"]
    assert not stale_path.exists()
