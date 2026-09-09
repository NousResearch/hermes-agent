"""Focused increment-3 proof over the real Kanban and WorkerStore owners."""

from contextvars import ContextVar
import json
from types import SimpleNamespace
import threading
import time

import pytest

from agent.shared_discovery import build_local_discovery_scope
from agent.team_orchestration import TeamOrchestrationService
from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService, SubagentState
from agent.worker_interfaces import (
    CANONICAL_TEAM_TOOL,
    InterfaceSelection,
    bind_worker_interface,
    canonical_worker_capability,
    dispatch_worker_interface_call,
    normalize_worker_call,
    project_worker_tool_definitions,
)
from agent.worker_store import WorkerStore
from gateway import hosted_rooms
from hermes_state import SessionDB
from tui_gateway.session_discovery import build_gateway_discovery_scope


TEAM_TOOLS = {
    "kanban_team", "delegate_task", "kanban_create", "kanban_heartbeat",
    "kanban_request_review", "kanban_request_changes", "kanban_complete",
    "kanban_block",
}


class ControlledExecution:
    """Real durable lifecycle with only provider execution held at its boundary."""

    def __init__(self, service, db, records, builds):
        self.service, self.store = service, WorkerStore(db)
        self.records, self.builds = records, builds

    def succeed(self, run_ref):
        run_id = run_ref.partition(":")[2]
        record = self.records[run_id]
        self.store.finish_run(
            run_id, record.owner_session_id, record.lease_token, status="SUCCEEDED",
            result={"summary": "fixture evidence", "termination": {"status": "SUCCEEDED"}},
            history=[],
        )
        record.state = SubagentState.SUCCEEDED

    def status(self, run_ref):
        run_id = run_ref.partition(":")[2]
        return self.store.get_run(run_id, self.records[run_id].owner_session_id)["status"]


def _definitions(*extra):
    names = ["delegate_task", "kanban_team", *extra]
    return [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in names
    ]


def _agent(scope, *, tools=TEAM_TOOLS):
    return SimpleNamespace(
        session_id="owner-synthetic",
        _shared_discovery_scope=scope,
        _worker_effective_tool_names=set(tools),
        _executable_tool_names=set(tools),
        valid_tool_names=set(tools),
        tools=[],
        provider="fixture",
        model="fixture",
    )


def _service(tmp_path, monkeypatch, *, tools=TEAM_TOOLS):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    board_db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    db = SessionDB(home / "state.db")
    scope = build_local_discovery_scope()
    agent = _agent(scope, tools=tools)
    agent._session_db = db
    service = TeamOrchestrationService(agent)
    cfg = {"max_iterations": 10, "max_concurrent_children": 2, "profiles": {}}
    monkeypatch.setattr("tools.delegate_tool_config._load_config", lambda: cfg)
    monkeypatch.setattr("tools.delegate_tool_config._get_child_timeout", lambda: None)
    monkeypatch.setattr("tools.delegate_tool._validate_spawn_admission", lambda *_args: None)

    def credentials(_cfg, _parent, profile=None, **_kwargs):
        return {
            "provider": "fixture", "model": f"{profile or 'default'}-model", "base_url": None,
            "api_key": None, "api_mode": "chat_completions", "request_overrides": {},
            "command": None, "args": [], "requested_profile": profile,
            "requested_provider": None, "requested_model": None,
            "requested_reasoning_effort": None, "resolved_provider": "fixture",
            "resolved_model": f"{profile or 'default'}-model", "resolved_reasoning_effort": None,
            "route_provenance": "fixture", "normalization_events": [], "reasoning_config": None,
            "max_iterations": 10, "execution_limits": SimpleNamespace(
                max_followups=8, max_tool_calls=20, timeout_seconds=None,
            ),
        }

    monkeypatch.setattr("tools.delegate_tool_config._resolve_delegation_credentials", credentials)
    builds = []

    class Child:
        def __init__(self, kwargs):
            self._subagent_id = f"fixture-child-{len(builds)}"
            self._delegate_role = kwargs.get("role", "leaf")
            self._delegate_depth = 1
            self.provider, self.model = "fixture", kwargs.get("model")
            self.ephemeral_system_prompt = kwargs.get("frozen_system_prompt") or "fixture prompt"
            self._worker_effective_tool_names = {"read_file"}
            self._executable_tool_names = {"read_file"}
            self.valid_tool_names = {"read_file"}
            self.closed = False
            self.interrupted = False
            self.steered = []

        def close(self):
            self.closed = True

        def steer(self, message):
            self.steered.append(message)
            return True

        def hard_interrupt(self, _reason, **_kwargs):
            self.interrupted = True
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
                    "requested_profile": profile, "requested_provider": None,
                    "requested_model": None, "requested_reasoning_effort": None,
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
    service._start_monitor = lambda *args, **kwargs: None
    return service, ControlledExecution(service.lifecycle, db, records, builds), board_db


def test_team_schema_projection_is_collision_safe_and_keeps_native_authority(monkeypatch):
    selection = bind_worker_interface(
        InterfaceSelection("codex", "explicit", "experimental_unqualified", "fixture", "fixture"),
        _definitions("team_task"),
    )
    aliases = dict(selection.aliases)
    assert aliases["team_task"] == "hermes_worker_team_task"
    projected = project_worker_tool_definitions(_definitions("team_task"), selection)
    names = [item["function"]["name"] for item in projected]
    assert names.count("team_task") == 1
    assert "hermes_worker_team_task" in names
    assert canonical_worker_capability(selection, "team_task") == "team_task"
    assert canonical_worker_capability(selection, "hermes_worker_team_task") == CANONICAL_TEAM_TOOL
    call = normalize_worker_call(
        selection, "hermes_worker_team_task", {"action": "start", "task_ref": "task:one"},
    )
    assert call.operation == "team"
    assert call.arguments == {"action": "start", "task_ref": "task:one"}

    parent = _agent(SimpleNamespace())
    parent._worker_interface_selection = selection

    class Routed:
        def __init__(self, _agent):
            pass

        def dispatch(self, arguments):
            return {"routed": arguments["task_ref"]}

    monkeypatch.setattr("agent.team_orchestration.TeamOrchestrationService", Routed)
    payload = json.loads(dispatch_worker_interface_call(
        parent, "hermes_worker_team_task", {"action": "start", "task_ref": "task:one"},
        lambda _args: pytest.fail("team call reached delegate dispatch"),
    ))
    assert payload["routed"] == "task:one"
    assert payload["orchestration_interface"]["canonical_tool"] == "kanban_team"


def test_parent_mode_claim_attachment_and_worker_admission_are_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="Parent run", assignee="alpha", execution_mode="parent",
        )
        task = kb.claim_task(conn, task_id, claimer="parent-lock", expected_execution_mode="parent")
        assert task is not None and task.execution_mode == "parent"
        assert kb.claim_task(
            conn, task_id, claimer="dispatcher", expected_execution_mode="dispatcher",
        ) is None
        assert not kbd._has_spawnable(conn, "ready")
        reference = {
            "version": "kanban-team-v1", "worker_ref": "worker:one", "run_ref": "run:one",
            "admission_hash": "a" * 64, "role": "implementer", "profile": "alpha",
        }
        attached = kb.attach_execution_reference(
            conn, task_id, task.current_run_id, claim_lock="parent-lock", reference=reference,
        )
        assert attached == reference
        assert kb.attach_execution_reference(
            conn, task_id, task.current_run_id, claim_lock="parent-lock", reference=reference,
        ) == reference
        with pytest.raises(ValueError, match="different execution"):
            kb.attach_execution_reference(
                conn, task_id, task.current_run_id, claim_lock="parent-lock",
                reference={**reference, "run_ref": "run:other"},
            )
    finally:
        conn.close()
    db = SessionDB(tmp_path / "state.db")
    try:
        store = WorkerStore(db)
        store.ensure_schema()
        worker, run = store.admit_team_run(
            "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
            policy={"role": "leaf"}, frozen_prompt="fixed", goal="Implement", context="Task",
            admission_ref="admission-one",
        )
        repeated = store.admit_team_run(
            "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
            policy={"role": "leaf"}, frozen_prompt="fixed", goal="Implement", context="Task",
            admission_ref="admission-one",
        )
        assert repeated[1]["run_id"] == run["run_id"]
        assert worker["worker_id"] == "worker-one"
        with pytest.raises(ValueError, match="immutable request"):
            store.admit_team_run(
                "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
                policy={"role": "leaf"}, frozen_prompt="fixed", goal="Changed", context="Task",
                admission_ref="admission-one",
            )
        assert store.claim_next_run("worker-one", "owner") is None
        assert store.claim_run(run["run_id"], "owner") is None
        leased = store.claim_team_run(
            run["run_id"], "owner", admission_ref="admission-one",
        )
        assert leased is not None and leased["status"] == "RUNNING"
    finally:
        db.close()


def test_team_crash_recovery_uses_one_held_run_and_revalidates_claim(tmp_path, monkeypatch):
    service, lifecycle, _ = _service(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb

    created = service.dispatch({
        "action": "create", "title": "Crash recovery", "profile": "alpha",
        "idempotency_key": "crash-recovery",
    })
    task_id = created["task_ref"].partition(":")[2]
    with service._board() as (scope, conn):
        task, claim_lock, role = service._claim(conn, kb.get_task(conn, task_id))
        admission_hash, worker_id, request_id = service._admission(
            scope, task_id, task.current_run_id, role,
        )
        request = SubagentLaunchRequest(
            goal=task.title, context=f"Parent-managed Kanban task reference: task:{task_id}",
            profile=task.assignee, parent_session_id="owner-synthetic",
        )
        worker, held = service.lifecycle.admit_team_execution(
            request, worker_id=worker_id, request_id=request_id,
            admission_ref=admission_hash,
        )
        assert lifecycle.store.claim_next_run(worker_id, "owner-synthetic") is None
        assert lifecycle.store.claim_run(held["run_id"], "owner-synthetic") is None
        assert service.lifecycle.control(
            "wait", worker_id=worker_id, run_id=held["run_id"], timeout_seconds=0,
        )["status"] == "PENDING"
        reference = {
            "version": "kanban-team-v1", "worker_ref": f"worker:{worker_id}",
            "run_ref": f"run:{held['run_id']}", "admission_hash": admission_hash,
            "role": role, "profile": worker.get("profile"),
        }
        kb.attach_execution_reference(
            conn, task_id, task.current_run_id, claim_lock=claim_lock,
            reference=reference, owner_session_id="owner-synthetic",
        )

    recovered = service.dispatch({"action": "start", "task_ref": created["task_ref"]})
    assert recovered["run_ref"] == reference["run_ref"]
    assert recovered["worker_status"] == "RUNNING"
    assert len(lifecycle.store.list_runs(worker_id, "owner-synthetic")) == 1

    other = service.dispatch({
        "action": "create", "title": "Stale claim", "profile": "beta",
        "idempotency_key": "stale-claim",
    })
    other_id = other["task_ref"].partition(":")[2]
    with service._board() as (scope, conn):
        task, claim_lock, role = service._claim(conn, kb.get_task(conn, other_id))
        digest, worker_id, request_id = service._admission(scope, other_id, task.current_run_id, role)
        request = SubagentLaunchRequest(
            goal=task.title, context=f"Parent-managed Kanban task reference: task:{other_id}",
            profile=task.assignee, parent_session_id="owner-synthetic",
        )
        worker, held = service.lifecycle.admit_team_execution(
            request, worker_id=worker_id, request_id=request_id, admission_ref=digest,
        )
        reference = {
            "version": "kanban-team-v1", "worker_ref": f"worker:{worker_id}",
            "run_ref": f"run:{held['run_id']}", "admission_hash": digest,
            "role": role, "profile": worker.get("profile"),
        }
        kb.attach_execution_reference(
            conn, other_id, task.current_run_id, claim_lock=claim_lock,
            reference=reference, owner_session_id="owner-synthetic",
        )
        conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (int(time.time()) - 1, other_id))
    rejected = service.dispatch({"action": "start", "task_ref": other["task_ref"]})
    assert "stale or mismatched" in rejected["error"]
    assert lifecycle.store.get_run(held["run_id"], "owner-synthetic")["status"] == "PENDING"


def test_two_worker_dependency_review_rejection_retained_correction_and_acceptance(tmp_path, monkeypatch):
    service, lifecycle, _board_db = _service(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    first = service.dispatch({
        "action": "create", "title": "Build", "profile": "alpha", "idempotency_key": "build",
    })
    dependent = service.dispatch({
        "action": "create", "title": "Package", "profile": "beta",
        "parent_refs": [first["task_ref"]], "idempotency_key": "package",
    })
    assert first["status"] == "ready" and dependent["status"] == "todo"

    implementation = service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    assert implementation["worker_status"] == "RUNNING"
    guidance = service.dispatch({
        "action": "guide", "targets": [first["task_ref"]], "message": "Preserve the API.",
        "idempotency_key": "guide-one",
    })
    assert guidance["outcomes"][0]["delivery"] == "RUNNING_STEER_PENDING_CHECKPOINT"
    lifecycle.succeed(implementation["run_ref"])
    submitted = service.dispatch({
        "action": "submit_review", "task_ref": first["task_ref"],
        "summary": "Implemented and checked.", "reviewer": "beta",
    })
    assert submitted["status"] == "review"

    reviewer = service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    assert reviewer["profile"] == "beta"
    lifecycle.succeed(reviewer["run_ref"])
    blocker = service.dispatch({
        "action": "create", "title": "Delayed prerequisite", "profile": "alpha",
        "idempotency_key": "delayed-prerequisite",
    })
    conn = kbc.connect()
    try:
        kb.link_tasks(
            conn, blocker["task_ref"].partition(":")[2], first["task_ref"].partition(":")[2],
        )
    finally:
        conn.close()
    delayed = service.dispatch({
        "action": "request_changes", "task_ref": first["task_ref"],
        "message": "Correct the boundary condition.",
    })
    assert delayed["status"] == "todo"
    conn = kbc.connect()
    try:
        assert kb.complete_task(
            conn, blocker["task_ref"].partition(":")[2], summary="Prerequisite supplied.",
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, first["task_ref"].partition(":")[2]).status == "ready"
    finally:
        conn.close()
    restarted_service = TeamOrchestrationService(service.agent)
    restarted_service._start_monitor = lambda *_args, **_kwargs: None
    correction = restarted_service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    assert correction["worker_ref"] == implementation["worker_ref"]
    assert correction["run_ref"] != implementation["run_ref"]
    correction_run = lifecycle.store.get_run(
        correction["run_ref"].partition(":")[2], "owner-synthetic",
    )
    assert correction_run["previous_run_id"] == implementation["run_ref"].partition(":")[2]
    correction_build = next(
        kwargs for _child, kwargs in lifecycle.builds
        if kwargs.get("goal") == "Correct the boundary condition."
    )
    assert correction_build["retained_parent_worker_id"] is None
    lifecycle.succeed(correction["run_ref"])
    assert restarted_service.dispatch({
        "action": "submit_review", "task_ref": first["task_ref"],
        "summary": "Correction complete.", "reviewer": "beta",
    })["status"] == "review"
    second_review = restarted_service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    reviewer_build = next(
        kwargs for _child, kwargs in reversed(lifecycle.builds)
        if str(kwargs.get("goal", "")).startswith("Review task:")
    )
    assert "Submitted handoff: Correction complete." in reviewer_build["goal"]
    assert "Parent-authorized implementation evidence: status=SUCCEEDED" in reviewer_build["goal"]
    assert "summary=fixture evidence" in reviewer_build["goal"]
    assert correction["worker_ref"] in reviewer_build["goal"]
    assert "do not inspect or control the implementation worker" in reviewer_build["goal"]
    assert "Inspect the exact implementation references" not in reviewer_build["goal"]
    lifecycle.succeed(second_review["run_ref"])
    accepted = restarted_service.dispatch({
        "action": "accept", "task_ref": first["task_ref"], "summary": "Accepted.",
    })
    assert accepted["status"] == "done"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, dependent["task_ref"].partition(":")[2]).status == "ready"
        roles = [
            event.payload.get("role")
            for event in kb.list_events(conn, first["task_ref"].partition(":")[2])
            if event.kind == "execution_attached"
        ]
        assert roles == ["implementer", "reviewer", "correction", "reviewer"]
        review_intents = [
            event.payload for event in kb.list_events(conn, first["task_ref"].partition(":")[2])
            if event.kind == "team_review_intent"
        ]
        assert review_intents[-1]["evidence"]["summary"] == "fixture evidence"
        assert review_intents[-1]["evidence"]["status"] == "SUCCEEDED"
    finally:
        conn.close()


def test_foreign_parent_and_list_only_policy_cannot_mutate(tmp_path, monkeypatch):
    denied, _lifecycle, _ = _service(tmp_path, monkeypatch, tools={"kanban_list"})
    result = denied.dispatch({"action": "create", "title": "Denied", "profile": "alpha"})
    assert "current executable tool policy" in result["error"]
    selection = bind_worker_interface(
        InterfaceSelection("codex", "explicit", "experimental_unqualified", "fixture", "fixture"),
        _definitions(),
    )
    denied.agent._worker_interface_selection = selection
    styled_name = dict(selection.aliases)["team_task"]
    styled = json.loads(dispatch_worker_interface_call(
        denied.agent, styled_name,
        {"action": "create", "title": "Still denied", "profile": "alpha"},
        lambda _args: pytest.fail("styled team call reached delegate dispatch"),
    ))
    assert "current executable tool policy" in styled["error"]

    service, lifecycle, _ = _service(tmp_path / "foreign", monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="Foreign", assignee="alpha", initial_status="running",
            session_id="different-parent", execution_mode="parent",
        )
        before = len(kb.list_runs(conn, task_id))
    finally:
        conn.close()
    denied_start = service.dispatch({"action": "start", "task_ref": f"task:{task_id}"})
    assert "Unknown or unavailable" in denied_start["error"]
    conn = kbc.connect()
    try:
        assert len(kb.list_runs(conn, task_id)) == before
        assert kb.get_task(conn, task_id).status == "ready"
        with lifecycle.store.db._read_ctx() as state_conn:
            assert state_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='orchestration_workers'"
            ).fetchone() is None
    finally:
        conn.close()


def test_dispatcher_direct_and_styled_team_calls_fail_before_store_write(tmp_path, monkeypatch):
    service, _lifecycle, board_db = _service(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "dispatcher-owned-task")
    direct = service.dispatch({"action": "create", "title": "Denied", "profile": "alpha"})
    assert "orchestrator-only" in direct["error"]
    assert not board_db.exists()

    selection = bind_worker_interface(
        InterfaceSelection("codex", "explicit", "experimental_unqualified", "fixture", "fixture"),
        _definitions(),
    )
    service.agent._worker_interface_selection = selection
    styled = json.loads(dispatch_worker_interface_call(
        service.agent, dict(selection.aliases)["team_task"],
        {"action": "create", "title": "Denied styled", "profile": "alpha"},
        lambda _args: pytest.fail("styled team call reached delegate dispatch"),
    ))
    assert "orchestrator-only" in styled["error"]
    assert not board_db.exists()


def test_real_bot_chat_injection_authorizes_guidance_with_session_ceiling(tmp_path, monkeypatch):
    service, _lifecycle, _board_db = _service(tmp_path, monkeypatch)
    profile = tmp_path / "home" / "profiles" / "researcher"
    profile.mkdir(parents=True)
    (profile / "profile.yaml").write_text(
        "description: fixture teammate\nui_meta:\n  hermes-bots:\n    shape: cloud\n",
        encoding="utf-8",
    )
    service.agent._session_db.ensure_session("owner-synthetic", source="test")
    assert service.agent._session_db.set_session_title("owner-synthetic", "Bot Chat")
    from tools import bot_mode_dm, bot_mode_probe
    bot_mode_probe._reset_cache_for_tests()
    assert bot_mode_dm.ensure_message_agent_tool(service.agent) is True
    assert "message_agent" not in service.agent._worker_effective_tool_names
    monkeypatch.setattr(
        bot_mode_dm, "message_agent_tool",
        lambda **kwargs: json.dumps({"status": "sent", "to": kwargs["target"]}),
    )
    delivered = service.dispatch({
        "action": "guide", "targets": ["bot:researcher"], "message": "Review the boundary.",
    })
    assert delivered["outcomes"] == [{"status": "sent", "target": "bot:researcher", "to": "researcher"}]

    service.agent.valid_tool_names.remove("message_agent")
    removed = service.dispatch({
        "action": "guide", "targets": ["bot:researcher"], "message": "Must remain denied.",
    })
    assert "Bot guidance is unavailable" in removed["error"]
    service.agent.valid_tool_names.add("message_agent")
    service.agent._bot_mode_protocol = False
    denied = service.dispatch({
        "action": "guide", "targets": ["bot:researcher"], "message": "Must remain denied.",
    })
    assert "Bot guidance is unavailable" in denied["error"]


def test_exact_cancel_ack_does_not_dispose_task_or_newer_run(tmp_path, monkeypatch):
    service, lifecycle, _ = _service(tmp_path, monkeypatch)

    created = service.dispatch({"action": "create", "title": "Cancel", "profile": "alpha"})
    running = service.dispatch({"action": "start", "task_ref": created["task_ref"]})
    cancelled = service.dispatch({
        "action": "cancel", "task_ref": created["task_ref"], "timeout_seconds": 0,
    })
    assert lifecycle.status(running["run_ref"]) == "RUNNING"
    assert cancelled["task_disposition"] == "pending_terminal_worker_evidence"
    assert cancelled["interrupt_requested"] is True

    lifecycle.succeed(running["run_ref"])
    worker_id = running["worker_ref"].partition(":")[2]
    old_run = running["run_ref"].partition(":")[2]
    newer = lifecycle.store.enqueue_run(
        worker_id, "owner-synthetic", goal="later authorized run",
        previous_run_id=old_run,
    )
    assert lifecycle.store.claim_run(newer["run_id"], "owner-synthetic") is not None
    repeated = service.dispatch({
        "action": "cancel", "task_ref": created["task_ref"], "timeout_seconds": 0,
    })
    assert repeated["task_disposition"] == "pending_terminal_worker_evidence"
    assert lifecycle.store.get_run(newer["run_id"], "owner-synthetic")["status"] == "RUNNING"


def test_room_guidance_requires_current_message_grant(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_db = tmp_path / "rooms.db"
    hosted_rooms.create_room(
        room_db, room_id="room-one", name="Room", members=[{"profile": "alpha"}],
        authority_gateway_id="gateway-one",
    )
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id_existing", lambda: "gateway-one")

    sent = []
    service = SimpleNamespace(
        db_path=room_db,
        runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
        send=lambda **kwargs: sent.append(kwargs) or {"event_id": kwargs["event_id"]},
    )
    from tui_gateway import methods_groups
    monkeypatch.setattr(methods_groups, "_service", service)
    cfg = {"orchestration": {"discovery": {"rooms": [{
        "id": "room-one", "actions": ["inspect", "message"], "participants": ["alpha"],
    }]}}}
    current = ContextVar("team-room-record", default=None)
    server = SimpleNamespace(
        _sessions={}, _sessions_lock=threading.RLock(), _current_runtime_session_record=current,
        _current_profile_name=lambda: "default", _load_cfg=lambda: cfg,
    )
    scope = build_gateway_discovery_scope(server, sid="sid-one", cfg=cfg, source="tui")
    agent = _agent(scope)
    record = {"agent": agent, "discovery_scope": scope, "source": "tui"}
    server._sessions["sid-one"] = record
    token = current.set(record)
    try:
        coordinator = TeamOrchestrationService(agent)
        coordinator._start_monitor = lambda *_args, **_kwargs: None
        result = coordinator.dispatch({
            "action": "guide", "targets": ["room:room-one"], "message": "Coordinate.",
            "idempotency_key": "room-guide",
        })
        assert result["outcomes"][0]["status"] == "accepted"
        assert sent[0]["room_id"] == "room-one"
        service.send = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("receipt lost"))
        uncertain = coordinator.dispatch({
            "action": "guide", "targets": ["room:room-one"], "message": "Retry safely.",
            "idempotency_key": "room-uncertain",
        })["outcomes"][0]
        assert uncertain["status"] == "delivery_uncertain"
        assert uncertain["delivery_id"].startswith("team-")
        assert uncertain["reconciliation_ref"] == f"room_event:{uncertain['delivery_id']}"
        cfg["orchestration"]["discovery"]["rooms"][0]["actions"] = ["inspect"]
        denied = coordinator.dispatch({
            "action": "guide", "targets": ["room:room-one"], "message": "Denied.",
        })["outcomes"][0]
        assert denied["status"] == "not_delivered"
        assert "delivery_id" not in denied
    finally:
        current.reset(token)
