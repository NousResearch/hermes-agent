import json

from hermes_os_integration.conversational import (
    ChatEnvelope,
    build_col_config,
    chat_response_from_plan,
    chief_of_staff_plan,
    col_audit_event,
    col_contracts,
    create_conversational_session,
    discover_chat_commands,
    transcript_from_turns,
)
from hermes_os_integration.phase_completion import complete_phases, completion_summary, phase_statuses, task_ids_for_phases
from hermes_os_integration.project_runtime_ops import (
    AgentFleetMember,
    AgentHeartbeat,
    ApprovalRequest,
    ConnectorManifest,
    CrossProjectBlocker,
    CrossProjectDependency,
    EvaluationResult,
    MemoryIndexRecord,
    agent_cost_score,
    agent_fleet_dashboard,
    agent_fleet_export,
    agent_latency_score,
    agent_routing_explanation,
    approval_command_contract,
    approval_dashboard_actions,
    approval_notification_event,
    approval_queue_view,
    apply_quality_gate,
    automation_dashboard_panel,
    automation_failure_report,
    automation_preflight,
    automation_replay_contract,
    automation_schedule_contract,
    blocker_registry,
    build_automation_workflow,
    connector_compatibility,
    connector_health_contract,
    connector_lifecycle_contract,
    cost_aware_evaluation_plan,
    create_approval_request,
    create_policy_override,
    create_telemetry_event,
    dashboard_build_verification,
    detect_memory_drift,
    detect_shared_resource_conflicts,
    detect_slow_operations,
    detect_stale_dependencies,
    detect_stale_heartbeats,
    diagnostics_bundle,
    discover_connectors,
    documentation_pass,
    enforce_memory_source_guard,
    evaluation_dashboard_panel,
    evaluation_export,
    expire_approvals,
    extract_lessons,
    failure_drill,
    index_project_memory,
    integration_suite_target,
    memory_bundle_export,
    memory_bundle_import,
    memory_compaction_plan,
    memory_query_panel,
    memory_refresh_schedule,
    metric_rollup,
    migration_compatibility_matrix,
    multi_project_dashboard_graph,
    multi_project_execution_queue,
    observability_dashboard_trends,
    orchestration_report_export,
    packaging_verification,
    project_run_command_contract,
    project_shutdown_automation,
    project_switch_automation,
    record_automation_run,
    regression_evaluation_set,
    release_checklist,
    release_notes_from_tasks,
    resolve_cross_project_dependencies,
    route_agent,
    rubric_evaluation_contract,
    run_evaluations,
    self_review_workflow,
    shared_infrastructure_map,
    summarize_project_memory,
    telemetry_import_export_compatibility,
    telemetry_integrity_check,
    telemetry_retention_policy,
    validate_connector_permission,
    waive_evaluation,
    write_telemetry_event,
    correlate_trace,
    prune_telemetry_events,
)


def test_phase_56_approval_governance_contracts():
    request = create_approval_request("approval-1", "alpha", requester="planner", scope="runtime", risk="high", action="deploy", expires_at="2026-01-01")
    approved = approval_command_contract("approve", "approval-1", reason="reviewed")
    rejected_without_reason = approval_command_contract("reject", "approval-1")
    override = create_policy_override("override-1", "alpha", action="deploy", reason="emergency fix", approver="operator")
    expired = expire_approvals([request], now="2027-01-01")[0]

    assert approval_queue_view([request], status="pending")["count"] == 1
    assert approved["valid"] is True
    assert rejected_without_reason["valid"] is False
    assert expired.status == "expired"
    assert override.reason == "emergency fix"
    assert {item["action"] for item in approval_dashboard_actions(request)} == {"approve", "reject", "request-more-context"}
    assert approval_notification_event(request)["type"] == "approval.requested"


def test_phase_57_automation_contracts():
    workflow = build_automation_workflow("alpha", ["switch", "restore", "start"])
    run = record_automation_run(workflow)
    switch = project_switch_automation("alpha")
    shutdown = project_shutdown_automation("alpha")

    assert project_run_command_contract("alpha", workflow.workflow_id)["dry_run"] is True
    assert switch.steps[0]["type"] == "memory-load"
    assert shutdown.steps[-1]["type"] == "final-status"
    assert automation_preflight(dirty_worktree=True)["ok"] is False
    assert automation_replay_contract(run)["workflow_id"] == workflow.workflow_id
    assert automation_dashboard_panel([run])["data"]["count"] == 1
    assert automation_schedule_contract("alpha", workflow.workflow_id, "0 8 * * *")["mode"] == "dry_run"
    assert "resolve preflight issues" in automation_failure_report(workflow, "step-001", "failed")["next_actions"]


def test_phase_58_multi_project_orchestration_contracts():
    dep = CrossProjectDependency("api", "web", "schema release", blocked_since="48")
    blocker = CrossProjectBlocker("blocker-1", "api", "web", "schema not released", age_hours=48)
    actions = [{"project_id": "api", "resource": "db"}, {"project_id": "web", "resource": "db"}]

    assert resolve_cross_project_dependencies([dep])["blocked_count"] == 1
    assert blocker_registry([blocker])["aged_count"] == 1
    assert multi_project_execution_queue([{"project_id": "api", "priority": 2}, {"project_id": "web", "priority": 1}])[0]["project_id"] == "api"
    assert shared_infrastructure_map(actions)["resources"]["db"] == ["api", "web"]
    assert multi_project_dashboard_graph([dep])["nodes"] == ["api", "web"]
    assert detect_stale_dependencies([dep], stale_hours=24)[0]["reason"] == "schema release"
    assert orchestration_report_export([dep], [blocker])["blockers"][0]["blocker_id"] == "blocker-1"
    assert detect_shared_resource_conflicts(actions)[0]["resource"] == "db"


def test_phase_59_agent_fleet_contracts():
    member = AgentFleetMember("planner", ["planning"], cost_score=0.9, latency_score=0.8, success_rate=1.0)
    heartbeat = AgentHeartbeat("planner", "alpha", status="degraded", failure_count=1)
    route = route_agent([member], ["planning"])

    assert detect_stale_heartbeats([heartbeat])
    assert agent_cost_score([{"agent_id": "planner", "cost_usd": 1}], "planner") == 0.9
    assert agent_latency_score([{"agent_id": "planner", "latency_ms": 1000}], "planner") == 0.9
    assert agent_routing_explanation(route["selected"], route["candidates"])["reason"].startswith("highest")
    assert agent_fleet_dashboard([member])["data"]["count"] == 1
    assert agent_fleet_export([member])["agents"][0]["agent_id"] == "planner"


def test_phase_60_telemetry_observability_contracts():
    events = []
    event = create_telemetry_event("event-1", "alpha", "runtime", "error", "test", "corr-1", {"duration_ms": 6000, "token": "secret"})
    write_telemetry_event(events, event)
    rollup = metric_rollup(events, project_id="alpha", date="2026-07-16")

    assert event.payload["token"] == "<redacted>"
    assert correlate_trace(events, "corr-1")["event_count"] == 1
    assert rollup.runtime_failures == 1
    assert observability_dashboard_trends([rollup])["data"]["rollups"][0]["project_id"] == "alpha"
    assert diagnostics_bundle("alpha", events=events)["events"][0]["event_id"] == "event-1"
    assert telemetry_retention_policy()["prune_enabled"] is True
    assert prune_telemetry_events(events * 3, max_events=2)[0].event_id == "event-1"
    assert telemetry_import_export_compatibility({"schema": "hermes-os-telemetry-v1"})["compatible"] is True
    assert detect_slow_operations(events)[0]["event_id"] == "event-1"
    assert telemetry_integrity_check(events)["ok"] is True


def test_phase_61_connector_boundary_contracts(tmp_path):
    connector_file = tmp_path / "github.json"
    connector_file.write_text(json.dumps({
        "connector_id": "github",
        "name": "GitHub",
        "permissions": ["read", "write"],
        "resources": ["repo"],
        "commands": ["issues"],
        "risk_profile": "high",
    }), encoding="utf-8")
    discovered = discover_connectors([str(tmp_path)])
    manifest = ConnectorManifest("github", "GitHub", permissions=["read", "write"], resources=["repo"], commands=["issues"], risk_profile="high")

    assert discovered["connectors"][0]["connector_id"] == "github"
    assert validate_connector_permission(manifest, "write")["requires_approval"] is True
    assert connector_health_contract(manifest)["status"] == "unknown"
    assert connector_lifecycle_contract(manifest, "install")["requires_approval"] is True
    assert connector_compatibility(manifest, project_permissions=["read", "write"])["compatible"] is True


def test_phase_62_evaluation_quality_gate_contracts():
    evaluations = run_evaluations("alpha", "artifact", [{"payload": {"id": "a"}, "required_fields": ["id", "title"]}])
    gate = apply_quality_gate(evaluations)
    waived = waive_evaluation(evaluations[0], reviewer="operator", reason="accepted", expires_at="2026-12-31")

    assert gate["allowed"] is False
    assert rubric_evaluation_contract("artifact", ["clarity"])["requires_model_review"] is True
    assert self_review_workflow("artifact")["steps"] == ["summarize", "critique", "revise", "validate"]
    assert waived.status == "waived"
    assert evaluation_dashboard_panel(evaluations)["data"]["failure_count"] == 1
    assert regression_evaluation_set("planning", [{"input": "build"}])["case_count"] == 1
    assert cost_aware_evaluation_plan([{"type": "deterministic"}, {"type": "model"}])["model_checks_deferred"] is True
    assert evaluation_export([EvaluationResult("eval-ok", "alpha", "artifact", "pass")])["evaluations"][0]["status"] == "pass"


def test_phase_63_memory_intelligence_contracts():
    records = [
        MemoryIndexRecord("r1", "alpha", "decisions.md", "Use SQLite", topic="decision", confidence=0.9),
        MemoryIndexRecord("r2", "alpha", "decisions.md", "Use Postgres", topic="decision", confidence=0.8),
    ]
    lessons = extract_lessons([{"project_id": "alpha", "status": "completed", "lesson": "Dry-run first"}])
    bundle = memory_bundle_export(records)

    assert index_project_memory(records)["record_count"] == 2
    assert lessons[0].summary == "Dry-run first"
    assert summarize_project_memory(records)["traceability"][0]["source_path"] == "decisions.md"
    assert detect_memory_drift(records)["conflict_count"] == 1
    assert memory_refresh_schedule("alpha")["mode"] == "dry_run"
    assert memory_query_panel(records, topic="decision")["data"]["count"] == 2
    assert memory_bundle_import(bundle)[0].record_id == "r1"
    assert enforce_memory_source_guard(source="agent_runtime_memory", target="source_of_truth")["allowed"] is False
    assert memory_compaction_plan(records, keep_latest=1)["compact"]


def test_phase_64_release_hardening_contracts():
    checklist = release_checklist()
    notes = release_notes_from_tasks([{"phase": "Release", "status": "completed"}])

    assert checklist[0]["area"] == "cli"
    assert migration_compatibility_matrix([183, 267], current=388)["versions"][0]["supported"] is True
    assert dashboard_build_verification([{"panel_id": "runtime", "title": "Runtime"}])["ok"] is True
    assert packaging_verification(["hermes_os_integration"], ["architect"])["ok"] is True
    assert documentation_pass(["command examples"])["complete"] is True
    assert failure_drill("worker unavailable", recovery_steps=["fallback"])["status"] == "ready"
    assert "Release: 1 completed" in notes
    assert integration_suite_target()["commands"][0].endswith("tests/hermes_os_integration -q")


def test_phase_65_conversational_operating_layer_contracts(tmp_path):
    session = create_conversational_session("session-1", "operator", "alpha", goal="Build OS", initiative="COL")
    plan = chief_of_staff_plan(ChatEnvelope(message="Build a CRM", session_id=session.session_id, project_id=session.project_id), project_path=str(tmp_path))
    response = chat_response_from_plan(plan)
    transcript = transcript_from_turns(session, [{"role": "user", "content": "Build a CRM"}], ["TASKS.md"])
    audit = col_audit_event("intent.routed", session.session_id, route=plan.route.workflow)

    assert col_contracts()["ownership"]["hermes_os"]
    assert session.goal == "Build OS"
    assert response.session_id == "session-1"
    assert transcript.source_refs == ["TASKS.md"]
    assert build_col_config()["feature_flags"]["chief_of_staff"] is True
    assert any(item["name"] == "architect" for item in discover_chat_commands()["commands"])
    assert audit["type"] == "intent.routed"


def test_phase_55_to_65_completion_tracking(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / "TASKS.md").write_text(
        "\n".join(f"- `task-{number:03d}`: Task {number}" for number in range(268, 398)),
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    result = complete_phases(tmp_path, range(55, 66))
    statuses = phase_statuses(json.loads((tmp_path / ".hermes" / "tasks.json").read_text(encoding="utf-8")), range(55, 66))
    summary = completion_summary(tmp_path, range(55, 66))

    assert task_ids_for_phases([55, 65])[0] == "task-268"
    assert task_ids_for_phases([55, 65])[-1] == "task-397"
    assert result["completed"] == 130
    assert result["percent"] == 100
    assert summary["completed"] == 130
    assert all(status.percent == 100 for status in statuses)
