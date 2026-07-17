import json

from hermes_os_integration.phase_completion import complete_phases, completion_summary, phase_statuses, task_ids_for_phases
from hermes_os_integration.persistence import SQLiteRepository
from hermes_os_integration.project_runtime import AgentMessage, ProjectRuntimeManager
from hermes_os_integration.project_runtime_ops import (
    AgentFleetMember,
    LiveRuntimeExecution,
    audit_command_surface,
    build_live_runtime_plan,
    build_restore_plan,
    cancel_live_runtime,
    command_completion_metadata,
    continuous_workspace_health,
    create_telemetry_event,
    diagnostics_bundle,
    discover_template_packs,
    ensure_project_runtime_schema,
    integration_suite_target,
    live_runtime_artifact_manifest,
    live_runtime_history,
    live_runtime_rollback_plan,
    live_runtime_validation_gate,
    load_template_pack_manifest,
    module_cli_redirects,
    normalize_command_envelope,
    partial_restore_result,
    persist_project_runtime_record,
    project_runtime_integrity_check,
    redact_environment,
    resume_live_runtime,
    route_agent,
    runtime_dashboard_modules,
    runtime_log_tail_contract,
    select_sandbox_profile,
    supervise_live_runtime_process,
    template_pack_install_plan,
    template_pack_uninstall_safety,
    template_pack_update_diff,
    transition_live_runtime,
    validate_template_pack,
)


def _runtime_workspace(root):
    project = root / "projects" / "alpha"
    project.mkdir(parents=True)
    registry = root / ".hermes" / "projects" / "alpha"
    registry.mkdir(parents=True)
    registry.joinpath("project.yaml").write_text(
        f"""
name: alpha
type: control-plane
path: {project}
dashboards:
  - http://localhost:3010
agents:
  - planner
  - reviewer
infrastructure:
  production_url: https://alpha.example.com
  vector_db: alpha_vectors
runtime:
  services:
    - name: dashboard
      command: python -m http.server 3010
      cwd: {project}
      dashboard_url: http://localhost:3010
""",
        encoding="utf-8",
    )
    return project


def test_phase_46_agent_messages_and_phase_47_unified_dashboard(tmp_path):
    _runtime_workspace(tmp_path)
    manager = ProjectRuntimeManager(str(tmp_path))
    definition = manager.load_project("alpha")

    trace = manager.record_agent_message(AgentMessage(
        project_id="alpha",
        source="planner",
        target="reviewer",
        type="handoff",
        priority="high",
        message="Review runtime readiness",
        correlation_id="corr-alpha",
    ))
    dashboard = manager.unified_dashboard()

    assert definition.agents == ["planner", "reviewer"]
    assert trace.correlation_id == "corr-alpha"
    assert manager.agent_trace_view("alpha")["timeline"][0]["sender"] == "planner"
    assert dashboard["projects"][0]["project"]["name"] == "alpha"
    assert dashboard["agent_health"][0]["name"] == "planner"
    assert dashboard["infrastructure"]["alpha"]["production_url"] == "https://alpha.example.com"
    assert dashboard["projects"][0]["vector_databases"]["vector_db"] == "alpha_vectors"


def test_phase_48_and_49_command_surface_and_guarded_runtime():
    audit = audit_command_surface()
    completion = command_completion_metadata()
    envelope = normalize_command_envelope("switch", {"project_id": "alpha"})
    error = normalize_command_envelope("switch", {"message": "missing"}, ok=False, error_code="missing_project")
    plan = build_live_runtime_plan(
        project_id="alpha",
        action="write",
        command="hermes-agent --oneshot safe",
        allowlist=["hermes-agent"],
        approved=False,
        estimated_cost_usd=2.0,
        retry_count=2,
        dry_run=False,
    )

    assert {item.command for item in audit} >= {"architect", "projects", "switch", "start", "snapshot"}
    assert completion["commands"]["switch"][0] == "<project>"
    assert module_cli_redirects()["python -m hermes_os_integration.project_runtime"] == "hermes projects|switch|start|snapshot"
    assert envelope["ok"] is True
    assert error["error"]["code"] == "missing_project"
    assert plan.allowed is False
    assert plan.command_allowed is True
    assert plan.artifact_quarantine["enabled"] is True
    assert plan.rollback["required_on_failure"] is True


def test_phase_50_restore_and_phase_51_dashboard_modules():
    restore = build_restore_plan(
        {
            "project_id": "alpha",
            "open_files": ["docs/PROJECT.md"],
            "browser_urls": ["https://alpha.example.com"],
            "active_terminals": ["npm run dev"],
            "running_services": ["dashboard"],
            "current_branch": "main",
            "open_tasks": ["task-001"],
        },
        dirty_worktree=True,
        running_services=["dashboard"],
    )
    partial = partial_restore_result(restore)
    modules = runtime_dashboard_modules(
        "alpha",
        {
            "runtime": {"services": [{"name": "dashboard", "status": "running"}]},
            "infrastructure": {"production_url": "https://alpha.example.com"},
            "vector_databases": {"main": "alpha_vectors"},
        },
        snapshots=[{"id": "snapshot-1"}],
        traces=[{"correlation_id": "corr-alpha"}],
        costs=[{"project_id": "alpha", "actual_cost_usd": 0.5}],
        approvals=[{"status": "pending"}],
        template_packs=[{"pack_id": "base"}],
    )
    panel_ids = {module["panel_id"] for module in modules}

    assert any(step.kind == "editor" for step in restore.steps)
    assert any(step.kind == "browser" for step in restore.steps)
    assert partial["statuses"]["blocked"] == 1
    assert any(item["type"] == "running-service" for item in partial["conflicts"])
    assert panel_ids >= {
        "project-runtime-services",
        "workspace-snapshots",
        "snapshot-restore-preview",
        "agent-trace-timeline",
        "agent-message-detail",
        "runtime-cost-budget",
        "runtime-approval-queue",
        "infrastructure-registry",
        "vector-registry",
        "runtime-dashboard-state",
    }


def test_phase_52_persistence_and_phase_53_template_packs(tmp_path):
    repo = SQLiteRepository(str(tmp_path / "runtime.sqlite3"))
    schema = ensure_project_runtime_schema(repo)
    persist_project_runtime_record(repo, "workspace-snapshots", "alpha", "snapshot-1", {"id": "snapshot-1"})
    persist_project_runtime_record(repo, "snapshot-restore-attempts", "alpha", "restore-1", {"id": "restore-1", "snapshot_id": "snapshot-1"})
    persist_project_runtime_record(repo, "agent-traces", "alpha", "trace-1", {"id": "trace-1", "correlation_id": "corr-alpha"})
    persist_project_runtime_record(repo, "runtime-approvals", "alpha", "approval-1", {"id": "approval-1", "status": "pending"})

    pack_dir = tmp_path / "packs" / "starter"
    pack_dir.mkdir(parents=True)
    (pack_dir / "template-pack.json").write_text(json.dumps({
        "pack_id": "starter",
        "name": "Starter Pack",
        "version": "1",
        "min_hermes_os_version": "1",
        "templates": ["project.json"],
        "dependencies": ["base"],
    }), encoding="utf-8")
    pack = load_template_pack_manifest(str(pack_dir))
    discovered = discover_template_packs([str(tmp_path / "packs")])

    assert schema["status"] == "ready"
    assert project_runtime_integrity_check(repo)["ok"] is True
    assert discovered["packs"][0]["pack_id"] == "starter"
    assert validate_template_pack(pack, installed_pack_ids=["base"])["valid"] is True
    assert validate_template_pack(pack, installed_pack_ids=[])["valid"] is False
    assert template_pack_install_plan(pack)["dry_run"] is True
    assert template_pack_update_diff(pack, pack)["changed_templates"] == []
    assert template_pack_uninstall_safety(pack, in_use_templates=["project.json"])["requires_approval"] is True


def test_phase_54_continuous_workspace_operations_and_phase_55_live_runtime():
    health = continuous_workspace_health(
        [{"project_id": "alpha", "architecture_score": 70, "stale_tasks": 1, "stale_snapshot": True, "cost_budget_usd": 0.1}],
        runtime_records=[{"project_id": "alpha", "status": "failed"}],
        approvals=[{"project_id": "alpha", "status": "pending"}],
        traces=[{"project_id": "alpha", "status": "failed"}],
        cost_records=[{"project_id": "alpha", "actual_cost_usd": 0.5}],
    )
    execution = LiveRuntimeExecution("exec-1", "alpha", "hermes-agent --oneshot safe")
    running = transition_live_runtime(execution, "running", pid=123)
    supervised = supervise_live_runtime_process(running, exit_code=0, stdout_ref="stdout.log", duration_ms=42)
    completed = transition_live_runtime(supervised, "completed")
    failed = supervise_live_runtime_process(running, exit_code=1, stderr_ref="stderr.log")
    canceled = cancel_live_runtime(running, requester="operator", reason="manual stop")
    resumed = resume_live_runtime(failed, context_ref="hermes-os://context/exec-1")
    manifest = live_runtime_artifact_manifest("alpha", [{"ref": "artifact.md", "checksum": "sha256:1", "validation_status": "passed"}])
    gate = live_runtime_validation_gate(manifest)
    rollback = live_runtime_rollback_plan(failed, [{"type": "file-write", "target": "artifact.md", "rollback_command": "git checkout -- artifact.md"}])
    sandbox = select_sandbox_profile({"sandbox_profiles": {"default": {"name": "default", "writes": False}, "write": {"name": "writer", "writes": True}}}, "write")
    event = create_telemetry_event("event-1", "alpha", "runtime", "warning", "test", "corr-alpha", {"token": "secret"})
    routed = route_agent([AgentFleetMember("planner", ["planning"], success_rate=1.0)], ["planning"])

    assert health["reports"][0]["blockers"] >= ["approval-aging", "architecture-drift"]
    assert health["activity_feed"][0]["type"] == "workspace-health"
    assert supervised.state == "validating"
    assert completed.state == "completed"
    assert failed.state == "failed"
    assert canceled["audit"]["type"] == "live-runtime-cancellation"
    assert resumed.resumed_from == "exec-1"
    assert runtime_log_tail_contract("exec-1")["path"].endswith("exec-1.stdout.log")
    assert gate["allowed"] is True
    assert rollback["required"] is True
    assert sandbox["requires_approval"] is True
    assert redact_environment({"SAFE": "1", "TOKEN": "secret"}, ["SAFE"])["TOKEN"] == "<redacted>"
    assert live_runtime_history([completed], project_id="alpha")[0]["execution_id"] == "exec-1"
    assert diagnostics_bundle("alpha", events=[event], executions=[completed])["events"][0]["payload"]["token"] == "<redacted>"
    assert routed["selected"]["agent_id"] == "planner"
    assert integration_suite_target()["name"] == "hermes-os-integration"


def test_phase_45_to_55_completion_tracking(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / "TASKS.md").write_text(
        "\n".join(f"- `task-{number:03d}`: Task {number}" for number in range(168, 280)),
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    result = complete_phases(tmp_path, range(45, 56))
    statuses = phase_statuses(json.loads((tmp_path / ".hermes" / "tasks.json").read_text(encoding="utf-8")), range(45, 56))
    summary = completion_summary(tmp_path, range(45, 56))

    assert task_ids_for_phases([45, 55])[0] == "task-168"
    assert task_ids_for_phases([45, 55])[-1] == "task-279"
    assert result["completed"] == 112
    assert result["percent"] == 100
    assert summary["completed"] == 112
    assert all(status.percent == 100 for status in statuses)
