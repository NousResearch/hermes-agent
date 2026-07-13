import json
import subprocess
import sys

from hermes_os_integration.architect_cli import main as architect_main
from hermes_os_integration.dashboard import build_project_dashboard
from hermes_os_integration.delegation import DelegationEngine
from hermes_os_integration.execution import build_dry_run_execution_report, ingest_artifacts
from hermes_os_integration.persistence import SQLiteRepository, persist_decision
from hermes_os_integration.plan_cli import main as plan_main
from hermes_os_integration.project_runtime import (
    AgentMessage,
    ProjectRuntimeManager,
    WorkspaceSnapshot,
    main as project_runtime_main,
)
from hermes_os_integration.project_runtime_ops import (
    AgentFleetMember,
    ApprovalRequest,
    ConnectorManifest,
    CrossProjectDependency,
    LiveRuntimeExecution,
    MemoryIndexRecord,
    audit_command_surface,
    build_live_runtime_plan,
    build_automation_workflow,
    build_restore_plan,
    command_completion_metadata,
    continuous_workspace_health,
    automation_dry_run_diff,
    automation_failure_report,
    automation_preflight,
    approval_audit_export,
    approval_risk_score,
    apply_quality_gate,
    connector_dry_run,
    connector_secret_policy,
    cost_aware_evaluation_plan,
    create_approval_request,
    create_telemetry_event,
    decide_approval,
    detect_memory_drift,
    detect_shared_resource_conflicts,
    diagnostics_bundle,
    discover_template_packs,
    expire_approvals,
    failure_drill,
    ensure_project_runtime_schema,
    index_project_memory,
    integration_suite_target,
    live_runtime_artifact_manifest,
    live_runtime_history,
    load_template_pack_manifest,
    memory_compaction_plan,
    migration_compatibility_matrix,
    module_cli_redirects,
    normalize_command_envelope,
    normalize_connector_output,
    partial_restore_result,
    persist_project_runtime_record,
    project_runtime_integrity_check,
    quarantine_agent,
    redact_environment,
    release_checklist,
    release_notes_from_tasks,
    resolve_cross_project_dependencies,
    route_agent,
    run_evaluations,
    runtime_dashboard_modules,
    score_agent,
    template_pack_install_plan,
    template_pack_uninstall_safety,
    template_pack_update_diff,
    telemetry_rollup,
    transition_live_runtime,
    validate_connector_permission,
    validate_template_pack,
    waive_evaluation,
)
from hermes_os_integration.review_loops import ScheduledReview, preview_scheduled_review, run_scheduled_review_job, scheduled_review_cron_payload
from hermes_os_integration.runtime_policies import RuntimePolicy, aggregate_cost_budget, approval_prompt_for_decision, evaluate_runtime_policy, retry_backoff_seconds
from hermes_os_integration.tasks import generate_tasks_from_review, next_task_number_from_text, write_task_artifacts
from hermes_os_integration.templates import discover_templates, template_compatible, template_registry_paths, TemplateLoader
from hermes_os_integration.work_graph import compile_work_graph


def test_architect_cli_persists_to_sqlite(tmp_path, capsys):
    project = tmp_path / "sample"
    project.mkdir()
    db = tmp_path / "records.sqlite3"

    code = architect_main([
        "review",
        "sample",
        "--projects-root",
        str(tmp_path),
        "--json",
        "--persist",
        "--db",
        str(db),
    ])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["project_id"] == "sample"
    assert SQLiteRepository(str(db)).get("review-reports", "sample")["project_id"] == "sample"


def test_plan_cli_loads_external_template_and_persists(tmp_path, capsys):
    project = tmp_path / "sample"
    project.mkdir()
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({
            "template_id": "launch",
            "name": "Launch",
            "nodes": [{"id": "design", "type": "task", "title": "Design"}],
        }),
        encoding="utf-8",
    )
    db = tmp_path / "records.sqlite3"

    code = plan_main([
        "sample",
        "--projects-root",
        str(tmp_path),
        "--template",
        str(template),
        "--json",
        "--persist",
        "--db",
        str(db),
    ])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["nodes"][0]["id"] == "design"
    assert SQLiteRepository(str(db)).get("work-graphs", "sample")["project_id"] == "sample"


def test_template_loader_reads_directory(tmp_path):
    (tmp_path / "one.yaml").write_text(
        "template_id: one\nname: One\nnodes:\n  - id: n1\n    type: task\n    title: N1\n",
        encoding="utf-8",
    )
    loaded, errors = TemplateLoader().load_path(str(tmp_path))

    assert errors == []
    assert loaded[0].template_id == "one"


def test_sqlite_repository_save_get_list_latest(tmp_path):
    repo = SQLiteRepository(str(tmp_path))
    persist_decision(repo, "decision-1", {"decision": "go"})
    persist_decision(repo, "decision-2", {"decision": "hold"})

    assert repo.get("decisions", "decision-1")["decision"] == "go"
    assert [item["decision"] for item in repo.list("decisions")] == ["go", "hold"]
    assert repo.latest("decisions")["decision"] == "hold"


def test_runtime_policy_blocks_cost_retry_and_unapproved_write():
    decision = evaluate_runtime_policy(
        action="write",
        estimated_cost_usd=5,
        retry_count=2,
        policy=RuntimePolicy(max_cost_usd=1, max_retries=1),
    )

    assert decision.allowed is False
    assert decision.approval_required is True
    assert decision.retry_allowed is False
    assert "estimated cost exceeds policy" in decision.reasons
    assert decision.audit["type"] == "runtime_policy_decision"


def test_scheduled_review_job_persists_score_history(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    repo = SQLiteRepository(str(tmp_path))

    result = run_scheduled_review_job(
        ScheduledReview(schedule="0 9 * * 1", project_scope=["sample"]),
        [{"project_id": "sample", "project_path": str(project), "present_documents": []}],
        repository=repo,
    )

    assert result["project_count"] == 1
    assert repo.get("review-reports", "sample")["project_id"] == "sample"
    assert repo.latest("score-history")["project_id"] == "sample"


def test_dashboard_summary_builds_native_panels(tmp_path):
    project = tmp_path / "sample"
    docs = project / "docs"
    docs.mkdir(parents=True)
    (docs / "PROJECT.md").write_text("# Project\n", encoding="utf-8")

    summary = build_project_dashboard("sample", str(tmp_path), launcher_path=str(tmp_path / "missing"))
    panel_ids = {panel["panel_id"] for panel in summary["panels"]}

    assert summary["project_id"] == "sample"
    assert "architecture-score" in panel_ids
    assert "runtime-delegation" in panel_ids
    assert "task-backlog" in panel_ids
    assert "dry-run-execution" in panel_ids
    assert "agent-trace-timeline" in panel_ids
    assert "runtime-approval-queue" in panel_ids


def test_installed_console_entrypoints_accept_native_commands(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    hermes = __import__("pathlib").Path(__file__).resolve().parents[2] / ".venv" / "bin" / "hermes"

    architect = subprocess.run(
        [str(hermes), "architect", "review", "sample", "--projects-root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    plan = subprocess.run(
        [str(hermes), "plan", "sample", "--projects-root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert architect.returncode in {2, 3}
    assert json.loads(architect.stdout)["project_id"] == "sample"
    assert plan.returncode == 2
    assert json.loads(plan.stdout)["project_id"] == "sample"


def test_review_to_task_artifacts(tmp_path):
    from hermes_os_integration.architecture_first import ArchitectureReviewRequest, review_architecture

    report = review_architecture(ArchitectureReviewRequest(
        project_id="sample",
        project_path=str(tmp_path),
        present_documents=[],
        completed_stages=[],
    ))
    tasks = generate_tasks_from_review(report, start_at=114)
    result = write_task_artifacts(str(tmp_path), tasks)

    assert tasks[0].id == "task-114"
    assert next_task_number_from_text("x task-153") == 154
    assert result["status"] == "written"
    assert (tmp_path / "TASKS.md").exists()
    assert (tmp_path / ".hermes" / "tasks.json").exists()


def test_plan_cli_generates_tasks(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    code = plan_main(["sample", "--projects-root", str(tmp_path), "--generate-tasks"])

    assert code == 2
    assert (project / "TASKS.md").exists()


def test_sqlite_migrations_import_export_and_integrity(tmp_path):
    from hermes_os_integration.persistence import LocalRepository

    local = LocalRepository(str(tmp_path / "project"))
    persist_decision(local, "decision-1", {"id": "decision-1", "decision": "go"})
    repo = SQLiteRepository(str(tmp_path / "records.sqlite3"))

    assert repo.schema_version() == 1
    assert repo.migrate(1)["to"] == 1
    assert repo.import_from_local(local)["imported"] == 1
    bundle = repo.export_bundle(str(tmp_path / "bundle.json"))
    assert json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))["schema_version"] == 1
    assert bundle.endswith("bundle.json")
    assert repo.integrity_check()["ok"] is True


def test_scheduled_review_preview_cron_payload_and_delta(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    repo = SQLiteRepository(str(tmp_path))
    scan = {"project_id": "sample", "project_path": str(project), "present_documents": []}
    scheduled = ScheduledReview(schedule="0 9 * * 1", project_scope=["sample"], mode="proposal")

    preview = preview_scheduled_review(scheduled, [scan])
    first = run_scheduled_review_job(scheduled, [scan], repository=repo)
    second = run_scheduled_review_job(scheduled, [scan], repository=repo)
    payload = scheduled_review_cron_payload("sample", str(tmp_path))

    assert preview["requires_approval"] is True
    assert first["outputs"][0]["score_delta"] is None
    assert second["outputs"][0]["score_delta"] == 0
    assert payload["name"] == "Hermes OS review: sample"


def test_runtime_policy_helpers_and_delegation_block():
    audits = []
    engine = DelegationEngine(persist_audit=audits.append)
    result = engine.delegate(type("Req", (), {
        "task_id": "task-1",
        "project_id": "sample",
        "task_type": "coding",
        "prompt": "write code",
        "working_directory": "/tmp",
        "dry_run": False,
        "opt_in_runtime": True,
    })())
    decision = evaluate_runtime_policy(action="write")

    assert result.response.status == "blocked"
    assert audits[0]["type"] == "runtime_policy_decision"
    assert retry_backoff_seconds(99) == 5
    assert aggregate_cost_budget([{"project_id": "sample", "actual_cost_usd": 1.5}], project_id="sample")["actual_cost_usd"] == 1.5
    assert approval_prompt_for_decision(decision)["required"] is True


def test_template_registry_discovery_and_compatibility(tmp_path):
    template_dir = tmp_path / ".hermes" / "templates"
    template_dir.mkdir(parents=True)
    (template_dir / "one.json").write_text(json.dumps({
        "template_id": "one",
        "name": "One",
        "version": "1",
        "nodes": [{"id": "n1", "type": "task", "title": "N1"}],
    }), encoding="utf-8")

    paths = template_registry_paths(str(tmp_path))
    templates, diagnostics = discover_templates(paths)

    assert diagnostics == []
    assert templates[0].source_path.endswith("one.json")
    assert template_compatible(templates[0]) is True


def test_execution_dry_run_and_artifact_ingestion(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    graph = compile_work_graph("sample", str(tmp_path))
    report = build_dry_run_execution_report(graph)
    updated = ingest_artifacts(graph, graph.nodes[0].id, ["file://artifact.md"], repository=SQLiteRepository(str(tmp_path)))

    assert report.project_id == "sample"
    assert updated.execution_results[-1].artifacts == ["file://artifact.md"]
    assert updated.validation_results[0].status == "passed"


def test_project_runtime_registry_memory_status_switch_snapshot_agent_trace_and_dashboard(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    project = workspace / "khashi"
    project.mkdir(parents=True)
    registry = workspace / ".hermes" / "projects" / "khashi"
    registry.mkdir(parents=True)
    registry.joinpath("project.yaml").write_text(
        """
name: khashi
type: research
path: {project_path}
repository:
  provider: github
startup:
  - npm run dashboard
dashboards:
  - http://localhost:3000
documents:
  - docs/architecture.md
agents:
  - amari
  - observer
infrastructure:
  database: khashi_db
  vector_db: khashi_vectors
runtime:
  services:
    - dashboard
status:
  enabled: true
""".format(project_path=str(project)),
        encoding="utf-8",
    )
    (project / ".hermes").mkdir()
    (project / ".hermes" / "tasks.json").write_text(
        json.dumps({"tasks": [{"id": "task-1", "status": "open"}, {"id": "task-2", "status": "completed"}]}),
        encoding="utf-8",
    )
    manager = ProjectRuntimeManager(str(workspace))

    definition = manager.load_project("khashi")
    assert manager.validate_project(definition).ok is True
    assert definition.infrastructure["database"] == "khashi_db"
    assert definition.vector_databases["vector_db"] == "khashi_vectors"

    switch = manager.switch_project("khashi")
    assert switch["dry_run"] is True
    assert switch["status"]["tasks"] == {"open": 1, "total": 2}
    assert (project / "memory" / "architecture.md").exists()
    assert manager.load_memory(definition)["architecture.md"].startswith("# Architecture")

    start = manager.start_project("khashi")
    assert start["services"][0]["status"] == "planned"

    snapshot_path = manager.save_snapshot(
        "khashi",
        WorkspaceSnapshot(
            project_id="khashi",
            open_files=["docs/architecture.md"],
            browser_urls=["http://localhost:3000"],
            running_services=["dashboard"],
            current_branch="main",
            open_tasks=["task-1"],
        ),
    )
    restored = manager.restore_snapshot("khashi")
    assert snapshot_path.endswith(".json")
    assert restored["restore_contracts"]["browser_urls"] == ["http://localhost:3000"]
    assert restored["restore_contracts"]["services"] == ["dashboard"]

    trace = manager.record_agent_message(AgentMessage(
        project_id="khashi",
        source="amari",
        target="observer",
        type="observation",
        priority="medium",
        message="Liquidity imbalance detected",
        correlation_id="corr-1",
    ))
    assert trace.sender == "amari"
    assert manager.agent_trace_view("khashi")["timeline"][0]["receiver"] == "observer"

    dashboard = manager.unified_dashboard()
    assert dashboard["projects"][0]["project"]["name"] == "khashi"
    assert dashboard["agent_health"][0]["name"] == "amari"
    assert dashboard["infrastructure"]["khashi"]["database"] == "khashi_db"

    code = project_runtime_main(["--workspace-root", str(workspace), "switch", "khashi"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["project_id"] == "khashi"


def test_project_runtime_extension_phase_contracts(tmp_path):
    audits = audit_command_surface()
    assert {audit.command for audit in audits} >= {"architect", "workspace", "snapshot"}
    assert command_completion_metadata()["commands"]["snapshot"][0] == "save"
    assert module_cli_redirects()["python -m hermes_os_integration.workspace_control"] == "hermes workspace"
    assert normalize_command_envelope("projects", {"ok": True})["ok"] is True

    live_plan = build_live_runtime_plan(
        project_id="sample",
        action="write",
        command="hermes-agent --oneshot ok",
        allowlist=["hermes-agent"],
        approved=False,
        estimated_cost_usd=2,
        retry_count=2,
        dry_run=False,
    )
    assert live_plan.allowed is False
    assert live_plan.command_allowed is True
    assert live_plan.approval["required"] is True
    assert live_plan.artifact_quarantine["ingest_after_validation"] is True
    assert live_plan.rollback["required_on_failure"] is True

    restore = build_restore_plan(
        {
            "project_id": "sample",
            "open_files": ["docs/architecture.md"],
            "browser_urls": ["http://localhost:3000"],
            "active_terminals": ["npm run dev"],
            "running_services": ["dashboard"],
            "current_branch": "main",
            "open_tasks": ["task-1"],
        },
        dirty_worktree=True,
        running_services=["dashboard"],
    )
    restore_result = partial_restore_result(restore)
    assert restore_result["statuses"]["blocked"] == 1
    assert any(item["type"] == "dirty-worktree" for item in restore_result["conflicts"])

    modules = runtime_dashboard_modules(
        "sample",
        {
            "runtime": {"services": [{"name": "dashboard", "status": "running"}]},
            "infrastructure": {"db": "local"},
            "vector_databases": {"main": "qdrant"},
        },
        snapshots=[{"id": "snap-1"}],
        traces=[{"correlation_id": "corr-1"}],
        costs=[{"project_id": "sample", "actual_cost_usd": 0.25}],
        approvals=[{"status": "pending"}],
        template_packs=[{"pack_id": "base"}],
    )
    panel_ids = {module["panel_id"] for module in modules}
    assert "project-runtime-services" in panel_ids
    assert "runtime-cost-budget" in panel_ids
    assert "template-packs" in panel_ids

    repo = SQLiteRepository(str(tmp_path / "runtime.sqlite3"))
    schema = ensure_project_runtime_schema(repo)
    persist_project_runtime_record(repo, "workspace-snapshots", "sample", "snap-1", {"id": "snap-1"})
    persist_project_runtime_record(repo, "agent-traces", "sample", "trace-1", {"id": "trace-1", "correlation_id": "corr-1"})
    assert schema["status"] == "ready"
    assert project_runtime_integrity_check(repo)["ok"] is True

    pack_dir = tmp_path / "packs" / "base"
    pack_dir.mkdir(parents=True)
    (pack_dir / "template-pack.json").write_text(
        json.dumps({
            "pack_id": "base",
            "name": "Base",
            "version": "1",
            "templates": ["base-project.json"],
            "dependencies": ["foundation"],
        }),
        encoding="utf-8",
    )
    pack = load_template_pack_manifest(str(pack_dir))
    discovered = discover_template_packs([str(tmp_path / "packs")])
    assert discovered["packs"][0]["pack_id"] == "base"
    assert validate_template_pack(pack, installed_pack_ids=["foundation"])["valid"] is True
    assert validate_template_pack(pack, installed_pack_ids=[])["valid"] is False
    assert template_pack_install_plan(pack)["dry_run"] is True
    assert template_pack_update_diff(pack, pack)["changed_templates"] == []
    assert template_pack_uninstall_safety(pack, in_use_templates=["base-project.json"])["allowed"] is False

    health = continuous_workspace_health(
        [{"project_id": "sample", "architecture_score": 70, "stale_tasks": 1, "cost_budget_usd": 0.1}],
        runtime_records=[{"project_id": "sample", "status": "failed"}],
        approvals=[{"project_id": "sample", "status": "pending"}],
        traces=[{"project_id": "sample", "status": "failed"}],
        cost_records=[{"project_id": "sample", "actual_cost_usd": 0.25}],
    )
    report = health["reports"][0]
    assert health["dry_run"] is True
    assert "architecture-drift" in report["blockers"]
    assert "cost-budget-drift" in report["blockers"]


def test_project_runtime_production_governance_fleet_observability_memory_and_release_contracts():
    execution = LiveRuntimeExecution(
        execution_id="exec-1",
        project_id="sample",
        command="hermes-agent --oneshot ok",
    )
    running = transition_live_runtime(execution, "running", pid=123)
    validating = transition_live_runtime(running, "validating", exit_code=0)
    completed = transition_live_runtime(validating, "completed")
    manifest = live_runtime_artifact_manifest("sample", [{"ref": "artifact.md", "checksum": "sha256:1", "validation_status": "passed"}])
    assert completed.state == "completed"
    assert manifest["valid"] is True
    assert live_runtime_history([completed], project_id="sample")[0]["execution_id"] == "exec-1"
    assert redact_environment({"SAFE": "1", "TOKEN": "secret"}, ["SAFE"])["TOKEN"] == "<redacted>"

    approval = create_approval_request(
        "approval-1",
        "sample",
        requester="amari",
        scope="runtime",
        risk="high",
        action="deploy",
        expires_at="2026-01-01T00:00:00Z",
    )
    decided = decide_approval(approval, reviewer="human", approved=True, reason="safe no-op")
    expired = expire_approvals([approval], now="2027-01-01T00:00:00Z")[0]
    assert decided.status == "approved"
    assert expired.status == "expired"
    assert approval_risk_score(action="deploy", risk="high", estimated_cost_usd=2, resources=["db"]) >= 90
    assert approval_audit_export([approval])["count"] == 1

    workflow = build_automation_workflow("sample", ["switch", "restore", "start"])
    preflight = automation_preflight(dirty_worktree=True, unavailable_tools=["code"])
    diff = automation_dry_run_diff({"branch": "main"}, {"branch": "feature"})
    failure = automation_failure_report(workflow, "step-002", "restore conflict")
    assert workflow.steps[1]["depends_on"] == ["step-001"]
    assert preflight["ok"] is False
    assert diff["change_count"] == 1
    assert "resolve preflight issues" in failure["next_actions"]

    dependency = CrossProjectDependency("api", "web", "schema release")
    orchestration = resolve_cross_project_dependencies([dependency])
    conflicts = detect_shared_resource_conflicts([
        {"project_id": "api", "resource": "database"},
        {"project_id": "web", "resource": "database"},
    ])
    assert orchestration["blocked_count"] == 1
    assert conflicts[0]["resource"] == "database"

    agent = AgentFleetMember("planner", ["planning", "review"], success_rate=0.9)
    routed = route_agent([agent], ["planning"])
    quarantined = quarantine_agent(agent, failures=4)
    assert score_agent(agent, required_capabilities=["planning"])["health_score"] > 0
    assert routed["selected"]["agent_id"] == "planner"
    assert quarantined.quarantined is True

    event = create_telemetry_event("event-1", "sample", "runtime", "warning", "test", "corr-1", {"prompt": "secret", "ok": True})
    rollup = telemetry_rollup([event])
    bundle = diagnostics_bundle("sample", events=[event], executions=[completed], approvals=[approval])
    assert event.payload["prompt"] == "<redacted>"
    assert rollup["by_project"]["sample"] == 1
    assert bundle["events"][0]["event_id"] == "event-1"

    connector = ConnectorManifest("github", "GitHub", permissions=["read", "write"], risk_profile="high")
    permission = validate_connector_permission(connector, "write")
    dry_run = connector_dry_run(connector, "write", "repo")
    normalized = normalize_connector_output(connector, {"issue": 1})
    assert permission["requires_approval"] is True
    assert dry_run["dry_run"] is True
    assert normalized["connector_id"] == "github"
    assert connector_secret_policy(["env:GITHUB_TOKEN"])["raw_secret_storage_allowed"] is False

    evaluations = run_evaluations("sample", "workgraph", [{"payload": {"id": "node"}, "required_fields": ["id", "title"]}])
    gate = apply_quality_gate(evaluations)
    waived = waive_evaluation(evaluations[0], reviewer="human", reason="accepted for dry-run", expires_at="2026-12-31")
    plan = cost_aware_evaluation_plan([{"type": "deterministic"}, {"type": "model"}])
    assert gate["allowed"] is False
    assert waived.status == "waived"
    assert plan["model_checks_deferred"] is True

    memory = [
        MemoryIndexRecord("m1", "sample", "decisions.md", "Use SQLite", topic="db", confidence=0.9, timestamp="2026-01-02"),
        MemoryIndexRecord("m2", "sample", "architecture.md", "Use Postgres", topic="db", confidence=0.8, timestamp="2026-01-01"),
    ]
    indexed = index_project_memory(memory)
    drift = detect_memory_drift(memory)
    compact = memory_compaction_plan(memory, keep_latest=1)
    assert indexed["record_count"] == 2
    assert drift["conflict_count"] == 1
    assert compact["compact"] == ["m2"]

    checklist = release_checklist()
    matrix = migration_compatibility_matrix([183, 267, 387], 387)
    drill = failure_drill("unavailable runtime worker", recovery_steps=["fall back to dry-run"])
    notes = release_notes_from_tasks([{"phase": "Release Hardening", "status": "completed"}])
    suite = integration_suite_target()
    assert checklist[0]["area"] == "cli"
    assert matrix["versions"][0]["supported"] is True
    assert drill["status"] == "ready"
    assert "Release Hardening" in notes
    assert suite["name"] == "hermes-os-integration"
