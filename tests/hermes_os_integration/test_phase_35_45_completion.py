import json

from hermes_os_integration.phase_completion import (
    complete_phases,
    completion_summary,
    phase_statuses,
    task_ids_for_phases,
)
from hermes_os_integration.project_runtime import ProjectRuntimeManager, WorkspaceSnapshot, main as runtime_main
from hermes_os_integration.project_runtime_ops import build_restore_plan, partial_restore_result, runtime_dashboard_modules


def _write_project(root, project_id="sample"):
    project_path = root / "projects" / project_id
    project_path.mkdir(parents=True)
    registry_path = root / ".hermes" / "projects" / project_id
    registry_path.mkdir(parents=True)
    (registry_path / "project.yaml").write_text(
        "\n".join([
            "name: sample",
            "type: app",
            f"path: {project_path}",
            "dashboards:",
            "  - http://localhost:9000/dashboard",
            "documents:",
            "  - docs/PROJECT.md",
            "agents:",
            "  - planner",
            "infrastructure:",
            "  production_url: https://sample.example.com",
            "runtime:",
            "  services:",
            "    - name: web",
            "      command: python -m http.server 9000",
            f"      cwd: {project_path}",
            "      dashboard_url: http://localhost:9000/dashboard",
            "      health_check: http://localhost:9000/health",
        ]),
        encoding="utf-8",
    )
    (project_path / ".hermes").mkdir()
    (project_path / ".hermes" / "tasks.json").write_text(
        json.dumps({"tasks": [{"id": "task-001", "status": "completed"}, {"id": "task-002", "status": "planned"}]}),
        encoding="utf-8",
    )
    return project_path


def test_project_runtime_mvp_switch_status_and_memory(tmp_path):
    _write_project(tmp_path)
    manager = ProjectRuntimeManager(str(tmp_path))

    projects = manager.list_projects()
    switched = manager.switch_project("sample")
    status = manager.project_status(projects[0])

    assert projects[0].runtime_services[0].name == "web"
    assert switched["dry_run"] is True
    assert "project_memory_loaded" in switched["steps"]
    assert status["tasks"] == {"open": 1, "total": 2}
    assert all((tmp_path / "projects" / "sample" / "memory" / name).exists() for name in ["architecture.md", "agents.md"])
    assert status["dashboards"] == ["http://localhost:9000/dashboard"]
    assert status["infrastructure"]["production_url"] == "https://sample.example.com"


def test_project_runtime_cli_projects_switch_start_snapshot(tmp_path, capsys):
    _write_project(tmp_path)

    assert runtime_main(["--workspace-root", str(tmp_path), "projects"]) == 0
    assert "sample" in capsys.readouterr().out

    assert runtime_main(["--workspace-root", str(tmp_path), "switch", "sample"]) == 0
    assert "workspace_restore_planned" in capsys.readouterr().out

    assert runtime_main(["--workspace-root", str(tmp_path), "start", "sample"]) == 0
    start_payload = json.loads(capsys.readouterr().out)
    assert start_payload["services"][0]["status"] == "planned"

    assert runtime_main(["--workspace-root", str(tmp_path), "snapshot", "save", "sample"]) == 0
    assert "snapshot_path" in capsys.readouterr().out
    assert runtime_main(["--workspace-root", str(tmp_path), "snapshot", "restore", "sample"]) == 0
    restore_payload = json.loads(capsys.readouterr().out)
    assert restore_payload["restore_contracts"]["browser_urls"] == ["http://localhost:9000/dashboard"]
    assert restore_payload["restore_contracts"]["services"] == ["web"]


def test_workspace_snapshot_restore_contracts_and_dashboard_modules(tmp_path):
    _write_project(tmp_path)
    manager = ProjectRuntimeManager(str(tmp_path))
    snapshot = WorkspaceSnapshot(
        project_id="sample",
        open_files=["docs/PROJECT.md"],
        browser_urls=["https://sample.example.com/dashboard"],
        active_terminals=["npm run dev"],
        running_services=["web"],
        current_branch="main",
        open_tasks=["task-002"],
    )
    path = manager.save_snapshot("sample", snapshot)
    restored = manager.restore_snapshot("sample")
    plan = build_restore_plan(restored["snapshot"], dirty_worktree=True, running_services=["web"])
    partial = partial_restore_result(plan)
    modules = runtime_dashboard_modules("sample", manager.project_status(manager.load_project("sample")), snapshots=[restored["snapshot"]])

    assert path.endswith(".json")
    assert restored["dry_run"] is True
    assert any(step.kind == "editor" and step.command[0] == "code" for step in plan.steps)
    assert any(step.kind == "browser" and step.command[0] == "open" for step in plan.steps)
    assert any(step.kind == "service" and step.status == "skipped" for step in plan.steps)
    assert any(conflict["type"] == "dirty-worktree" for conflict in partial["conflicts"])
    assert {module["panel_id"] for module in modules} >= {"project-runtime-services", "workspace-snapshots", "runtime-cost-budget"}


def test_phase_completion_marks_tasks_114_to_173(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / "TASKS.md").write_text(
        "\n".join(f"- `task-{number:03d}`: Task {number}" for number in range(114, 174)),
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "tasks.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")

    result = complete_phases(tmp_path)
    statuses = phase_statuses(json.loads((tmp_path / ".hermes" / "tasks.json").read_text(encoding="utf-8")))
    summary = completion_summary(tmp_path)

    assert task_ids_for_phases([35, 45]) == ["task-114", "task-115", "task-116", "task-117", "task-118", "task-168", "task-169", "task-170", "task-171", "task-172", "task-173"]
    assert result["percent"] == 100
    assert summary["completed"] == 60
    assert all(status.percent == 100 for status in statuses)
    assert (tmp_path / ".hermes" / "phase-35-45-completion.json").exists()
