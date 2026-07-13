"""Architecture dashboard panel data contracts."""

from dataclasses import dataclass, field
from typing import Dict, List

from .architecture_first import ArchitectureReviewReport
from .contracts import RuntimeStatus
from .health import check_runtime_health
from .scanners import scan_project
from .architecture_first import ArchitectureReviewRequest, review_architecture
from .work_graph import compile_work_graph, resolve_dependencies
from .execution import build_dry_run_execution_report
from .tasks import generate_tasks_from_review, generate_tasks_from_work_graph, task_summary
from .templates import discover_templates, template_registry_paths
from .project_runtime_ops import runtime_dashboard_modules
from .conversational import col_dashboard_panels


@dataclass(frozen=True)
class DashboardPanel:
    panel_id: str
    title: str
    data: Dict[str, object] = field(default_factory=dict)


def architecture_score_panel(report: ArchitectureReviewReport):
    return DashboardPanel(
        panel_id="architecture-score",
        title="Architecture Score",
        data={
            "project_id": report.project_id,
            "score": report.architecture_score,
            "blocked": report.blocked,
            "critical_gaps": report.critical_gaps,
        },
    )


def gap_panel(report: ArchitectureReviewReport):
    return DashboardPanel(
        panel_id="architecture-gaps",
        title="Architecture Gaps",
        data={
            "missing_documents": report.missing_documents,
            "missing_schemas": report.missing_schemas,
            "missing_dashboards": report.missing_dashboards,
            "missing_approvals": report.missing_approvals,
            "roadmap": report.priority_roadmap,
        },
    )


def approvals_panel(approvals: List[Dict[str, object]], blocked_executions: List[Dict[str, object]]):
    return DashboardPanel(
        panel_id="approvals-and-blocks",
        title="Approvals And Blocks",
        data={
            "approvals": approvals,
            "blocked_executions": blocked_executions,
            "pending_count": len([item for item in approvals if item.get("status") == "pending"]),
        },
    )


def runtime_delegation_panel(status: RuntimeStatus, recent_errors=None, dry_run: bool = True):
    return DashboardPanel(
        panel_id="runtime-delegation",
        title="Runtime Delegation",
        data={
            "available": status.available,
            "provider": status.provider,
            "version": status.version,
            "latency_ms": status.latency_ms,
            "mode": "dry_run" if dry_run else "live",
            "recent_errors": recent_errors or status.recent_errors,
        },
    )


def work_graph_summary_panel(graph):
    blocked = [node for node in graph.nodes if node.status == "blocked"]
    return DashboardPanel(
        panel_id="work-graph-summary",
        title="Work Graph",
        data={
            "project_id": graph.project_id,
            "node_count": len(graph.nodes),
            "dependency_count": len(graph.dependencies),
            "blocked_count": len(blocked),
            "assignment_count": len(graph.assignments),
            "approval_count": len([node for node in graph.nodes if node.type == "approval"]),
        },
    )


def dependency_block_panel(resolution):
    return DashboardPanel(
        panel_id="dependency-blocks",
        title="Dependencies And Blocked Work",
        data={
            "ordered": resolution.get("ordered", []),
            "cycles": resolution.get("cycles", []),
            "blocked": resolution.get("blocked", []),
        },
    )


def execution_validation_panel(graph):
    statuses = {}
    for node in graph.nodes:
        statuses[node.status] = statuses.get(node.status, 0) + 1
    validation_statuses = {}
    for result in graph.validation_results:
        validation_statuses[result.status] = validation_statuses.get(result.status, 0) + 1
    return DashboardPanel(
        panel_id="execution-validation",
        title="Execution And Validation",
        data={
            "node_statuses": statuses,
            "validation_statuses": validation_statuses,
            "execution_results": len(graph.execution_results),
        },
    )


def agent_assignment_panel(graph, runtime_usage=None):
    by_agent = {}
    for assignment in graph.assignments:
        by_agent[assignment.agent_kind] = by_agent.get(assignment.agent_kind, 0) + 1
    return DashboardPanel(
        panel_id="agent-assignments",
        title="Agent Assignments",
        data={
            "assignments_by_agent": by_agent,
            "fallback_count": len([assignment for assignment in graph.assignments if assignment.fallback]),
            "runtime_usage": runtime_usage or [],
        },
    )


def task_backlog_panel(tasks):
    return DashboardPanel(
        panel_id="task-backlog",
        title="Task Backlog",
        data={
            **task_summary(tasks),
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "phase": task.phase,
                    "status": task.status,
                    "risk": task.risk,
                    "dependencies": task.dependencies,
                }
                for task in tasks
            ],
        },
    )


def template_panel(templates, diagnostics):
    return DashboardPanel(
        panel_id="templates",
        title="Templates",
        data={
            "template_count": len(templates),
            "compile_failure_count": len(diagnostics),
            "templates": [
                {"template_id": template.template_id, "name": template.name, "version": template.version}
                for template in templates
            ],
            "diagnostics": diagnostics,
        },
    )


def dry_run_execution_panel(report):
    return DashboardPanel(
        panel_id="dry-run-execution",
        title="Dry-run Execution",
        data={
            "batch_count": len(report.batches),
            "expected_artifacts": report.expected_artifacts,
            "policy_count": len(report.policies),
        },
    )


def build_project_dashboard(project: str, projects_root: str | None = None, launcher_path: str = ""):
    scan = scan_project(project, projects_root)
    report = review_architecture(ArchitectureReviewRequest(
        project_id=scan.project_id,
        project_path=scan.project_path,
        present_documents=scan.present_documents,
        completed_stages=scan.completed_stages,
    ))
    graph = compile_work_graph(project, projects_root)
    runtime = check_runtime_health(launcher_path or None)
    tasks = generate_tasks_from_review(report, start_at=1)
    tasks.extend(generate_tasks_from_work_graph(graph, start_at=len(tasks) + 1))
    templates, diagnostics = discover_templates(template_registry_paths(scan.project_path))
    dry_run = build_dry_run_execution_report(graph)
    panels = [
        architecture_score_panel(report),
        gap_panel(report),
        task_backlog_panel(tasks),
        work_graph_summary_panel(graph),
        dependency_block_panel(resolve_dependencies(graph)),
        execution_validation_panel(graph),
        agent_assignment_panel(graph),
        template_panel(templates, diagnostics),
        dry_run_execution_panel(dry_run),
        runtime_delegation_panel(runtime),
    ]
    panels.extend(
        DashboardPanel(
            panel_id=module["panel_id"],
            title=module["title"],
            data=module["data"],
        )
        for module in runtime_dashboard_modules(
            scan.project_id,
            {
                "runtime": {"services": []},
                "infrastructure": {},
                "vector_databases": {},
            },
        )
    )
    panels.extend(
        DashboardPanel(
            panel_id=module["panel_id"],
            title=module["title"],
            data=module["data"],
        )
        for module in col_dashboard_panels(scan.project_path)
    )
    return {
        "project_id": scan.project_id,
        "project_path": scan.project_path,
        "panels": [_panel_to_dict(panel) for panel in panels],
    }


def _panel_to_dict(panel: DashboardPanel):
    return {
        "panel_id": panel.panel_id,
        "title": panel.title,
        "data": panel.data,
    }
