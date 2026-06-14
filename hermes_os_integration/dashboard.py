"""Architecture dashboard panel data contracts."""

from dataclasses import dataclass, field
from typing import Dict, List

from .architecture_first import ArchitectureReviewReport
from .contracts import RuntimeStatus


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
