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
