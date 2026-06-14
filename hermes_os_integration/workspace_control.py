"""Cross-project workspace control plane summaries."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .architecture_first import ArchitectureReviewRequest, review_architecture
from .dashboard import DashboardPanel
from .scanners import discover_projects, scan_project


@dataclass(frozen=True)
class ProjectWorkspaceSummary:
    project_id: str
    project_path: str
    architecture_score: int
    blockers: List[str] = field(default_factory=list)
    approvals: List[str] = field(default_factory=list)
    runtime_usage: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceSummary:
    projects: List[ProjectWorkspaceSummary]
    blocker_count: int
    approval_count: int


def build_workspace_summary(projects_root: str):
    summaries = []
    for discovered in discover_projects(projects_root):
        scan = scan_project(discovered["project_path"], projects_root)
        report = review_architecture(ArchitectureReviewRequest(
            project_id=scan.project_id,
            project_path=scan.project_path,
            present_documents=scan.present_documents,
            completed_stages=scan.completed_stages,
        ))
        summaries.append(ProjectWorkspaceSummary(
            project_id=scan.project_id,
            project_path=scan.project_path,
            architecture_score=report.architecture_score,
            blockers=report.critical_gaps,
            approvals=report.missing_approvals,
            runtime_usage={"status": "unknown"},
        ))
    return WorkspaceSummary(
        projects=summaries,
        blocker_count=sum(len(item.blockers) for item in summaries),
        approval_count=sum(len(item.approvals) for item in summaries),
    )


def workspace_dashboard_panel(summary: WorkspaceSummary):
    return DashboardPanel(
        panel_id="workspace-control-plane",
        title="Workspace Control Plane",
        data={
            "project_count": len(summary.projects),
            "blocker_count": summary.blocker_count,
            "approval_count": summary.approval_count,
            "projects": [asdict(project) for project in summary.projects],
        },
    )


def blocker_approval_summary(summary: WorkspaceSummary):
    return {
        "blockers_by_project": {project.project_id: project.blockers for project in summary.projects},
        "approvals_by_project": {project.project_id: project.approvals for project in summary.projects},
        "highest_risk": [project.project_id for project in summary.projects if project.blockers],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hermes workspace")
    parser.add_argument("--projects-root", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    summary = build_workspace_summary(args.projects_root)
    if args.json:
        sys.stdout.write(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write("Workspace projects: " + str(len(summary.projects)) + "\n")
        sys.stdout.write("Blockers: " + str(summary.blocker_count) + "\n")
        sys.stdout.write("Approvals: " + str(summary.approval_count) + "\n")
    return 2 if summary.blocker_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
