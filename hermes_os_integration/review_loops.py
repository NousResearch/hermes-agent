"""Autonomous architecture review loop contracts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from .architecture_first import ArchitectureReviewRequest, review_architecture
from .doc_generation import roadmap_from_review
from .persistence import (
    persist_review_report,
    persist_score_history,
)


@dataclass(frozen=True)
class ScheduledReview:
    schedule: str
    project_scope: List[str]
    mode: str = "read_only"
    outputs: List[str] = field(default_factory=lambda: ["review_report", "score_history", "roadmap_proposals"])


def run_review_loop(project_scans: List[Dict[str, object]], mode: str = "read_only"):
    results = []
    for scan in project_scans:
        report = review_architecture(ArchitectureReviewRequest(
            project_id=str(scan["project_id"]),
            project_path=str(scan["project_path"]),
            present_documents=list(scan.get("present_documents", [])),
            completed_stages=list(scan.get("completed_stages", [])),
        ))
        results.append({
            "project_id": report.project_id,
            "score": report.architecture_score,
            "roadmap_updates": roadmap_from_review(report),
            "requires_approval": mode != "read_only" and bool(report.critical_gaps),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return results


def run_scheduled_review_job(
    scheduled: ScheduledReview,
    project_scans: List[Dict[str, object]],
    repository=None,
):
    selected = [
        scan for scan in project_scans
        if not scheduled.project_scope or str(scan.get("project_id")) in scheduled.project_scope
    ]
    outputs = []
    for scan in selected:
        report = review_architecture(ArchitectureReviewRequest(
            project_id=str(scan["project_id"]),
            project_path=str(scan["project_path"]),
            present_documents=list(scan.get("present_documents", [])),
            completed_stages=list(scan.get("completed_stages", [])),
        ))
        review_ref = "scheduled-review:" + report.project_id
        previous = repository.latest("score-history") if repository else None
        previous_score = previous.get("score") if isinstance(previous, dict) and previous.get("project_id") == report.project_id else None
        record = {
            "project_id": report.project_id,
            "score": report.architecture_score,
            "score_delta": None if previous_score is None else report.architecture_score - int(previous_score),
            "blocked": report.blocked,
            "critical_gaps": report.critical_gaps,
            "roadmap_updates": roadmap_from_review(report),
            "requires_approval": scheduled.mode != "read_only" and bool(report.critical_gaps),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if repository:
            persist_review_report(repository, report)
            persist_score_history(
                repository,
                report.project_id + ":" + record["timestamp"],
                score_history_record(report.project_id, report.architecture_score, review_ref),
            )
        outputs.append(record)
    return {
        "schedule": scheduled.schedule,
        "mode": scheduled.mode,
        "project_count": len(outputs),
        "outputs": outputs,
    }


def scheduled_review_cron_payload(project: str, projects_root: str, schedule: str = "0 9 * * 1", mode: str = "read_only"):
    prompt = (
        "Run Hermes OS scheduled architecture review for project %s in %s mode. "
        "Persist review report, score history, and roadmap proposals; require approval before writes."
    ) % (project, mode)
    return {
        "schedule": schedule,
        "prompt": prompt,
        "name": "Hermes OS review: " + project,
        "workdir": projects_root,
        "deliver": "local",
    }


def preview_scheduled_review(scheduled: ScheduledReview, project_scans: List[Dict[str, object]]):
    selected = [
        str(scan.get("project_id"))
        for scan in project_scans
        if not scheduled.project_scope or str(scan.get("project_id")) in scheduled.project_scope
    ]
    return {
        "schedule": scheduled.schedule,
        "mode": scheduled.mode,
        "project_ids": selected,
        "would_write": scheduled.mode == "write",
        "requires_approval": scheduled.mode != "read_only",
        "outputs": scheduled.outputs,
    }


def score_history_record(project_id: str, score: int, review_ref: str):
    return {
        "project_id": project_id,
        "score": int(score),
        "review_ref": review_ref,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def autonomous_review_policy(mode: str, high_risk_write: bool = False):
    if mode not in {"read_only", "proposal", "write"}:
        return {"allowed": False, "requires_approval": True, "reason": "unknown mode"}
    if mode == "write" and high_risk_write:
        return {"allowed": False, "requires_approval": True, "reason": "high-risk autonomous write"}
    return {"allowed": True, "requires_approval": mode != "read_only", "reason": "within autonomous review policy"}
