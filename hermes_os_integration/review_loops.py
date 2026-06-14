"""Autonomous architecture review loop contracts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from .architecture_first import ArchitectureReviewRequest, review_architecture
from .doc_generation import roadmap_from_review


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
