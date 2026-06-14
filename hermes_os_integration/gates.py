"""Execution gates that enforce architecture readiness."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from .architecture_first import runtime_delegation_readiness, specification_to_tasks


CODING_TASK_TYPES = {"coding", "implementation", "refactor", "deployment"}


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    status: str
    missing: List[str] = field(default_factory=list)
    message: str = ""
    audit: Dict[str, str] = field(default_factory=dict)


def evaluate_task_execution(task_type: str, completed_stages: Iterable[str], dry_run: bool = False, override: Optional[Dict[str, str]] = None):
    readiness = runtime_delegation_readiness(completed_stages, dry_run=dry_run)
    if override:
        return GateResult(
            allowed=True,
            status="override",
            missing=list(readiness["missing"]),
            message="Human override approved execution.",
            audit=create_override_audit(override.get("approver", ""), override.get("reason", "")),
        )
    if task_type in CODING_TASK_TYPES and not readiness["ready"]:
        return GateResult(
            allowed=False,
            status="blocked",
            missing=list(readiness["missing"]),
            message="Architecture prerequisites missing before coding execution.",
        )
    return GateResult(
        allowed=True,
        status="allowed",
        missing=list(readiness["missing"]),
        message="Task execution allowed.",
    )


def create_override_audit(approver: str, reason: str):
    return {
        "approver": str(approver),
        "reason": str(reason),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "architecture_gate_override",
    }


def enforce_task_traceability(task: Dict[str, object]):
    missing = [key for key in ["specification_ref", "review_ref"] if not task.get(key)]
    if missing:
        return GateResult(
            allowed=False,
            status="blocked",
            missing=missing,
            message="Generated task lacks required traceability.",
        )
    return GateResult(allowed=True, status="allowed", message="Task traceability satisfied.")


def generate_traceable_tasks(specification: Dict[str, object], architecture_approved: bool, review_ref: str):
    tasks, error = specification_to_tasks(specification, architecture_approved)
    if error:
        return None, error
    for task in tasks:
        task["specification_ref"] = str(specification["spec_id"])
        task["review_ref"] = str(review_ref)
    return tasks, None
