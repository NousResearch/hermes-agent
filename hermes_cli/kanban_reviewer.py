"""Structured reviewer-result V1 routing for native Kanban tasks.

This module deliberately uses :mod:`kanban_db` for task lifecycle and graph
persistence. Validation happens before any graph mutation; rejected payloads
are recorded as sanitized audit events on the reviewed task.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb

SCHEMA_VERSION = 1
VERDICTS = frozenset({"APPROVED", "CHANGES_REQUESTED", "BLOCKED"})
SEVERITIES = frozenset({"low", "medium", "high", "critical"})
MAX_CORRECTION_CYCLES = 3


@dataclass(frozen=True)
class ReviewerResult:
    verdict: str
    summary: str
    findings: tuple[dict[str, Any], ...]
    ambiguity_or_blocker_reason: Optional[str]


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_reviewer_result(payload: Mapping[str, Any]) -> ReviewerResult:
    """Validate and normalize a schema-v1 result without touching the DB."""
    if not isinstance(payload, Mapping):
        raise ValueError("reviewer result must be an object")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported reviewer result schema_version")
    verdict = payload.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("verdict must be APPROVED, CHANGES_REQUESTED, or BLOCKED")
    if not _text(payload.get("summary")):
        raise ValueError("summary must be a non-empty string")
    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("findings must be a list")
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in raw_findings:
        if not isinstance(finding, Mapping):
            raise ValueError("each finding must be an object")
        required = ("finding_id", "severity", "affected_files_or_areas", "required_changes", "verification_evidence")
        if any(key not in finding for key in required):
            raise ValueError("finding is missing a required field")
        fid = finding["finding_id"]
        if not _text(fid) or fid in seen:
            raise ValueError("finding_id must be unique and non-empty")
        if finding["severity"] not in SEVERITIES:
            raise ValueError("finding severity is invalid")
        for key in required[2:]:
            values = finding[key]
            if not isinstance(values, list) or not values or any(not _text(v) for v in values):
                raise ValueError(f"{key} must be a non-empty list of strings")
        seen.add(fid)
        findings.append({key: finding[key] for key in required})
    reason = payload.get("ambiguity_or_blocker_reason")
    if reason is not None and not _text(reason):
        raise ValueError("ambiguity_or_blocker_reason must be non-empty when supplied")
    if verdict == "CHANGES_REQUESTED" and not findings:
        raise ValueError("CHANGES_REQUESTED requires concrete findings")
    if verdict == "BLOCKED" and not _text(reason):
        raise ValueError("BLOCKED requires ambiguity_or_blocker_reason")
    if verdict == "APPROVED" and findings:
        raise ValueError("APPROVED cannot contain findings")
    return ReviewerResult(verdict, payload["summary"].strip(), tuple(findings), reason.strip() if reason else None)


def _canonical(result: ReviewerResult) -> str:
    return json.dumps({"schema_version": 1, "verdict": result.verdict, "summary": result.summary,
                       "findings": list(result.findings), "ambiguity_or_blocker_reason": result.ambiguity_or_blocker_reason},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _audit(conn, task_id: str, kind: str, payload: Mapping[str, Any]) -> None:
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, kind, dict(payload))


def submit_reviewer_result(conn, task_id: str, payload: Mapping[str, Any], *, reviewer: str = "reviewer") -> dict[str, Any]:
    """Persist a validated reviewer result and route it through native Kanban.

    Invalid input is auditable but never changes task status, links, or creates a
    correction. Correction children are idempotent by reviewed task + canonical
    result and are capped at three cycles.
    """
    try:
        result = validate_reviewer_result(payload)
    except (TypeError, ValueError) as exc:
        _audit(conn, task_id, "reviewer_result_rejected", {"schema_version": 1, "reason": str(exc), "reviewer": reviewer})
        return {"accepted": False, "reason": str(exc), "verdict": None}
    task = kb.get_task(conn, task_id)
    if task is None:
        return {"accepted": False, "reason": "unknown task", "verdict": result.verdict}
    canonical = _canonical(result)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    base = {"schema_version": 1, "verdict": result.verdict, "reviewer": reviewer, "payload": json.loads(canonical), "payload_digest": digest}
    if result.verdict == "APPROVED":
        if task.status != "done":
            kb.complete_task(conn, task_id, summary=result.summary, result="approved")
        _audit(conn, task_id, "reviewer_result_approved", base)
        return {"accepted": True, "verdict": result.verdict, "task_id": task_id, "correction_task_id": None}
    if result.verdict == "BLOCKED":
        if task.status != "blocked":
            kb.block_task(conn, task_id, reason=result.ambiguity_or_blocker_reason or result.summary, kind="needs_input")
        _audit(conn, task_id, "reviewer_result_blocked", base)
        return {"accepted": True, "verdict": result.verdict, "task_id": task_id, "correction_task_id": None}
    prior = conn.execute("SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = 'reviewer_correction_created'", (task_id,)).fetchone()["n"]
    if prior >= MAX_CORRECTION_CYCLES:
        kb.block_task(conn, task_id, reason="maximum reviewer correction cycles reached", kind="needs_input")
        _audit(conn, task_id, "reviewer_result_escalated", {**base, "cycle": prior})
        return {"accepted": True, "verdict": result.verdict, "task_id": task_id, "correction_task_id": None, "escalated": True}
    key = f"reviewer-correction:{task_id}:{digest}"
    correction = kb.create_task(conn, title=f"Correction for {task.title}", body=result.summary, assignee=task.assignee or "builder", parents=[task_id], idempotency_key=key)
    _audit(conn, task_id, "reviewer_correction_created", {**base, "cycle": prior + 1, "correction_task_id": correction})
    return {"accepted": True, "verdict": result.verdict, "task_id": task_id, "correction_task_id": correction, "cycle": prior + 1}


apply_reviewer_result = submit_reviewer_result
record_reviewer_result = submit_reviewer_result
