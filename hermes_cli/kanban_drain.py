"""Safe Kanban drain controller slices.

This module intentionally implements only the narrow review-packet
PASS/FAIL consumption path.  It plans by default and mutates only when
``apply=True`` so operators can inspect the same report shape before
committing changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

DRAIN_AUTHOR = "kanban-drain-controller"
REVIEW_PASS = "consume_review_pass"
REVIEW_FAIL = "consume_review_fail"
_HOLD_RE = re.compile(
    r"\b("
    r"operator|credential|credentials|secret|secrets|oauth|"
    r"production|prod|deploy|deployment|rollout|payment|payments|pii|"
    r"public[- ]posting|approval|approve|decision"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReviewPacket:
    review_task_id: str
    source_task_id: str
    approved: bool
    evidence: str
    findings: list[str]
    run_id: Optional[int] = None
    comment_id: Optional[int] = None


def drain_review_packets(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Plan or consume completed Review Packet PASS/FAIL rows.

    The only mutating classes are:
    - PASS: audit-comment the blocked review-required source and complete it.
    - FAIL: audit-comment the source and create/identify one idempotent rework
      task for the original source assignee.

    Human/credential/prod-risk block text, non-review-required blocks, missing
    parents, malformed packets, and source-card state drift are reported as
    refusals/skips rather than mutated.
    """
    packets = _completed_review_packets(conn)
    if limit is not None:
        packets = packets[: max(0, int(limit))]

    actions: list[dict[str, Any]] = []
    for packet in packets:
        action = _build_action(conn, packet)
        if apply and action["status"] == "planned":
            action = _apply_action(conn, packet, action)
        actions.append(action)

    summary = {
        "total_packets": len(packets),
        "planned": sum(1 for a in actions if a["status"] == "planned"),
        "applied": sum(1 for a in actions if a["status"] == "applied"),
        "already_applied": sum(1 for a in actions if a["status"] == "already_applied"),
        "refused": sum(1 for a in actions if a["status"] == "refused"),
        "skipped": sum(1 for a in actions if a["status"] == "skipped"),
    }
    return {
        "dry_run": not apply,
        "class": "review_packets",
        "summary": summary,
        "actions": actions,
    }


def _completed_review_packets(conn: sqlite3.Connection) -> list[ReviewPacket]:
    packets: list[ReviewPacket] = []
    seen: set[tuple[str, str]] = set()

    done_tasks = conn.execute(
        "SELECT id FROM tasks WHERE status = 'done' ORDER BY completed_at ASC, id ASC"
    ).fetchall()
    for task_row in done_tasks:
        review_task_id = task_row["id"]
        for run in kb.list_runs(conn, review_task_id, include_active=False):
            if run.metadata:
                packet = _packet_from_dict(
                    run.metadata,
                    review_task_id=review_task_id,
                    evidence=f"run:{run.id}",
                    run_id=run.id,
                )
                if packet and (packet.review_task_id, packet.source_task_id) not in seen:
                    packets.append(packet)
                    seen.add((packet.review_task_id, packet.source_task_id))
        for comment in kb.list_comments(conn, review_task_id):
            parsed = _json_from_text(comment.body)
            if parsed:
                packet = _packet_from_dict(
                    parsed,
                    review_task_id=review_task_id,
                    evidence=f"comment:{comment.id}",
                    comment_id=comment.id,
                )
                if packet and (packet.review_task_id, packet.source_task_id) not in seen:
                    packets.append(packet)
                    seen.add((packet.review_task_id, packet.source_task_id))
    return packets


def _packet_from_dict(
    data: dict[str, Any],
    *,
    review_task_id: str,
    evidence: str,
    run_id: Optional[int] = None,
    comment_id: Optional[int] = None,
) -> Optional[ReviewPacket]:
    if data.get("drain_action") or data.get("drain_action_id"):
        return None

    packet_data = data
    nested = data.get("review_packet") or data.get("review_packet_metadata")
    if isinstance(nested, dict):
        packet_data = nested

    source_task_id = _first_text(
        packet_data,
        "source_task_id",
        "source_card",
        "source_card_id",
        "source_task",
    )
    if not source_task_id:
        return None

    approved = _approved_value(packet_data)
    if approved is None:
        return None

    findings_raw = packet_data.get("findings") or packet_data.get("blocking_findings")
    findings: list[str]
    if isinstance(findings_raw, list):
        findings = [str(item) for item in findings_raw if str(item).strip()]
    elif isinstance(findings_raw, str) and findings_raw.strip():
        findings = [findings_raw.strip()]
    else:
        findings = []

    return ReviewPacket(
        review_task_id=review_task_id,
        source_task_id=source_task_id,
        approved=approved,
        evidence=evidence,
        findings=findings,
        run_id=run_id,
        comment_id=comment_id,
    )


def _approved_value(data: dict[str, Any]) -> Optional[bool]:
    if "approved" in data:
        value = data.get("approved")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "approved"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "rejected"}:
                return False
    verdict = data.get("verdict")
    if isinstance(verdict, str):
        normalized = verdict.strip().upper()
        if normalized == "PASS":
            return True
        if normalized == "FAIL":
            return False
    return None


def _first_text(data: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_from_text(body: str) -> Optional[dict[str, Any]]:
    stripped = body.strip()
    if not stripped:
        return None
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_action(conn: sqlite3.Connection, packet: ReviewPacket) -> dict[str, Any]:
    action_name = REVIEW_PASS if packet.approved else REVIEW_FAIL
    action_id = _drain_action_id(packet, action_name)
    action = {
        "drain_action_id": action_id,
        "action": action_name,
        "source_task_id": packet.source_task_id,
        "review_task_id": packet.review_task_id,
        "approved": packet.approved,
        "evidence": [packet.evidence],
        "status": "planned",
    }
    source = kb.get_task(conn, packet.source_task_id)
    if source is None:
        action.update(status="refused", reason="source task not found")
        return action
    review = kb.get_task(conn, packet.review_task_id)
    if review is None or review.status != "done":
        action.update(status="refused", reason="review task is not completed")
        return action
    if not _parents_terminal(conn, packet.source_task_id):
        action.update(status="refused", reason="source task has unsatisfied parent dependencies")
        return action

    latest_block = _latest_block_reason(conn, packet.source_task_id)
    if source.status == "done" and _comment_exists(conn, packet.source_task_id, action_id):
        action.update(status="already_applied")
        return action
    if source.status != "blocked":
        action.update(status="skipped", reason=f"source task is {source.status!r}, not 'blocked'")
        return action
    if not latest_block:
        action.update(status="refused", reason="source task has no sticky block event")
        return action
    if not latest_block.lower().startswith("review-required:"):
        action.update(status="refused", reason="latest source block is not review-required")
        return action
    if _HOLD_RE.search(latest_block):
        action.update(status="refused", reason="latest source block text looks like a human/credential/prod-risk hold")
        return action

    if packet.approved:
        if _comment_exists(conn, packet.source_task_id, action_id) and source.status == "done":
            action.update(status="already_applied")
    else:
        rework_key = _rework_idempotency_key(packet)
        action["rework_idempotency_key"] = rework_key
        existing = _find_task_by_idempotency_key(conn, rework_key)
        if existing:
            action["rework_task_id"] = existing
        if _comment_exists(conn, packet.source_task_id, action_id) and existing:
            action.update(status="already_applied")
    return action


def _apply_action(
    conn: sqlite3.Connection,
    packet: ReviewPacket,
    action: dict[str, Any],
) -> dict[str, Any]:
    action_id = str(action["drain_action_id"])
    source = kb.get_task(conn, packet.source_task_id)
    if source is None:
        action.update(status="refused", reason="source task disappeared before apply")
        return action

    if packet.approved:
        comment_existed = _comment_exists(conn, packet.source_task_id, action_id)
        if not comment_existed:
            kb.add_comment(
                conn,
                packet.source_task_id,
                DRAIN_AUTHOR,
                _audit_comment(packet, REVIEW_PASS, action_id),
            )
        completed = False
        current = kb.get_task(conn, packet.source_task_id)
        if current and current.status == "blocked":
            completed = kb.complete_task(
                conn,
                packet.source_task_id,
                summary=(
                    "Review-approved and drain-consumed: "
                    f"review task {packet.review_task_id} approved this source card."
                ),
                metadata={
                    "drain_action": REVIEW_PASS,
                    "drain_action_id": action_id,
                    "review_task_id": packet.review_task_id,
                    "review_evidence": [packet.evidence],
                    "source_task_id": packet.source_task_id,
                    "approved": True,
                },
            )
        if completed or not comment_existed:
            action.update(status="applied")
        else:
            action.update(status="already_applied")
        return action

    rework_key = _rework_idempotency_key(packet)
    existing_rework = _find_task_by_idempotency_key(conn, rework_key)
    comment_existed = _comment_exists(conn, packet.source_task_id, action_id)
    if existing_rework is None:
        rework_id = kb.create_task(
            conn,
            title=f"rework after review FAIL: {source.title}",
            body=_rework_body(source, packet, action_id),
            assignee=source.assignee,
            created_by=DRAIN_AUTHOR,
            priority=source.priority,
            idempotency_key=rework_key,
        )
    else:
        rework_id = existing_rework
    if not comment_existed:
        kb.add_comment(
            conn,
            packet.source_task_id,
            DRAIN_AUTHOR,
            _audit_comment(packet, REVIEW_FAIL, action_id, rework_task_id=rework_id),
        )
    action["rework_task_id"] = rework_id
    if comment_existed and existing_rework is not None:
        action.update(status="already_applied")
    else:
        action.update(status="applied")
    return action


def _latest_block_reason(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or row["kind"] != "blocked":
        return None
    try:
        payload = json.loads(row["payload"]) if row["payload"] else {}
    except Exception:
        payload = {}
    reason = payload.get("reason") if isinstance(payload, dict) else None
    return reason if isinstance(reason, str) else ""


def _parents_terminal(conn: sqlite3.Connection, task_id: str) -> bool:
    rows = conn.execute(
        "SELECT t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?",
        (task_id,),
    ).fetchall()
    return all(row["status"] in {"done", "archived"} for row in rows)


def _comment_exists(conn: sqlite3.Connection, task_id: str, marker: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM task_comments WHERE task_id = ? AND body LIKE ? LIMIT 1",
        (task_id, f"%{marker}%"),
    ).fetchone()
    return row is not None


def _find_task_by_idempotency_key(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (key,),
    ).fetchone()
    return row["id"] if row else None


def _drain_action_id(packet: ReviewPacket, action_name: str) -> str:
    payload = {
        "action": action_name,
        "review_task_id": packet.review_task_id,
        "source_task_id": packet.source_task_id,
        "approved": packet.approved,
        "evidence": packet.evidence,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _rework_idempotency_key(packet: ReviewPacket) -> str:
    return f"kanban-drain:review-fail:{packet.source_task_id}:{packet.review_task_id}"


def _audit_comment(
    packet: ReviewPacket,
    action_name: str,
    action_id: str,
    *,
    rework_task_id: Optional[str] = None,
) -> str:
    body = {
        "drain_action_id": action_id,
        "action": action_name,
        "source_task_id": packet.source_task_id,
        "review_task_id": packet.review_task_id,
        "approved": packet.approved,
        "evidence": [packet.evidence],
        "findings": packet.findings,
    }
    if rework_task_id:
        body["rework_task_id"] = rework_task_id
    return "kanban drain review-packet consumption:\n" + json.dumps(
        body,
        indent=2,
        sort_keys=True,
    )


def _rework_body(source: kb.Task, packet: ReviewPacket, action_id: str) -> str:
    findings = "\n".join(f"- {finding}" for finding in packet.findings) or "- Review failed; see review task evidence."
    return (
        "Review Packet FAIL rework created by Kanban drain controller.\n\n"
        f"Source card: {packet.source_task_id}\n"
        f"Review task: {packet.review_task_id}\n"
        f"Drain action id: {action_id}\n\n"
        "Findings:\n"
        f"{findings}\n\n"
        "Acceptance: address the review findings, run the relevant verification, "
        "and end with review-required for a fresh independent review."
    )
