"""Read-only Kanban board liveness scanner.

This module intentionally does not mutate board state. It is used by
``hermes kanban liveness`` and by external no-agent health wrappers that need a
small, stable JSON contract instead of scraping ``kanban list``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from hermes_cli import kanban_db as kb


@dataclass
class LivenessFinding:
    kind: str
    severity: str
    task_id: str
    title: str
    status: str
    detail: str
    assignee: Optional[str] = None
    age_minutes: Optional[float] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "kind": self.kind,
            "severity": self.severity,
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        if self.assignee:
            doc["assignee"] = self.assignee
        if self.age_minutes is not None:
            doc["age_minutes"] = round(float(self.age_minutes), 1)
        if self.data:
            doc["data"] = self.data
        return doc


def _task_age_minutes(row, now: int) -> float:
    started = row["started_at"] if "started_at" in row.keys() else None
    created = row["created_at"] if "created_at" in row.keys() else None
    base = int(started or created or now)
    return max(0.0, (now - base) / 60.0)


def _health(findings: list[LivenessFinding]) -> str:
    if not findings:
        return "ok"
    if any(f.severity == "critical" for f in findings):
        return "critical"
    if any(f.severity == "error" for f in findings):
        return "degraded"
    return "warning"


def scan_liveness(
    conn,
    *,
    tenant: Optional[str] = None,
    ready_sla_minutes: int = 240,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Return a read-only board health digest.

    Findings are deliberately conservative and operator-actionable:
    blocked tasks, stale ready tasks, and running tasks whose claim has expired
    or whose heartbeat is stale. No state transitions happen here.
    """
    now = int(now or time.time())
    params: list[Any] = []
    where = "status IN ('blocked', 'ready', 'running', 'triage', 'todo')"
    if tenant is not None:
        where += " AND tenant = ?"
        params.append(tenant)
    rows = conn.execute(
        f"""
        SELECT id, title, assignee, status, created_at, started_at,
               claim_lock, claim_expires, worker_pid, last_heartbeat_at,
               tenant, priority
          FROM tasks
         WHERE {where}
         ORDER BY priority DESC, created_at ASC
        """,
        params,
    ).fetchall()

    findings: list[LivenessFinding] = []
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
        age_min = _task_age_minutes(row, now)
        title = str(row["title"] or "")
        assignee = row["assignee"]
        task_id = str(row["id"])

        if status == "blocked":
            findings.append(LivenessFinding(
                kind="blocked_task",
                severity="warning",
                task_id=task_id,
                title=title,
                status=status,
                assignee=assignee,
                age_minutes=age_min,
                detail="Task is blocked and needs review, unblock, reroute, or an explicit decision.",
            ))
        elif status == "ready" and age_min >= max(0, int(ready_sla_minutes)):
            findings.append(LivenessFinding(
                kind="ready_sla_exceeded",
                severity="error",
                task_id=task_id,
                title=title,
                status=status,
                assignee=assignee,
                age_minutes=age_min,
                detail=f"Task has been ready for >= {int(ready_sla_minutes)} minutes without being claimed.",
            ))
        elif status == "running":
            claim_expires = row["claim_expires"]
            last_hb = row["last_heartbeat_at"]
            if claim_expires is not None and int(claim_expires) < now:
                findings.append(LivenessFinding(
                    kind="expired_running_claim",
                    severity="critical",
                    task_id=task_id,
                    title=title,
                    status=status,
                    assignee=assignee,
                    age_minutes=age_min,
                    detail="Running task claim has expired; dispatcher should reclaim or extend it on the next tick.",
                    data={"claim_expires": int(claim_expires), "claim_lock": row["claim_lock"]},
                ))
            elif last_hb is not None and (now - int(last_hb)) > kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS:
                findings.append(LivenessFinding(
                    kind="stale_running_heartbeat",
                    severity="critical",
                    task_id=task_id,
                    title=title,
                    status=status,
                    assignee=assignee,
                    age_minutes=age_min,
                    detail="Running task heartbeat is stale; worker may be wedged.",
                    data={"last_heartbeat_at": int(last_hb)},
                ))

    finding_docs = [f.to_dict() for f in findings]
    summary = {
        "health": _health(findings),
        "finding_count": len(finding_docs),
        "active_count": len(rows),
        "by_status": counts,
        "tenant": tenant,
        "ready_sla_minutes": int(ready_sla_minutes),
    }
    return {
        "schema": "hermes.kanban_liveness.v1",
        "summary": summary,
        "findings": finding_docs,
    }
