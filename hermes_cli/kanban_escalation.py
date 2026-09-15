"""Atomic ensure-escalation: one open [ESC] per [AUD-*] incident, assignee=default.

The whole find/create/link/comment path runs in one ``write_txn`` (BEGIN IMMEDIATE).
No application SQL outside this module's use of existing kanban_db helpers.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import write_txn

OPEN_PREFIX = "HERMES_ESC_OPEN_V1"
LINK_PREFIX = "HERMES_ESC_LINK_V1"
REASON_PREFIX = "HERMES_ESC_REASON_V1"
ESC_TITLE_RE = re.compile(r"^\[ESC\] (t_[a-zA-Z0-9]+) G(\d+)$")
INCIDENT_RE = re.compile(r"^t_[a-zA-Z0-9]+$")
UUID_V4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)
REASONS = frozenset({
    "open_p0", "open_p1", "revision_limit", "ambiguity", "recurrence",
})
SOURCES = frozenset({"auditor_reason", "gate", "revision_limit"})
OPEN_STATUSES = frozenset({"triage", "todo", "scheduled", "ready", "running", "blocked", "review"})
TERMINAL = frozenset({"done", "archived"})


class EnsureEscalationError(ValueError):
    """Invalid input or incident that cannot be escalated."""


def _parse_prefixed(body: str, prefix: str) -> Optional[dict[str, Any]]:
    text = (body or "").strip()
    if text.startswith(prefix + "{"):
        raw = text[len(prefix):]
    elif text.startswith(prefix + " {"):
        raw = text[len(prefix) + 1:]
    else:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _dump(prefix: str, payload: dict[str, Any]) -> str:
    return prefix + " " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_payload(data: dict[str, Any]) -> dict[str, Any]:
    incident_id = data.get("incident_id")
    reason = data.get("reason")
    source = data.get("source")
    if not isinstance(incident_id, str) or not INCIDENT_RE.match(incident_id):
        raise EnsureEscalationError("invalid incident_id")
    if reason not in REASONS:
        raise EnsureEscalationError("invalid reason")
    if source not in SOURCES:
        raise EnsureEscalationError("invalid source")
    delivery_id = data.get("delivery_id") or None
    fix_id = data.get("fix_id") or data.get("fix_task_id") or None
    if source == "gate":
        if not isinstance(delivery_id, str) or not UUID_V4.match(delivery_id):
            raise EnsureEscalationError("delivery_id required for source=gate")
    elif delivery_id:
        raise EnsureEscalationError("delivery_id only valid for source=gate")
    if source == "revision_limit":
        if not isinstance(fix_id, str) or not INCIDENT_RE.match(fix_id):
            raise EnsureEscalationError("fix_id required for source=revision_limit")
    elif fix_id:
        raise EnsureEscalationError("fix_id only valid for source=revision_limit")
    return {
        "incident_id": incident_id,
        "reason": reason,
        "source": source,
        "delivery_id": delivery_id,
        "fix_id": fix_id,
    }


def _open_esc_for_incident(conn: sqlite3.Connection, incident_id: str) -> Optional[kb.Task]:
    rows = conn.execute(
        "SELECT id FROM tasks WHERE assignee = ? AND title LIKE ? "
        "AND status NOT IN ('done', 'archived')",
        ("default", "[ESC] %"),
    ).fetchall()
    found: list[kb.Task] = []
    for row in rows:
        task = kb.get_task(conn, row["id"])
        if task is None:
            continue
        parsed = _parse_prefixed(task.body or "", OPEN_PREFIX)
        if parsed and parsed.get("incident_id") == incident_id:
            found.append(task)
    if len(found) > 1:
        raise EnsureEscalationError("multiple open ESC cards for incident")
    return found[0] if found else None


def _next_generation(conn: sqlite3.Connection, incident_id: str) -> int:
    rows = conn.execute(
        "SELECT title, body FROM tasks WHERE title LIKE ?",
        (f"[ESC] {incident_id} G%",),
    ).fetchall()
    max_g = 0
    for row in rows:
        m = ESC_TITLE_RE.match(row["title"] or "")
        if not m or m.group(1) != incident_id:
            continue
        parsed = _parse_prefixed(row["body"] or "", OPEN_PREFIX)
        if parsed and parsed.get("incident_id") != incident_id:
            continue
        max_g = max(max_g, int(m.group(2)))
    return max_g + 1


def _ensure_link_comment(conn: sqlite3.Connection, task_id: str, body: str, author: str) -> None:
    existing = conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ?", (task_id,),
    ).fetchall()
    for row in existing:
        if (row["body"] or "").strip() == body.strip():
            return
    kb.add_comment(conn, task_id, author, body)


def ensure_escalation(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    *,
    author: str = "ensure-escalation",
) -> dict[str, Any]:
    """Find or create the single open [ESC] for an [AUD-*] incident.

    Must be the only writer on ``conn`` for this incident: the whole body
    runs inside ``write_txn`` / BEGIN IMMEDIATE.
    """
    spec = _validate_payload(payload)
    incident_id = spec["incident_id"]
    with write_txn(conn):
        incident = kb.get_task(conn, incident_id)
        if incident is None:
            raise EnsureEscalationError("incident not found")
        if not (incident.title or "").startswith("[AUD-"):
            raise EnsureEscalationError("incident title must start with [AUD-")
        if incident.assignee != "auditor":
            raise EnsureEscalationError("incident assignee must remain auditor")
        if incident.status in TERMINAL:
            raise EnsureEscalationError("incident is terminal")

        existing = _open_esc_for_incident(conn, incident_id)
        created = False
        if existing is None:
            generation = _next_generation(conn, incident_id)
            open_body = _dump(OPEN_PREFIX, {
                "incident_id": incident_id,
                "generation": generation,
                "reason": spec["reason"],
                "source": spec["source"],
            })
            esc_id = kb.create_task(
                conn,
                title=f"[ESC] {incident_id} G{generation}",
                body=open_body,
                assignee="default",
                created_by=author,
                idempotency_key=f"ESC:{incident_id}:G{generation}",
            )
            esc = kb.get_task(conn, esc_id)
            if esc is None or esc.status != "ready" or esc.assignee != "default":
                raise EnsureEscalationError("created ESC is not ready/default")
            created = True
        else:
            if existing.status != "ready" or existing.assignee != "default":
                # Already claimed/running is still the open ESC; do not spawn another.
                if existing.status not in OPEN_STATUSES or existing.assignee != "default":
                    raise EnsureEscalationError("open ESC is not usable")
            esc = existing
            generation = 0
            parsed = _parse_prefixed(esc.body or "", OPEN_PREFIX) or {}
            generation = int(parsed.get("generation") or 0)

        link = _dump(LINK_PREFIX, {
            "escalation_id": esc.id,
            "incident_id": incident_id,
            "generation": generation,
        })
        reason_body = _dump(REASON_PREFIX, {
            "incident_id": incident_id,
            "reason": spec["reason"],
            "source": spec["source"],
            "delivery_id": spec["delivery_id"],
            "fix_id": spec["fix_id"],
        })
        _ensure_link_comment(conn, incident_id, link, author)
        _ensure_link_comment(conn, esc.id, link, author)
        kb.add_comment(conn, esc.id, author, reason_body)
        kb.add_comment(conn, incident_id, author, reason_body)

        # Incident owner must be unchanged even after comments.
        still = kb.get_task(conn, incident_id)
        if still is None or still.assignee != "auditor":
            raise EnsureEscalationError("incident assignee changed")

        return {
            "ok": True,
            "escalation_id": esc.id,
            "created": created,
            "generation": generation,
            "appended_reason": not created,
            "incident_id": incident_id,
            "incident_assignee": "auditor",
        }
