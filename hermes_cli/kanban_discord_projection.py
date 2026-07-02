"""Kanban issue -> Discord parent-thread projection seams.

This module keeps Discord as an append-only projection/log surface.  It never
uses Discord delivery, archive state, reactions, or typing to decide Kanban
lifecycle.  Callers persist Kanban lifecycle first, then enqueue/retry durable
projection events here.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb

LINK_STATES = {"pending", "linked", "failed", "projection_blocked"}
TERMINAL_ARCHIVE_STATUSES = {"archive_candidate", "closed"}
CHECKPOINT_TYPES = {
    "accepted",
    "claimed",
    "progress",
    "blocker",
    "waiting_for_approval",
    "handoff",
    "test",
    "review",
    "done",
    "failed",
}


class ProjectionBlocked(RuntimeError):
    """Raised when an issue cannot project to its required parent thread."""


class ParentThreadConflict(RuntimeError):
    """Raised when linking would violate the 1 issue <-> 1 parent thread rule."""


@dataclass(frozen=True)
class ParentThread:
    issue_id: str
    guild_id: str
    channel_id: str
    thread_id: str
    link_state: str


@dataclass(frozen=True)
class LinkResult:
    issue_id: str
    thread_id: str
    created: bool
    link_state: str = "linked"


@dataclass(frozen=True)
class OutboxEvent:
    id: int
    event_type: str
    issue_id: str
    parent_thread_id: str
    idempotency_key: str
    delivery_status: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ProjectionResult:
    delivered: bool
    event_id: Optional[int] = None
    idempotency_key: Optional[str] = None
    message_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TypingIndicatorState:
    last_sent_by_thread: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchiveDryRunPlan:
    issue_id: str
    status: Optional[str]
    allowed: bool
    live_mode: bool
    reasons: list[str]
    parent_thread_id: Optional[str]


def _now() -> int:
    return int(time.time())


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def mark_projection_required(conn: sqlite3.Connection, issue_id: str) -> None:
    """Enroll an issue in the mandatory Discord-parent projection gate."""
    with kb.write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET discord_thread_link_state = COALESCE(discord_thread_link_state, 'pending')
             WHERE id = ?
            """,
            (issue_id,),
        )
        if cur.rowcount != 1:
            raise ValueError(f"unknown issue_id: {issue_id}")


def get_parent_thread(conn: sqlite3.Connection, issue_id: str) -> Optional[ParentThread]:
    row = conn.execute(
        """
        SELECT id, parent_discord_guild_id, parent_discord_channel_id,
               parent_discord_thread_id, discord_thread_link_state
          FROM tasks
         WHERE id = ?
        """,
        (issue_id,),
    ).fetchone()
    if not row or not row["parent_discord_thread_id"]:
        return None
    return ParentThread(
        issue_id=row["id"],
        guild_id=row["parent_discord_guild_id"],
        channel_id=row["parent_discord_channel_id"],
        thread_id=row["parent_discord_thread_id"],
        link_state=row["discord_thread_link_state"] or "pending",
    )


def is_dispatchable(conn: sqlite3.Connection, issue_id: str) -> bool:
    row = conn.execute(
        """
        SELECT status, parent_discord_thread_id, discord_thread_link_state
          FROM tasks
         WHERE id = ?
        """,
        (issue_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] not in {"ready", "review"}:
        return False
    return bool(row["parent_discord_thread_id"] and row["discord_thread_link_state"] == "linked")


def _ensure_thread_not_owned_by_other_active_issue(
    conn: sqlite3.Connection, issue_id: str, thread_id: str
) -> None:
    row = conn.execute(
        """
        SELECT id FROM tasks
         WHERE parent_discord_thread_id = ?
           AND id != ?
         LIMIT 1
        """,
        (thread_id, issue_id),
    ).fetchone()
    if row:
        raise ParentThreadConflict(
            f"discord thread {thread_id} is already parent for active issue {row['id']}"
        )


def link_parent_thread(
    conn: sqlite3.Connection,
    *,
    issue_id: str,
    guild_id: str,
    channel_id: str,
    thread_id: str,
    idempotency_key: str,
) -> LinkResult:
    """Idempotently bind exactly one Discord parent thread to one issue."""
    if not all([issue_id, guild_id, channel_id, thread_id, idempotency_key]):
        raise ValueError("issue_id, guild_id, channel_id, thread_id, and idempotency_key are required")
    with kb.write_txn(conn):
        row = conn.execute(
            """
            SELECT parent_discord_thread_id, discord_thread_link_state,
                   discord_thread_link_idempotency_key
              FROM tasks
             WHERE id = ?
            """,
            (issue_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown issue_id: {issue_id}")
        existing = row["parent_discord_thread_id"]
        if existing:
            if existing == thread_id:
                return LinkResult(issue_id=issue_id, thread_id=thread_id, created=False)
            raise ParentThreadConflict(
                f"issue {issue_id} already has parent Discord thread {existing}"
            )
        _ensure_thread_not_owned_by_other_active_issue(conn, issue_id, thread_id)
        conn.execute(
            """
            UPDATE tasks
               SET parent_discord_guild_id = ?,
                   parent_discord_channel_id = ?,
                   parent_discord_thread_id = ?,
                   discord_thread_link_state = 'linked',
                   discord_thread_link_idempotency_key = ?,
                   discord_thread_linked_at = ?
             WHERE id = ?
            """,
            (guild_id, channel_id, thread_id, idempotency_key, _now(), issue_id),
        )
        kb._append_event(
            conn,
            issue_id,
            "discord_parent_thread_linked",
            {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "thread_id": thread_id,
                "idempotency_key": idempotency_key,
            },
        )
    return LinkResult(issue_id=issue_id, thread_id=thread_id, created=True)


def _parent_thread_or_raise(conn: sqlite3.Connection, issue_id: str) -> ParentThread:
    parent = get_parent_thread(conn, issue_id)
    if not parent or parent.link_state != "linked":
        raise ProjectionBlocked(f"issue {issue_id} has no linked parent Discord thread")
    return parent


def _status_message(
    *, from_status: str, to_status: str, actor: str, reason: Optional[str], next_step: Optional[str]
) -> str:
    lines = [
        f"Kanban status changed: {from_status} -> {to_status}",
        f"Actor: {actor}",
    ]
    if reason:
        lines.append(f"Reason: {reason}")
    if next_step:
        lines.append(f"Next: {next_step}")
    return "\n".join(lines)


def record_status_change(
    conn: sqlite3.Connection,
    *,
    issue_id: str,
    from_status: str,
    to_status: str,
    actor: str,
    reason: Optional[str] = None,
    summary: Optional[str] = None,
    next_step: Optional[str] = None,
    sequence: int,
) -> int:
    """Create a durable Discord projection event for a Kanban state change."""
    parent = _parent_thread_or_raise(conn, issue_id)
    idempotency_key = f"kanban-status:{issue_id}:{sequence}:{from_status}->{to_status}"
    content = _status_message(
        from_status=from_status,
        to_status=to_status,
        actor=actor,
        reason=reason or summary,
        next_step=next_step,
    )
    payload = {
        "content": content,
        "from_status": from_status,
        "to_status": to_status,
        "actor": actor,
        "reason": reason,
        "summary": summary,
        "next_step": next_step,
        "sequence": sequence,
    }
    with kb.write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO discord_projection_outbox (
                event_type, issue_id, from_status, to_status, actor, reason,
                summary, sequence, parent_thread_id, idempotency_key,
                delivery_status, payload, created_at
            ) VALUES ('status_changed', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                issue_id,
                from_status,
                to_status,
                actor,
                reason,
                summary,
                int(sequence),
                parent.thread_id,
                idempotency_key,
                _json(payload),
                _now(),
            ),
        )
        row = conn.execute(
            "SELECT id FROM discord_projection_outbox WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return int(row["id"])


def _row_to_outbox(row: sqlite3.Row) -> OutboxEvent:
    payload = json.loads(row["payload"] or "{}")
    return OutboxEvent(
        id=int(row["id"]),
        event_type=row["event_type"],
        issue_id=row["issue_id"],
        parent_thread_id=row["parent_thread_id"],
        idempotency_key=row["idempotency_key"],
        delivery_status=row["delivery_status"],
        payload=payload,
    )


def list_outbox(conn: sqlite3.Connection) -> list[OutboxEvent]:
    rows = conn.execute(
        "SELECT * FROM discord_projection_outbox ORDER BY id"
    ).fetchall()
    return [_row_to_outbox(row) for row in rows]


def mark_outbox_delivered(
    conn: sqlite3.Connection, idempotency_key: str, discord_message_id: str
) -> bool:
    with kb.write_txn(conn):
        cur = conn.execute(
            """
            UPDATE discord_projection_outbox
               SET delivery_status = 'delivered', discord_message_id = ?, delivered_at = ?
             WHERE idempotency_key = ?
            """,
            (discord_message_id, _now(), idempotency_key),
        )
        return cur.rowcount == 1


def project_next_outbox_event(
    conn: sqlite3.Connection,
    send: Callable[[str, str, str], str],
) -> ProjectionResult:
    """Deliver one pending/failed projection event.

    ``send`` receives ``(thread_id, content, idempotency_key)``.  Delivery
    failure only marks the outbox row retryable; it never changes Kanban task
    status.
    """
    row = conn.execute(
        """
        SELECT * FROM discord_projection_outbox
         WHERE delivery_status IN ('pending', 'failed')
         ORDER BY id
         LIMIT 1
        """
    ).fetchone()
    if not row:
        return ProjectionResult(delivered=False)
    event = _row_to_outbox(row)
    try:
        message_id = send(
            event.parent_thread_id,
            str(event.payload.get("content") or ""),
            event.idempotency_key,
        )
    except Exception as exc:
        with kb.write_txn(conn):
            conn.execute(
                """
                UPDATE discord_projection_outbox
                   SET delivery_status = 'failed', attempts = attempts + 1,
                       last_error = ?
                 WHERE id = ?
                """,
                (str(exc), event.id),
            )
        return ProjectionResult(
            delivered=False,
            event_id=event.id,
            idempotency_key=event.idempotency_key,
            error=str(exc),
        )
    mark_outbox_delivered(conn, event.idempotency_key, str(message_id))
    return ProjectionResult(
        delivered=True,
        event_id=event.id,
        idempotency_key=event.idempotency_key,
        message_id=str(message_id),
    )


def _checkpoint_content(
    *,
    checkpoint_type: str,
    agent_profile: str,
    session_id: str,
    run_id: str,
    summary: str,
    next_step: Optional[str],
    blocker: Optional[str],
    evidence: Optional[str],
) -> str:
    lines = [
        f"Agent checkpoint: {checkpoint_type}",
        f"Agent: {agent_profile}",
        f"Session/run: {session_id} / {run_id}",
        f"Summary: {summary}",
    ]
    if next_step:
        lines.append(f"Next: {next_step}")
    if blocker:
        lines.append(f"Blocker: {blocker}")
    if evidence:
        lines.append(f"Evidence: {evidence}")
    return "\n".join(lines)


def emit_agent_checkpoint(
    conn: sqlite3.Connection,
    *,
    issue_id: str,
    checkpoint_type: str,
    agent_profile: str,
    session_id: str,
    run_id: str,
    summary: str,
    next_step: Optional[str] = None,
    blocker: Optional[str] = None,
    evidence: Optional[str] = None,
    now: Optional[int] = None,
) -> LinkResult:
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(f"unknown checkpoint_type: {checkpoint_type}")
    parent = _parent_thread_or_raise(conn, issue_id)
    semantic = _json(
        {
            "issue_id": issue_id,
            "checkpoint_type": checkpoint_type,
            "agent_profile": agent_profile,
            "session_id": session_id,
            "run_id": run_id,
            "summary": summary,
            "next_step": next_step,
            "blocker": blocker,
            "evidence": evidence,
        }
    )
    digest = hashlib.sha256(semantic.encode("utf-8")).hexdigest()[:24]
    idempotency_key = f"kanban-checkpoint:{issue_id}:{digest}"
    content = _checkpoint_content(
        checkpoint_type=checkpoint_type,
        agent_profile=agent_profile,
        session_id=session_id,
        run_id=run_id,
        summary=summary,
        next_step=next_step,
        blocker=blocker,
        evidence=evidence,
    )
    payload = {
        "content": content,
        "checkpoint_type": checkpoint_type,
        "agent_profile": agent_profile,
        "session_id": session_id,
        "run_id": run_id,
        "summary": summary,
        "next_step": next_step,
        "blocker": blocker,
        "evidence": evidence,
    }
    with kb.write_txn(conn):
        before = conn.execute(
            "SELECT id FROM discord_projection_outbox WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        conn.execute(
            """
            INSERT OR IGNORE INTO discord_projection_outbox (
                event_type, issue_id, actor, summary, parent_thread_id,
                idempotency_key, delivery_status, payload, created_at
            ) VALUES ('agent_checkpoint', ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                issue_id,
                agent_profile,
                summary,
                parent.thread_id,
                idempotency_key,
                _json(payload),
                int(now if now is not None else _now()),
            ),
        )
    return LinkResult(issue_id=issue_id, thread_id=parent.thread_id, created=before is None)


def send_typing_best_effort(
    thread_id: str,
    send_typing: Callable[[str], None],
    *,
    state: TypingIndicatorState,
    now: Optional[float] = None,
    min_interval_seconds: float = 5.0,
) -> bool:
    if not thread_id:
        return False
    current = float(now if now is not None else time.monotonic())
    previous = state.last_sent_by_thread.get(thread_id)
    if previous is not None and (current - previous) < min_interval_seconds:
        return False
    state.last_sent_by_thread[thread_id] = current
    try:
        send_typing(thread_id)
        return True
    except Exception:
        return False


def _latest_required_projection_delivered(
    conn: sqlite3.Connection, issue_id: str, status: str
) -> bool:
    row = conn.execute(
        """
        SELECT delivery_status
          FROM discord_projection_outbox
         WHERE issue_id = ?
           AND event_type = 'status_changed'
           AND to_status = ?
         ORDER BY sequence DESC, id DESC
         LIMIT 1
        """,
        (issue_id, status),
    ).fetchone()
    return bool(row and row["delivery_status"] == "delivered")


def plan_archive_dry_run(conn: sqlite3.Connection, issue_id: str) -> ArchiveDryRunPlan:
    """Evaluate archive safety gates without Discord side effects.

    MVP deliberately keeps live_mode False.  ``done`` posts status but does not
    auto-archive; only ``archive_candidate`` and ``closed`` are archive-eligible.
    """
    row = conn.execute(
        """
        SELECT status, parent_discord_thread_id, discord_thread_link_state,
               current_run_id, claim_lock
          FROM tasks
         WHERE id = ?
        """,
        (issue_id,),
    ).fetchone()
    if row is None:
        return ArchiveDryRunPlan(issue_id, None, False, False, ["issue not found"], None)

    status = row["status"]
    parent_thread_id = row["parent_discord_thread_id"]
    reasons: list[str] = []
    if status == "done":
        reasons.append("status done is not archive-eligible in MVP")
    elif status not in TERMINAL_ARCHIVE_STATUSES:
        reasons.append(f"status {status} is not archive-eligible in MVP")
    if not parent_thread_id or row["discord_thread_link_state"] != "linked":
        reasons.append("parent Discord thread is not linked")
    if row["current_run_id"] or row["claim_lock"]:
        reasons.append("active worker/session present")
    if status in TERMINAL_ARCHIVE_STATUSES and not _latest_required_projection_delivered(conn, issue_id, status):
        reasons.append("required terminal status projection is not delivered")

    allowed = not reasons
    with kb.write_txn(conn):
        conn.execute(
            """
            INSERT INTO discord_close_audit (
                issue_id, parent_thread_id, observed_status, dry_run,
                allowed, reasons, created_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?)
            """,
            (issue_id, parent_thread_id, status, 1 if allowed else 0, _json({"reasons": reasons}), _now()),
        )
    return ArchiveDryRunPlan(
        issue_id=issue_id,
        status=status,
        allowed=allowed,
        live_mode=False,
        reasons=reasons,
        parent_thread_id=parent_thread_id,
    )
