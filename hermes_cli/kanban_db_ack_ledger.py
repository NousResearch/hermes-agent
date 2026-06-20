"""Shadow ACK delivery ledger helpers for Hermes Kanban.

This module is split out of ``kanban_db.py`` to keep the core DB file from
growing further and to make the ledger surface easy to test in isolation.
All helpers operate on an existing kanban SQLite connection and write to the
``ack_*`` tables defined in ``kanban_db.SCHEMA_SQL``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import Optional

from hermes_cli.kanban_db import write_txn


def _safe_summary(text: Optional[str], max_len: int = 400) -> Optional[str]:
    """Return a single-line, length-capped summary safe for ledger storage.

    Strips surrounding whitespace, collapses newlines, and truncates. Does not
    redact secrets — callers should run sensitive text through
    ``_sanitize_error_text`` / ``_sanitize_active_wake_text`` before storing
    error or wake payloads.
    """
    if not text:
        return None
    s = str(text).strip().replace("\r\n", " ").replace("\n", " ").strip()
    if not s:
        return None
    return s[:max_len]


def _ack_correlation_id(
    kind: str,
    task_id: str,
    platform: Optional[str],
    chat_id: Optional[str],
    thread_id: Optional[str],
    now: int,
) -> str:
    """Build a short, non-secret correlation id for an ACK delivery row.

    The id is stable for the same (kind, task, subscription target, second)
    tuple so idempotent callers do not create duplicate rows. Raw message text
    is NOT included in the correlation value.
    """
    key = f"{kind}:{task_id}:{platform or ''}:{chat_id or ''}:{thread_id or ''}:{now}"
    return f"ack_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"


def record_ack_task_verdict(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: Optional[int] = None,
    event_id: Optional[int] = None,
    verdict: Optional[str] = None,
    status: Optional[str] = None,
    summary_ref: Optional[str] = None,
    summary_safe: Optional[str] = None,
    created_at: Optional[int] = None,
) -> int:
    """Shadow-write a durable task verdict record. Returns the new row id."""
    now = created_at if created_at is not None else int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO ack_task_verdict
                (task_id, run_id, event_id, verdict, status,
                 summary_ref, summary_safe, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                run_id,
                event_id,
                verdict,
                status,
                summary_ref,
                _safe_summary(summary_safe),
                now,
            ),
        )
        return int(cur.lastrowid)


def record_ack_subscription(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    subscription_id: Optional[int] = None,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    desired_delivery_mode: Optional[str] = None,
    active_wake_required: bool = False,
    operator_receipt_required: bool = False,
    created_at: Optional[int] = None,
) -> int:
    """Shadow-write a snapshot of the explicit subscription used for ACKs."""
    now = created_at if created_at is not None else int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO ack_subscription
                (task_id, subscription_id, platform, chat_id, thread_id,
                 notifier_profile, desired_delivery_mode, active_wake_required,
                 operator_receipt_required, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                subscription_id,
                platform,
                chat_id,
                thread_id or "",
                notifier_profile,
                desired_delivery_mode or "passive",
                1 if active_wake_required else 0,
                1 if operator_receipt_required else 0,
                now,
            ),
        )
        return int(cur.lastrowid)


def record_ack_passive_delivery(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    subscription_id: Optional[int] = None,
    message_id: Optional[str] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    created_at: Optional[int] = None,
) -> int:
    """Shadow-write a passive delivery attempt. ``error`` is sanitized."""
    from tools.send_message_tool import _sanitize_error_text

    now = created_at if created_at is not None else int(time.time())
    safe_error = _sanitize_error_text(error) if error else None
    corr = correlation_id or _ack_correlation_id(
        "passive", task_id, None, None, None, now
    )
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO ack_passive_delivery
                (task_id, subscription_id, message_id, status, error_safe,
                 correlation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                subscription_id,
                message_id,
                status,
                safe_error,
                corr,
                now,
            ),
        )
        return int(cur.lastrowid)


def record_ack_active_wake(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    subscription_id: Optional[int] = None,
    triggered_agent: bool = False,
    trigger_error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    created_at: Optional[int] = None,
) -> int:
    """Shadow-write an active wake attempt. ``trigger_error`` is sanitized."""
    from tools.send_message_tool import _sanitize_error_text

    now = created_at if created_at is not None else int(time.time())
    safe_error = _sanitize_error_text(trigger_error) if trigger_error else None
    corr = correlation_id or _ack_correlation_id(
        "active_wake", task_id, None, None, None, now
    )
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO ack_active_wake
                (task_id, subscription_id, triggered_agent, trigger_error,
                 correlation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                subscription_id,
                1 if triggered_agent else 0,
                safe_error,
                corr,
                now,
            ),
        )
        return int(cur.lastrowid)


def record_ack_operator_receipt(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    status: str,
    actor: Optional[str] = None,
    actor_ref: Optional[str] = None,
    correlation_id: Optional[str] = None,
    created_at: Optional[int] = None,
) -> int:
    """Shadow-write an operator receipt state change.

    ``status`` must be one of: pending, observed, timed_out, escalated.
    """
    if status not in {"pending", "observed", "timed_out", "escalated"}:
        raise ValueError(f"invalid operator_receipt status: {status!r}")
    now = created_at if created_at is not None else int(time.time())
    corr = correlation_id or _ack_correlation_id(
        "operator_receipt", task_id, None, None, None, now
    )
    with write_txn(conn):
        cur = conn.execute(
            """
            INSERT INTO ack_operator_receipt
                (task_id, status, actor, actor_ref, correlation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, status, actor, actor_ref, corr, now),
        )
        return int(cur.lastrowid)


def list_ack_task_verdicts(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ack_task_verdict WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_ack_subscriptions(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ack_subscription WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_ack_passive_deliveries(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ack_passive_delivery WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_ack_active_wakes(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ack_active_wake WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_ack_operator_receipts(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM ack_operator_receipt WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]
