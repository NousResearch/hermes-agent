"""Durable admission and outcome receipts for keyed plugin gateway injections.

The key is claimed before scheduling. A repeated call never starts another turn,
including after a gateway crash. The reply's ordinary delivery obligation remains
the authority for Telegram retries; this table only links that obligation to the
request that caused it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time

from hermes_constants import get_process_hermes_home
from hermes_cli.sqlite_util import open_db, transaction


def _connect() -> sqlite3.Connection:
    return open_db(
        get_process_hermes_home() / "state.db",
        db_label="state.db (plugin_injection_ledger)",
        busy_timeout_ms=10_000,
        initialize=lambda conn: conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_injections (
                plugin_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                session_key TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                delivery_obligation_id TEXT,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (plugin_id, idempotency_key)
            )
        """),
    )


def claim(plugin_id: str, idempotency_key: str, session_key: str, content: str) -> str:
    """Claim a key once, or retry a deferred request before its turn starts.

    ``retry_after_notice`` skips the Telegram notice already delivered by the
    previous attempt. No other state can be reclaimed, including an ambiguous
    notice send or a dispatched turn.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = time.time()
    with transaction(_connect()) as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO plugin_injections
                (plugin_id, idempotency_key, session_key, content_sha256,
                 state, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'scheduled', ?, ?)
        """, (plugin_id, idempotency_key, session_key, digest, now, now))
        if cursor.rowcount:
            return "new"
        row = conn.execute("""
            SELECT session_key, content_sha256, state FROM plugin_injections
            WHERE plugin_id=? AND idempotency_key=?
        """, (plugin_id, idempotency_key)).fetchone()
        if row is None or tuple(row[:2]) != (session_key, digest):
            raise ValueError("injection key already belongs to a different request")
        previous = row[2]
        if previous in {"deferred", "notice_deferred"}:
            next_state = "scheduled" if previous == "deferred" else "notice_sent"
            cursor = conn.execute("""
                UPDATE plugin_injections SET state=?, updated_at=?, last_error=NULL
                WHERE plugin_id=? AND idempotency_key=? AND state=?
            """, (next_state, now, plugin_id, idempotency_key, previous))
            if cursor.rowcount:
                return "retry" if previous == "deferred" else "retry_after_notice"
        return "existing"


def state(plugin_id: str, idempotency_key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("""
            SELECT session_key, state, delivery_obligation_id, last_error,
                   created_at, updated_at FROM plugin_injections
            WHERE plugin_id=? AND idempotency_key=?
        """, (plugin_id, idempotency_key)).fetchone()
        if row is None:
            return None
        session_key, lifecycle, obligation_id, error, created, updated = row
        result = {
            "session_key": session_key, "state": lifecycle,
            "delivery_obligation_id": obligation_id, "last_error": error,
            "created_at": created, "updated_at": updated,
        }
        if obligation_id:
            delivery = conn.execute("""
                SELECT state, last_error, content FROM delivery_obligations WHERE obligation_id=?
            """, (obligation_id,)).fetchone()
            if delivery:
                result["delivery_state"], result["delivery_error"], result["response"] = delivery
        return result


def advance(plugin_id: str, idempotency_key: str, state: str, *,
            obligation_id: str | None = None, error: str | None = None) -> None:
    with transaction(_connect()) as conn:
        conn.execute("""
            UPDATE plugin_injections
            SET state=?, delivery_obligation_id=COALESCE(?, delivery_obligation_id),
                last_error=?, updated_at=?
            WHERE plugin_id=? AND idempotency_key=?
        """, (state, obligation_id, error, time.time(), plugin_id, idempotency_key))


def release_scheduled(plugin_id: str, idempotency_key: str, admission: str = "new") -> None:
    """Undo a failed scheduler admission without losing a retryable key."""
    with transaction(_connect()) as conn:
        if admission == "new":
            conn.execute("""
                DELETE FROM plugin_injections
                WHERE plugin_id=? AND idempotency_key=? AND state='scheduled'
            """, (plugin_id, idempotency_key))
        else:
            current = "scheduled" if admission == "retry" else "notice_sent"
            previous = "deferred" if admission == "retry" else "notice_deferred"
            conn.execute("""
                UPDATE plugin_injections SET state=?, updated_at=?
                WHERE plugin_id=? AND idempotency_key=? AND state=?
            """, (previous, time.time(), plugin_id, idempotency_key, current))
