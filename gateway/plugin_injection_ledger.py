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
from contextlib import closing

from gateway.delivery_ledger import _owner_alive, _owner_stamp
from hermes_constants import get_process_hermes_home
from hermes_cli.sqlite_util import add_column_if_missing, open_db, transaction


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plugin_injections (
            plugin_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_id TEXT,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            content_sha256 TEXT NOT NULL,
            state TEXT NOT NULL,
            delivery_obligation_id TEXT,
            last_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (plugin_id, idempotency_key)
        )
    """)
    if "session_id" not in {row[1] for row in conn.execute("PRAGMA table_info(plugin_injections)")}:
        add_column_if_missing(conn, "plugin_injections", "session_id", "session_id TEXT")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(plugin_injections)")}
    if "owner_pid" not in columns:
        add_column_if_missing(conn, "plugin_injections", "owner_pid", "owner_pid INTEGER")
    if "owner_started_at" not in columns:
        add_column_if_missing(conn, "plugin_injections", "owner_started_at", "owner_started_at INTEGER")


def _connect() -> sqlite3.Connection:
    return open_db(
        get_process_hermes_home() / "state.db",
        db_label="state.db (plugin_injection_ledger)",
        busy_timeout_ms=10_000,
        initialize=_initialize_schema,
    )


def claim(plugin_id: str, idempotency_key: str, session_key: str, content: str) -> str:
    """Claim a key once, or retry a deferred request before its turn starts.

    ``retry_after_notice`` skips the Telegram notice already delivered by the
    previous attempt. No other state can be reclaimed, including an ambiguous
    notice send or a dispatched turn.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    now = time.time()
    pid, started = _owner_stamp()
    with transaction(_connect()) as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO plugin_injections
                (plugin_id, idempotency_key, session_key, content_sha256,
                 state, owner_pid, owner_started_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)
        """, (plugin_id, idempotency_key, session_key, digest, pid, started, now, now))
        if cursor.rowcount:
            return "new"
        row = conn.execute("""
            SELECT session_key, content_sha256, state, owner_pid, owner_started_at
            FROM plugin_injections
            WHERE plugin_id=? AND idempotency_key=?
        """, (plugin_id, idempotency_key)).fetchone()
        if row is None or tuple(row[:2]) != (session_key, digest):
            raise ValueError("injection key already belongs to a different request")
        previous = row[2]
        retryable = previous in {"deferred", "notice_deferred"}
        abandoned_pre_turn = (previous in {"scheduled", "notice_sent"}
                              and not _owner_alive(row[3], row[4]))
        if retryable or abandoned_pre_turn:
            next_state = "scheduled" if previous in {"deferred", "scheduled"} else "notice_sent"
            cursor = conn.execute("""
                UPDATE plugin_injections
                SET state=?, owner_pid=?, owner_started_at=?, updated_at=?, last_error=NULL
                WHERE plugin_id=? AND idempotency_key=? AND state=?
                  AND owner_pid IS ? AND owner_started_at IS ?
            """, (next_state, pid, started, now, plugin_id, idempotency_key,
                  previous, row[3], row[4]))
            if cursor.rowcount:
                return "retry" if next_state == "scheduled" else "retry_after_notice"
        return "existing"


def bind_session(plugin_id: str, idempotency_key: str, session_id: str) -> bool:
    """Pin the first resolved session generation; reject a later /new replay."""
    if not session_id:
        return False
    with transaction(_connect()) as conn:
        cursor = conn.execute("""
            UPDATE plugin_injections SET session_id=COALESCE(session_id, ?), updated_at=?
            WHERE plugin_id=? AND idempotency_key=?
              AND (session_id IS NULL OR session_id=?)
        """, (session_id, time.time(), plugin_id, idempotency_key, session_id))
        return bool(cursor.rowcount)


def state(plugin_id: str, idempotency_key: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute("""
            SELECT session_key, session_id, state, delivery_obligation_id, last_error,
                   created_at, updated_at FROM plugin_injections
            WHERE plugin_id=? AND idempotency_key=?
        """, (plugin_id, idempotency_key)).fetchone()
        if row is None:
            return None
        session_key, session_id, lifecycle, obligation_id, error, created, updated = row
        result = {
            "session_key": session_key, "session_id": session_id, "state": lifecycle,
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
