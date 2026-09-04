"""Bounded idempotency receipts for messaging-triggered Group Chat retries."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MAX_RETRY_RECEIPTS = 4096
RETRY_RECEIPT_RETENTION_SECONDS = 30 * 24 * 60 * 60
PENDING_RETRY_RECEIPT_TTL_SECONDS = 24 * 60 * 60
EXPIRED_RETRY_RESULT = (
    "This Retry expired before Hermes could confirm its outcome. "
    "Send a new Retry command."
)


class MessagingRetryReceiptError(ValueError):
    """Raised when a retry receipt conflicts or cannot be admitted safely."""


def _prepare(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_messaging_retries (
               command_id TEXT PRIMARY KEY,
               room_id TEXT NOT NULL,
               actor_json TEXT NOT NULL,
               task_ids_json TEXT NOT NULL,
               state TEXT NOT NULL,
               result_text TEXT,
               created_at REAL NOT NULL,
               updated_at REAL NOT NULL
           )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_messaging_retries_retention
           ON hosted_room_messaging_retries(state, updated_at, command_id)"""
    )


def _prune_receipts(conn: sqlite3.Connection, *, now: float) -> int:
    conn.execute(
        """UPDATE hosted_room_messaging_retries
              SET state='expired', result_text=?, updated_at=?
            WHERE state='pending' AND updated_at<?""",
        (EXPIRED_RETRY_RESULT, now, now - PENDING_RETRY_RECEIPT_TTL_SECONDS),
    )
    if conn.execute(
        """SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='hosted_rooms'"""
    ).fetchone():
        conn.execute(
            """DELETE FROM hosted_room_messaging_retries
                 WHERE room_id IN (
                     SELECT room_id FROM hosted_rooms
                      WHERE disbanded_at IS NOT NULL
                 )"""
        )
    conn.execute(
        """DELETE FROM hosted_room_messaging_retries
             WHERE command_id IN (
                 SELECT command_id FROM hosted_room_messaging_retries
                  WHERE state IN ('completed','expired') AND updated_at<?
                  ORDER BY updated_at, command_id
                  LIMIT ?
             )""",
        (now - RETRY_RECEIPT_RETENTION_SECONDS, MAX_RETRY_RECEIPTS),
    )
    count = int(
        conn.execute("SELECT COUNT(*) FROM hosted_room_messaging_retries").fetchone()[0]
    )
    if count >= MAX_RETRY_RECEIPTS:
        conn.execute(
            """DELETE FROM hosted_room_messaging_retries
                 WHERE command_id IN (
                     SELECT command_id FROM hosted_room_messaging_retries
                      WHERE state='completed'
                      ORDER BY updated_at, command_id
                      LIMIT ?
                 )""",
            (count - MAX_RETRY_RECEIPTS + 1,),
        )
        count = int(
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_messaging_retries"
            ).fetchone()[0]
        )
    return count


def retry_receipt_plan(
    db_path: Path,
    *,
    command_id: str,
    room_id: str,
    actor: Mapping[str, Any],
    task_ids: list[str],
    now: float | None = None,
) -> tuple[list[str], str | None]:
    """Freeze one delivery to a bounded retry decision."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (Group Chat retry receipts)")
        conn.execute("BEGIN IMMEDIATE")
        _prepare(conn)
        timestamp = time.time() if now is None else float(now)
        receipt_count = _prune_receipts(conn, now=timestamp)
        encoded_actor = json.dumps(
            dict(actor), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        existing = conn.execute(
            "SELECT * FROM hosted_room_messaging_retries WHERE command_id=?",
            (command_id,),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["room_id"]) != room_id
                or str(existing["actor_json"]) != encoded_actor
            ):
                raise MessagingRetryReceiptError(
                    "This retry delivery was already used for different Group Chat work."
                )
            frozen = [
                str(item)
                for item in json.loads(str(existing["task_ids_json"]))
                if str(item)
            ]
            result = (
                str(existing["result_text"])
                if existing["state"] in {"completed", "expired"}
                and existing["result_text"]
                else None
            )
            if existing["state"] == "pending":
                conn.execute(
                    """UPDATE hosted_room_messaging_retries SET updated_at=?
                        WHERE command_id=? AND state='pending'""",
                    (timestamp, command_id),
                )
            conn.commit()
            return frozen, result
        if not task_ids:
            raise MessagingRetryReceiptError(
                "This Group Chat has no failed work to retry."
            )
        if receipt_count >= MAX_RETRY_RECEIPTS:
            raise MessagingRetryReceiptError(
                "Stored Group Chat retries are full. Finish pending retries and try again."
            )
        conn.execute(
            """INSERT INTO hosted_room_messaging_retries (
                   command_id, room_id, actor_json, task_ids_json, state,
                   result_text, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?)""",
            (
                command_id,
                room_id,
                encoded_actor,
                json.dumps(task_ids, ensure_ascii=True, separators=(",", ":")),
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
        return task_ids, None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def complete_retry_receipt(
    db_path: Path,
    *,
    command_id: str,
    result: str,
    now: float | None = None,
) -> None:
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("BEGIN IMMEDIATE")
        changed = conn.execute(
            """UPDATE hosted_room_messaging_retries
                  SET state='completed', result_text=?, updated_at=?
                WHERE command_id=? AND state='pending'""",
            (result, time.time() if now is None else float(now), command_id),
        )
        if changed.rowcount not in {0, 1}:
            raise RuntimeError("Group Chat retry receipt changed more than once")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
