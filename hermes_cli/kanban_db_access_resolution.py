"""Atomic persistence for guarded Kanban access-resolution requests."""

from __future__ import annotations

import hmac
import json
import sqlite3
import time
from typing import Any

from hermes_cli.kanban_access_resolution import AccessResolutionError, compact_json

ACCESS_RESOLUTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS kanban_access_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    payload_digest TEXT NOT NULL,
    mode TEXT NOT NULL,
    machine TEXT NOT NULL,
    task_id TEXT NOT NULL,
    expected_block_event_id INTEGER NOT NULL,
    receipt_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
"""


def _authority_snapshot(conn: sqlite3.Connection, request: dict[str, Any]) -> sqlite3.Row:
    """Return the card's current block generation, or raise ``authority_mismatch``.

    A request is current only while the newest generation event is the ``blocked`` event it
    names. ``unblocked``/``promoted`` end a generation but leave ``block_kind`` set, and the
    dispatcher circuit breaker re-parks a card ``blocked`` with only a ``gave_up`` event, so
    status and ``block_kind`` alone cannot tell a stale request from a current one.
    """
    row = conn.execute(
        """
        SELECT t.assignee, t.status, t.block_kind,
               e.kind AS generation_kind, e.id AS block_event_id, e.created_at AS blocked_at
          FROM tasks t
          LEFT JOIN task_events e ON e.id = (
              SELECT id FROM task_events
               WHERE task_id = t.id
                 AND kind IN ('blocked', 'unblocked', 'promoted', 'gave_up')
               ORDER BY id DESC LIMIT 1
          )
         WHERE t.id = ?
        """,
        (request["cardId"],),
    ).fetchone()
    if row is None or (
        row["assignee"] != request["expectedAssignee"]
        or row["status"] != request["expectedStatus"]
        or row["block_kind"] != request["expectedBlockKind"]
        or row["generation_kind"] != "blocked"
        or row["block_event_id"] != request["expectedBlockEventId"]
        or row["blocked_at"] != request["expectedBlockedAt"]
    ):
        raise AccessResolutionError("authority_mismatch")
    return row


def _replayed_receipt(row: sqlite3.Row, digest: str) -> dict[str, Any]:
    if not hmac.compare_digest(str(row["payload_digest"]), digest):
        raise AccessResolutionError("request_id_conflict")
    receipt = json.loads(row["receipt_json"])
    receipt["replayed"] = True
    return receipt


def resolve_access_request(
    conn: sqlite3.Connection,
    request: dict[str, Any],
    *,
    machine: str,
) -> dict[str, Any]:
    """Reserve, audit and optionally unblock one exact generation atomically."""
    from hermes_cli import kanban_db as _kb

    if request["machine"] != machine:
        raise AccessResolutionError("machine_mismatch")
    with _kb.write_txn(conn):
        existing = conn.execute(
            "SELECT payload_digest, receipt_json FROM kanban_access_resolutions WHERE request_id = ?",
            (request["requestId"],),
        ).fetchone()
        if existing is not None:
            return _replayed_receipt(existing, request["payloadDigest"])

        authority = _authority_snapshot(conn, request)
        now = int(time.time())
        reservation = conn.execute(
            """
            INSERT INTO kanban_access_resolutions
                (request_id, payload_digest, mode, machine, task_id,
                 expected_block_event_id, receipt_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, '{}', ?)
            """,
            (
                request["requestId"], request["payloadDigest"], request["mode"], machine,
                request["cardId"], request["expectedBlockEventId"], now,
            ),
        )
        receipt_id = int(reservation.lastrowid or 0)
        comment_id = _kb._insert_comment(
            conn, request["cardId"], request["author"], request["comment"], now,
        )
        attribution = {
            "access_resolution_id": receipt_id,
            "request_id": request["requestId"],
            "payload_digest": request["payloadDigest"],
        }
        comment_event_id = _kb._append_event(
            conn,
            request["cardId"],
            "commented",
            {
                "comment_id": comment_id,
                "author": request["author"],
                "len": len(request["comment"]),
                **attribution,
            },
        )

        if request["mode"] == "route":
            # The audit comment leaves the block untouched, so the snapshot above still holds.
            unblock_event_id = None
            status_after = authority["status"]
            block_event_after = authority["block_event_id"]
            blocked_at_after = authority["blocked_at"]
        else:
            unblocked = _kb._unblock_task_in_txn(
                conn,
                request["cardId"],
                attribution={**attribution, "expected_block_event_id": request["expectedBlockEventId"]},
            )
            if unblocked is None:
                raise AccessResolutionError("authority_mismatch")
            unblock_event_id, status_after = unblocked
            block_event_after = None
            blocked_at_after = None

        receipt = {
            "version": 1,
            "receiptId": receipt_id,
            "requestId": request["requestId"],
            "payloadDigest": request["payloadDigest"],
            "mode": request["mode"],
            "machine": machine,
            "cardId": request["cardId"],
            "expectedBlockEventId": request["expectedBlockEventId"],
            "commentEventId": comment_event_id,
            "unblockEventId": unblock_event_id,
            "statusAfter": status_after,
            "blockEventIdAfter": block_event_after,
            "blockedAtAfter": blocked_at_after,
            "replayed": False,
        }
        conn.execute(
            "UPDATE kanban_access_resolutions SET receipt_json = ? WHERE id = ?",
            (compact_json(receipt), receipt_id),
        )
        return receipt
