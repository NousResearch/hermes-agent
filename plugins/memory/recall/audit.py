"""Audit-chain helpers for Recall memory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


def canonical_event_payload(row: sqlite3.Row | dict[str, Any]) -> str:
    """Canonical payload used for audit event hashing."""
    data = {
        "seq": row["seq"],
        "event_id": row["event_id"],
        "phase": row["phase"],
        "operation": row["operation"],
        "target": row["target"],
        "content_preview": row["content_preview"] or "",
        "prev_hash": row["prev_hash"] or "",
        "created_at": row["created_at"],
        "metadata_json": row["metadata_json"] or "{}",
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_event(row: sqlite3.Row | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_event_payload(row).encode("utf-8")).hexdigest()


def verify_audit_chain(conn: sqlite3.Connection) -> dict[str, Any]:
    """Verify the append-only hash chain in audit_events."""
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM audit_events ORDER BY seq ASC").fetchall()
        prev_hash = ""
        for row in rows:
            if (row["prev_hash"] or "") != prev_hash:
                return {"ok": False, "failed_seq": row["seq"], "reason": "prev_hash mismatch"}
            expected = hash_event(row)
            if row["event_hash"] != expected:
                return {"ok": False, "failed_seq": row["seq"], "reason": "event_hash mismatch"}
            prev_hash = row["event_hash"]
        return {"ok": True, "count": len(rows), "head": prev_hash}
    finally:
        conn.row_factory = old_factory
