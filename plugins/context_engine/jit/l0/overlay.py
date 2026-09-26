"""L0 Session overlay and WAL event persistence."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_authority_for_role(origin: str, role: str) -> float:
    if origin == "direct_user" or role == "user":
        return 1.0
    if origin == "tool" or role == "tool":
        return 0.9
    if origin == "system" or role == "system":
        return 0.8
    return 0.0


def ensure_session(
    conn: sqlite3.Connection,
    session_id: str,
    default_scope: str = "general",
) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor = conn.execute(
        "SELECT session_id, active_scope, scope_epoch, scope_confidence, last_seq, last_cwd FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = cursor.fetchone()
    if row:
        return dict(row)

    conn.execute(
        """
        INSERT INTO sessions (session_id, active_scope, scope_epoch, scope_confidence, last_seq, created_at, updated_at)
        VALUES (?, ?, 1, 1.0, 0, ?, ?)
        """,
        (session_id, default_scope, now, now),
    )
    conn.commit()
    return {
        "session_id": session_id,
        "active_scope": default_scope,
        "scope_epoch": 1,
        "scope_confidence": 1.0,
        "last_seq": 0,
        "last_cwd": None,
    }


def get_session_cwd(conn: sqlite3.Connection, session_id: str) -> Optional[str]:
    cursor = conn.execute(
        "SELECT last_cwd FROM sessions WHERE session_id = ?", (session_id,)
    )
    row = cursor.fetchone()
    return row["last_cwd"] if row else None


def append_event(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    origin: str = "direct_user",
    turn_id: Optional[str] = None,
    fact_key: Optional[str] = None,
    fact_value: Optional[str] = None,
) -> Tuple[str, int]:
    ensure_session(conn, session_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    content_hash = compute_content_hash(content)
    authority = get_authority_for_role(origin, role)

    cursor = conn.execute(
        "SELECT last_seq FROM sessions WHERE session_id = ?", (session_id,)
    )
    row = cursor.fetchone()
    seq = (row["last_seq"] if row else 0) + 1

    conn.execute(
        """
        INSERT INTO events (event_id, session_id, seq, turn_id, origin, role, content, content_hash, authority, effective_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            session_id,
            seq,
            turn_id,
            origin,
            role,
            content,
            content_hash,
            authority,
            now,
            now,
        ),
    )

    if fact_key and fact_value:
        entry_id = f"ovl_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO overlay (entry_id, session_id, kind, key, value, source_event_id, seq, authority, created_at, updated_at)
            VALUES (?, ?, 'statement', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                session_id,
                fact_key,
                fact_value,
                event_id,
                seq,
                authority,
                now,
                now,
            ),
        )

    conn.execute(
        "UPDATE sessions SET last_seq = ?, updated_at = ? WHERE session_id = ?",
        (seq, now, session_id),
    )
    conn.commit()
    return event_id, seq


def get_active_overlays(
    conn: sqlite3.Connection, session_id: str, limit: int = 10
) -> List[Dict[str, Any]]:
    cursor = conn.execute(
        "SELECT entry_id, kind, key, value, authority FROM overlay WHERE session_id = ? AND status = 'active' ORDER BY seq DESC LIMIT ?",
        (session_id, limit),
    )
    return [dict(r) for r in cursor.fetchall()]
