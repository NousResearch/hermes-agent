"""Reply candidates written with their terminal receipt or detached transcript note."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from persistence.authority.admissions import ReplyCompletion

if TYPE_CHECKING:
    import sqlite3

    from persistence.sqlite._holder import StoreHolder

_RETENTION_SECONDS = 24 * 60 * 60
_MAX_PAGE_SIZE = 100


def record_reply_completion(
    conn: sqlite3.Connection, *, session_id: str, admission_id: str, kind: Literal["turn", "note"]
) -> None:
    """Write a metadata-only event in the caller's transaction."""
    now = time.time()
    conn.execute(
        "INSERT INTO session_reply_completions (session_id, admission_id, kind, completed_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (kind, admission_id) DO NOTHING",
        (session_id, admission_id, kind, now),
    )
    conn.execute(
        "DELETE FROM session_reply_completions WHERE completed_at <= ?",
        (now - _RETENTION_SECONDS,),
    )


def reply_completions(holder: StoreHolder, *, after: int, limit: int) -> list[ReplyCompletion]:
    if type(after) is not int or not 0 <= after < 2**63:
        raise ValueError("after must be a non-negative SQLite sequence")
    if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_SIZE:
        raise ValueError("limit must be between 1 and 100")
    with holder._read_ctx() as conn:
        rows = conn.execute(
            "SELECT sequence, session_id, admission_id, kind, completed_at "
            "FROM session_reply_completions WHERE sequence > ? AND completed_at > ? "
            "ORDER BY sequence LIMIT ?",
            (after, time.time() - _RETENTION_SECONDS, limit),
        ).fetchall()
    return [ReplyCompletion(**dict(row)) for row in rows]
