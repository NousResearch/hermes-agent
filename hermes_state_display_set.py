"""A display read's rows and the identities of the lineage's removed rows, read as one generation.

``SessionMessagesMixin._dedupe_display_generations(drop_removed=True)`` hides a message whose newest copy
is removed (rewound, or superseded by the copy a compaction carried). It needs the removed rows only for
their identity, never their body. Both reads must see the same generation: a compaction committing
between them marks a row removed after it was read live, so the removed read hides the message while
the live read never saw its new copy, and a display read silently drops it (a resume or a stored
transcript page alike).
"""

from __future__ import annotations

from typing import Any

_REMOVED_IDENTITIES_SQL = (
    "SELECT session_id, id, role, timestamp, tool_call_id, tool_calls, tool_name, display_kind, display_metadata, "
    "message_uid, absorbed_message_uids, display_identity, active, compacted, "
    "CASE WHEN display_identity IS NULL THEN content END AS content "
    "FROM messages WHERE session_id IN ({ids}) AND active = 0 AND compacted = 0 ORDER BY id")


def with_removed_identities(db, rows, session_ids: list[str], conn=None) -> list[Any]:
    """*rows* (ordered by id) merged with the identities of *session_ids*' removed rows (``active=0,
    compacted=0``): no body (a legacy row without a stored display identity keeps its content for the
    key). *conn*: read on the caller's transaction, else on a read of its own (*db* is a ``SessionDB`` or
    a worker's ``CompressionSnapshot``)."""
    sql = _REMOVED_IDENTITIES_SQL.format(ids=", ".join("?" for _ in session_ids))
    removed = conn.execute(sql, tuple(session_ids)).fetchall() if conn is not None else db._read_all(sql, tuple(session_ids))
    return sorted([*rows, *removed], key=lambda row: row["id"]) if removed else list(rows)


def display_set(db, sql: str, session_ids: list[str]) -> list[Any]:
    """*sql*'s rows (bound to *session_ids*) with the removed rows' identities, in ONE read transaction."""
    def read(conn):
        conn.execute("BEGIN")
        try:
            return with_removed_identities(db, conn.execute(sql, tuple(session_ids)).fetchall(), session_ids, conn)
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
    return db._read_retrying_ioerr(read)
