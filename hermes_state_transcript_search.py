"""Scoped display search and bounded match windows over a compression lineage."""

from contextlib import contextmanager

from agent.compaction_display import project_compaction_message_for_display
from hermes_state_messages import DISPLAY_VISIBLE_SQL
from hermes_state_search import SessionSearchMixin, _flatten_text, _LIKE_SNIPPET_SQL, _search_select_sql
from hermes_state_timeline import _snapshot


_BODY_COLUMNS = "content, role, display_kind, _compressed_summary"
_IDENTITY_COLUMNS = "role, content, timestamp, tool_call_id, tool_calls, tool_name, display_kind, display_metadata"


def transcript_display_rows_sql(session_ids):
    """Fixed-width identities only; payloads are read for the selected window."""
    placeholders = ",".join("?" for _ in session_ids)
    return f"""WITH candidates AS (
        SELECT id, active, COALESCE(display_identity,
            transcript_identity({_IDENTITY_COLUMNS})) AS identity
        FROM messages WHERE session_id IN ({placeholders})
          AND (active = 1 OR compacted = 1)
          AND transcript_body({_BODY_COLUMNS}) IS NOT NULL{DISPLAY_VISIBLE_SQL}
    ), ranked AS (
        SELECT id, identity, MIN(id) OVER (PARTITION BY identity) AS sort_id,
               ROW_NUMBER() OVER (PARTITION BY identity ORDER BY active DESC, id DESC) AS preference
        FROM candidates
    ), display_rows AS (
        SELECT id AS row_id, identity, sort_id FROM ranked WHERE preference = 1
    )""", tuple(session_ids)


def transcript_search_filter(session_ids):
    sql, params = transcript_display_rows_sql(session_ids)
    return f"m.id IN ({sql} SELECT row_id FROM display_rows)", params


def _register_display_functions(db, conn):
    keys = _IDENTITY_COLUMNS.split(", ")

    def identity(*values):
        return db._display_identity(db._display_dedupe_key(dict(zip(keys, values))))

    def body(content, role, kind, summary):
        projected = project_compaction_message_for_display({
            "content": db._decode_content(content), "role": role,
            "display_kind": kind, "_compressed_summary": bool(summary)})
        if projected is None or projected.get("display_kind") == "hidden":
            return None
        return _flatten_text(projected.get("content"))

    conn.create_function("transcript_identity", len(keys), identity, deterministic=True)
    conn.create_function("transcript_body", 4, body, deterministic=True)


class _SnapshotSearch(SessionSearchMixin):
    """Keep every search route on the snapshot that registered display identities."""

    def __init__(self, db, conn):
        self.db, self.conn = db, conn

    def __getattr__(self, name):
        return getattr(self.db, name)

    def _read_all(self, sql, params=()):
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def _read_one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def _like_rows(self, where, params, *, order_by, limit_sql):
        # Search the same human-visible text that the existing history projector
        # renders; never match a base64 image or an internal compaction handoff.
        source = f"""(SELECT id, session_id, role, timestamp, tool_name, tool_calls, active, compacted,
                     '' AS display_kind, transcript_body({_BODY_COLUMNS}) AS content FROM messages) m"""
        return self._read_all(_search_select_sql(_LIKE_SNIPPET_SQL, source, where, order_by, limit_sql), params)

    @contextmanager
    def _read_ctx(self):
        yield self.conn


def search_transcript(db, session_id, query, *, limit=50, offset=0):
    session_ids = db._resume_lineage_ids(session_id)
    with _snapshot(db) as conn:
        _register_display_functions(db, conn)
        # This is an explicit full-transcript search: tool bodies beyond their
        # bounded FTS prefix must remain searchable via the shared canonical fallback.
        matches = _SnapshotSearch(db, conn)._search_messages_impl(
            query, session_ids=session_ids,
            role_filter=["user", "assistant", "tool"], sort="oldest", limit=limit + 1, offset=offset,
            fields=["id", "role", "snippet", "timestamp"])
    has_more = len(matches) > limit
    return {"results": [{"row_id": row["id"], "role": row["role"],
                         "snippet": row["snippet"], "timestamp": row["timestamp"]} for row in matches[:limit]],
            "pagination": {"limit": limit, "offset": offset, "has_more": has_more,
                           "next_offset": offset + limit if has_more else None}}


def get_transcript_match_window(db, session_id, row_id, *, limit=120):
    session_ids = db._resume_lineage_ids(session_id)
    with _snapshot(db) as conn:
        _register_display_functions(db, conn)
        sql, scope = transcript_display_rows_sql(session_ids)
        anchor = conn.execute(sql + """
            SELECT d.row_id, d.sort_id FROM display_rows d
            JOIN candidates c ON c.identity = d.identity WHERE c.id = ?
        """, (*scope, row_id)).fetchone()
        if anchor is None:
            return None
        start = anchor["sort_id"]
        counts = conn.execute(sql + """
            SELECT COUNT(*) AS total, COALESCE(SUM(sort_id < ?), 0) AS offset FROM display_rows
        """, (*scope, start)).fetchone()
        rows = conn.execute(sql + """
            SELECT m.* FROM (SELECT row_id, sort_id FROM display_rows
                            WHERE sort_id >= ? ORDER BY sort_id LIMIT ?) AS page
            JOIN messages m ON m.id = page.row_id ORDER BY page.sort_id
        """, (*scope, start, limit)).fetchall()
    messages = [db._row_to_message_dict(row, warn_context="search jump", summary_flag=True) for row in rows]
    return {"messages": messages, "pagination": {
        "row_id": anchor["row_id"], "limit": limit, "returned": len(messages), "order": "oldest",
        "offset": counts["offset"], "total": counts["total"], "has_older": counts["offset"] > 0,
        "has_newer": counts["offset"] + len(messages) < counts["total"]}}
