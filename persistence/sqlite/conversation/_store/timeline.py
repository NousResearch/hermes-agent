"""The session timeline and prompt pages, read in one snapshot."""


from __future__ import annotations
import json
import re
from contextlib import contextmanager

from session_rules.turns import project_compaction_message_for_display, user_originated_turn_view
from persistence.sqlite.conversation._store.compaction_display import not_a_later_tool_copy_sql
from persistence.sqlite.conversation._store.messages import (
    _decode_content, _dedupe_display_generations, _encode_content, _row_to_message_dict,
    _with_removed_identities,
)
from persistence.sqlite.conversation._store.messages_sql import DISPLAY_VISIBLE_SQL


_SYNTHETIC_PROMPT = re.compile(
    r"^\s*(?:\[IMPORTANT: Background process |\[ASYNC (?:DELEGATION )?(?:BATCH )?COMPLETE\b|"
    r"A background fan-out of \d+ subagent\(s\) you dispatched earlier has finished\.|"
    r"A background subagent you dispatched earlier has finished\.)",
    re.IGNORECASE,
)


def _prompt_preview(db, content, display_kind, summary):
    message = project_compaction_message_for_display({
        "role": "user", "content": _decode_content(content),
        "display_kind": display_kind, "_compressed_summary": bool(summary),
    })
    if message is None or user_originated_turn_view(message) is None:
        return ""
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content if isinstance(part, (str, dict)))
    if not isinstance(content, str):
        return ""
    text = " ".join(content.split())
    if not text or _SYNTHETIC_PROMPT.match(text):
        return ""
    return text if len(text) <= 120 else text[:119].rstrip() + "…"


@contextmanager
def _snapshot(db):
    # The count and page must see the same compaction/rewind generation.
    with db._read_ctx() as conn:
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")


def _display_rows_sql(conn, session_id, *, users_only=False):
    """Return only representative ids and their stable first-row order, never bodies.

    Legacy stores cannot backfill on a GET. SQL groups their payload identities in
    SQLite; only user content crosses the Python boundary for carrier normalization.
    Current stores use the durable display index, including protected-tail copies.
    """
    filters = (" AND role = 'user'" if users_only else "") + DISPLAY_VISIBLE_SQL
    indexed = conn.execute(
        "SELECT 1 FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) "
        f"{filters} AND (display_order IS NULL OR display_identity IS NULL) LIMIT 1",
        (session_id,),
    ).fetchone() is None
    if indexed:
        return f"""WITH display_rows AS (
            SELECT (SELECT candidate.id FROM messages candidate
                    WHERE candidate.session_id = :sid
                      AND candidate.display_order = m.display_order
                      AND (candidate.active = 1 OR candidate.compacted = 1){DISPLAY_VISIBLE_SQL}{not_a_later_tool_copy_sql('candidate')}
                    ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1) AS row_id,
                   m.display_order AS sort_id
            FROM messages m WHERE session_id = :sid AND (active = 1 OR compacted = 1){filters}{not_a_later_tool_copy_sql('m')}
            GROUP BY m.display_order
        )"""
    return f"""WITH ranked AS (
        SELECT id, MIN(id) OVER identity AS sort_id,
               ROW_NUMBER() OVER (identity ORDER BY active DESC, id DESC) AS preference
        FROM messages WHERE session_id = :sid AND (active = 1 OR compacted = 1){filters}
        WINDOW identity AS (PARTITION BY role,
            CASE WHEN role = 'user' THEN timeline_identity_content(content, display_kind) ELSE content END,
            timestamp, tool_call_id, tool_calls, tool_name)
    ), display_rows AS (SELECT id AS row_id, sort_id FROM ranked WHERE preference = 1)"""


def _compression_chains(conn, segments):
    reasons = dict(conn.execute(
        "SELECT id, end_reason FROM sessions WHERE id IN (SELECT value FROM json_each(?))",
        (json.dumps(segments),),
    ))
    # The receipt runs from oldest to newest. A reset starts an independent history, even
    # when its prompts have the same content and timestamps as a prior transcript.
    start = 0
    for index, segment in enumerate(segments[:-1], start=1):
        if reasons.get(segment) != "compression":
            yield segments[start:index]
            start = index
    yield segments[start:]


def _chain_display_rows(db, conn, chain, *, users_only):
    if len(chain) == 1:
        sql = _display_rows_sql(conn, chain[0], users_only=users_only)
        return conn.execute(sql + " SELECT row_id, sort_id FROM display_rows", {"sid": chain[0]}).fetchall()

    # Use the same identity and rewind rules as a standalone compression lineage. Read only
    # the columns that determine identities now and fetch full payloads for the requested page.
    rows = conn.execute("""
        SELECT id, session_id, role, content, timestamp, tool_call_id, tool_calls, tool_name,
               display_kind, display_metadata, display_identity, display_order, message_uid,
               absorbed_message_uids, active, compacted, _compressed_summary
        FROM messages WHERE session_id IN (SELECT value FROM json_each(?))
          AND (active = 1 OR compacted = 1) ORDER BY id
    """, (json.dumps(chain),)).fetchall()
    first_ids = {}
    rows = _with_removed_identities(db, rows, chain, conn)
    chosen = _dedupe_display_generations(db, rows, drop_removed=True, first_ids=first_ids)
    return [(row["id"], first_ids[row["id"]]) for row in chosen if not users_only or row["role"] == "user"]


def _conversation_display_sql(db, conn, session_id, segments, *, users_only=False) -> tuple[str, dict[str, str | int]]:
    if not segments:
        return _display_rows_sql(conn, session_id, users_only=users_only), {"sid": session_id}
    rows = [tuple(row) for chain in _compression_chains(conn, segments)
            for row in _chain_display_rows(db, conn, chain, users_only=users_only)]
    return """WITH display_rows AS (
        SELECT value ->> 0 AS row_id, value ->> 1 AS sort_id FROM json_each(:display_rows)
    )""", {"display_rows": json.dumps(rows)}


def read_display_segments(db, segments, *, limit, offset, latest):
    """Bounded payload page of a verified local conversation's display history."""
    direction = "DESC" if latest else "ASC"
    with _snapshot(db) as conn:
        _register_functions(db, conn)
        sql, params = _conversation_display_sql(db, conn, segments[-1], segments)
        rows = conn.execute(sql + f"""
            SELECT m.* FROM (SELECT row_id, sort_id FROM display_rows ORDER BY sort_id {direction}
                            LIMIT :limit OFFSET :offset) AS page
            JOIN messages m ON m.id = page.row_id ORDER BY page.sort_id
        """, dict(params, limit=-1 if limit is None else limit, offset=offset)).fetchall()
    return [_row_to_message_dict(db, row, warn_context="local display history", summary_flag=True) for row in rows]


def _register_functions(db, conn):
    from session_rules.turns import split_user_originated_turn

    def identity_content(content, display_kind):
        handoff, live = split_user_originated_turn({
            "role": "user", "content": _decode_content(content), "display_kind": display_kind})
        return _encode_content(live.get("content")) if handoff is not None and live is not None else content

    conn.create_function("timeline_identity_content", 2, identity_content, deterministic=True)
    conn.create_function("timeline_preview", 3,
                         lambda content, kind, summary: _prompt_preview(db, content, kind, summary),
                         deterministic=True)


def get_session_messages_around(db, session_id, row_id, *, limit=120, segments=()):
    """Read at most *limit* display rows starting at an exact human prompt.

    Existence probes/counts contain ids only. Full payloads are fetched only for
    the selected bounded page, even when the anchor is deep in a transcript.
    """
    with _snapshot(db) as conn:
        _register_functions(db, conn)
        anchor = conn.execute(
            "SELECT content, display_kind, _compressed_summary FROM messages "
            "WHERE session_id IN (SELECT value FROM json_each(?)) "
            "AND id = ? AND role = 'user' AND (active = 1 OR compacted = 1)",
            (json.dumps(segments or (session_id,)), row_id),
        ).fetchone()
        if anchor is None or not _prompt_preview(db, *anchor):
            return None
        sql, params = _conversation_display_sql(db, conn, session_id, segments)
        params.update(row_id=row_id, limit=limit)
        selected = conn.execute(sql + """
            SELECT sort_id FROM display_rows WHERE row_id = :row_id
        """, params).fetchone()
        if selected is None:
            return None
        params["start"] = selected["sort_id"]
        counts = conn.execute(sql + """
            SELECT COUNT(*) AS total, COALESCE(SUM(sort_id < :start), 0) AS offset FROM display_rows
        """, params).fetchone()
        rows = conn.execute(sql + """
            SELECT m.* FROM (SELECT row_id, sort_id FROM display_rows
                            WHERE sort_id >= :start ORDER BY sort_id LIMIT :limit) AS page
            JOIN messages m ON m.id = page.row_id ORDER BY page.sort_id
        """, params).fetchall()
    messages = [_row_to_message_dict(db, row, warn_context="timeline jump", summary_flag=True) for row in rows]
    return {"messages": messages, "pagination": {
        "row_id": row_id, "limit": limit, "returned": len(messages), "order": "oldest",
        "offset": counts["offset"], "total": counts["total"],
        "has_older": counts["offset"] > 0,
        "has_newer": counts["offset"] + len(messages) < counts["total"],
    }}


def get_session_timeline(db, session_id, *, limit=500, after_row_id=0, segments=()):
    """Chronological prompts. Cursor is the first physical row id of a logical turn."""
    with _snapshot(db) as conn:
        _register_functions(db, conn)
        sql, params = _conversation_display_sql(db, conn, session_id, segments, users_only=True)
        sql += """,
            prompts AS MATERIALIZED (
                SELECT row_id, sort_id, m.timestamp,
                       timeline_preview(m.content, m.display_kind, m._compressed_summary) AS preview
                FROM display_rows JOIN messages m ON m.id = row_id
            ), eligible AS MATERIALIZED (SELECT * FROM prompts WHERE preview <> '')
        """
        params.update(after=after_row_id, limit=limit + 1)
        rows = conn.execute(sql + """
            SELECT row_id, sort_id, timestamp, preview, (SELECT COUNT(*) FROM eligible) AS total
            FROM eligible WHERE sort_id > :after ORDER BY sort_id LIMIT :limit
        """, params).fetchall()
        total = rows[0]["total"] if rows else conn.execute(
            sql + "SELECT COUNT(*) FROM eligible", params).fetchone()[0]
    has_more = len(rows) > limit
    page = rows[:limit]
    return {
        "entries": [{"row_id": row["row_id"], "preview": row["preview"], "timestamp": row["timestamp"]}
                    for row in page],
        "pagination": {"limit": limit, "after_row_id": after_row_id, "returned": len(page),
                       "total": total, "has_more": has_more,
                       "next_cursor": page[-1]["sort_id"] if has_more else None},
    }
