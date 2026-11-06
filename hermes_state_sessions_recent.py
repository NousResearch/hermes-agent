"""Latency-bounded recent-conversation browse (``session_search()``): body of
:meth:`hermes_state_sessions.SessionSessionsMixin.list_recent_sessions_bounded`."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from hermes_state_common import (
    _LISTABLE_CHILD_SQL,
    _shape_preview,
    _sql_preview_raw,
    _sql_session_last_active_by_id,
)


def list_recent_sessions_bounded(
    db,
    *,
    limit: int = 20,
    exclude_sources: list[str] | None = None,
    timeout_seconds: float = 3.0,
    candidate_limit: int | None = None,
    lineage_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Latency-bounded recent-conversation browse (``session_search()``): preselect a small candidate set
    from the indexed durable activity timestamp (fallback ``started_at``), resolve only those across
    compression ancestry/chains, then hydrate activity/previews for that bounded set. Lineage traversal
    uses ``UNION`` plus a total-row ceiling so a corrupt cycle or a deep/branching lineage cannot defeat
    the bound; a lineage that hits the ceiling before a terminal root/tip is omitted, not expanded.
    A cooperative SQLite progress deadline interrupts sustained work past ``timeout_seconds`` and raises
    ``TimeoutError`` (cheap statements may finish between callbacks). Supports only the agent-tool
    browse filters; rich callers keep using
    :meth:`hermes_state_sessions.SessionSessionsMixin.list_sessions_rich`."""
    from hermes_state_sessions import _delegate_from_json

    limit = max(1, int(limit))
    timeout_seconds = max(0.0, float(timeout_seconds))
    if candidate_limit is None:
        candidate_limit = max(128, limit * 8)
    candidate_limit = max(limit, min(int(candidate_limit), 2048))
    if lineage_limit is None:
        lineage_limit = min(8192, candidate_limit * 8)
    lineage_limit = max(candidate_limit, min(int(lineage_limit), 8192))

    candidate_clauses = [
        "s.archived = 0",
        "s.hidden = 0",
        f"{_delegate_from_json('s.model_config')} IS NULL",
    ]
    candidate_params: list[Any] = []
    if exclude_sources:
        placeholders = ",".join("?" for _ in exclude_sources)
        candidate_clauses.append(f"s.source NOT IN ({placeholders})")
        candidate_params.extend(exclude_sources)
    candidate_where = " AND ".join(candidate_clauses)

    # A compression continuation is an implementation edge, unlike /new
    # reset and /branch children which are independent user-visible
    # conversations.  The same predicate is used in both directions so a
    # candidate tip maps to its logical root and the root maps back to the
    # freshest live tip.
    compression_parent_edge = f"""
        parent.end_reason = 'compression'
        AND child.parent_session_id = parent.id
        AND json_extract(
            COALESCE(child.model_config, '{{}}'), '$._branched_from'
        ) IS NULL
        AND {_delegate_from_json("child.model_config")} IS NULL
        AND COALESCE(child.source, '') != 'tool'
    """

    query = f"""
        WITH RECURSIVE
        recent_candidates(id) AS (
            SELECT s.id
            FROM sessions s
            WHERE {candidate_where}
            ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC,
                     s.started_at DESC, s.id DESC
            LIMIT ?
        ),
        ancestors(candidate_id, cur_id) AS (
            SELECT id, id FROM recent_candidates
            UNION
            SELECT a.candidate_id, parent.id
            FROM ancestors a
            JOIN sessions child ON child.id = a.cur_id
            JOIN sessions parent ON {compression_parent_edge}
            LIMIT ?
        ),
        candidate_roots(root_id) AS (
            SELECT DISTINCT a.cur_id
            FROM ancestors a
            JOIN sessions child ON child.id = a.cur_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM sessions parent
                WHERE {compression_parent_edge}
            )
        ),
        chain(root_id, cur_id) AS (
            SELECT root_id, root_id FROM candidate_roots
            UNION
            SELECT c.root_id, child.id
            FROM chain c
            JOIN sessions parent ON parent.id = c.cur_id
            JOIN sessions child ON {compression_parent_edge}
            LIMIT ?
        ),
        chain_rows AS (
            SELECT
                c.root_id,
                c.cur_id,
                {_sql_session_last_active_by_id("c.cur_id")} AS activity,
                CASE WHEN EXISTS (
                    SELECT 1
                    FROM sessions parent
                    JOIN sessions child ON {compression_parent_edge}
                    WHERE parent.id = c.cur_id
                ) THEN 0 ELSE 1 END AS is_tip
            FROM chain c
        ),
        ranked_tips AS (
            SELECT root_id, cur_id, activity,
                   ROW_NUMBER() OVER (
                       PARTITION BY root_id
                       ORDER BY activity DESC, cur_id DESC
                   ) AS rank_in_root
            FROM chain_rows
            WHERE is_tip = 1
        )
        SELECT
            tip.id,
            tip.source,
            tip.model,
            COALESCE(tip.title, s.title) AS title,
            s.started_at AS started_at,
            tip.ended_at,
            tip.end_reason,
            tip.message_count,
            tip.tool_call_count,
            rt.activity AS last_active,
            {_sql_preview_raw("tip.id")},
            CASE WHEN s.id != tip.id THEN s.id ELSE NULL END
                AS _lineage_root_id
        FROM ranked_tips rt
        JOIN sessions s ON s.id = rt.root_id
        JOIN sessions tip ON tip.id = rt.cur_id
        WHERE rt.rank_in_root = 1
          AND s.archived = 0
          AND s.hidden = 0
          AND {_LISTABLE_CHILD_SQL}
          AND {_delegate_from_json("s.model_config")} IS NULL
        ORDER BY rt.activity DESC, s.started_at DESC, tip.id DESC
        LIMIT ?
    """
    params = candidate_params + [
        candidate_limit,
        lineage_limit,
        lineage_limit,
        limit,
    ]
    deadline = time.monotonic() + timeout_seconds
    interrupted_by_deadline = False

    def _deadline_progress_handler() -> int:
        nonlocal interrupted_by_deadline
        if time.monotonic() >= deadline:
            interrupted_by_deadline = True
            return 1
        return 0

    try:
        with db._read_ctx() as conn:
            conn.set_progress_handler(_deadline_progress_handler, 1000)
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.set_progress_handler(None, 0)
    except sqlite3.OperationalError as exc:
        if interrupted_by_deadline and "interrupt" in str(exc).lower():
            raise TimeoutError(
                f"recent-session browse exceeded {timeout_seconds:g}s deadline"
            ) from exc
        raise

    sessions = []
    for row in rows:
        session = db._session_row_dict(row)
        session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
        session["unread"] = db.session_unread(session)
        sessions.append(session)
    return sessions
