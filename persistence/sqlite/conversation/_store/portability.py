"""Rich session rows, cron runs, skill-scaffolded sessions and working directories."""


from typing import Any, Optional
from hermes_state_keys import (
    CRON_PRINCIPAL_ID, CRON_REQUEST_PREFIX, cron_request_range, cron_run_from_request, legacy_cron_run_job, prefix_range,
)
from hermes_state_common import _shape_preview
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema import _parse_schema_columns
from persistence.sqlite._schema_ddl import SCHEMA_SQL
from persistence.sqlite.conversation._store.core import _session_row_dict
from persistence.sqlite.conversation._store.sql_common import _sql_preview_raw, _sql_session_last_active
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.rules import (
    AUTO_LOAD_SCAFFOLD_SQL_LIKE, SKILL_SCAFFOLD_SQL_LIKE,
)


def _rich_select(select_cols: str, where: str, tail: str = "", prompt_select: Optional[str] = "") -> str:
    """``list_sessions_rich``-shaped SELECT: resolved prompt (``prompt_select`` fragment;
    None omits prompt columns AND the join), preview, last_active. Whitespace matches
    the historical inline queries (SQL text is pinned)."""
    prompt_join = "" if prompt_select is None else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
    return f"""
            SELECT {select_cols}{prompt_select or ""},
                {_sql_preview_raw()},
                {_sql_session_last_active("s")} AS last_active
            FROM sessions s
            {prompt_join}
            WHERE {where}{tail}
        """


_PROMPT_RESOLVED_SQL = "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"


_CRON_POLICY_JOB_SQL = (
    "(SELECT CASE WHEN json_valid(r) THEN json_extract(r, '$.cron_job.id') END FROM"
    " (SELECT CASE WHEN json_valid(value) THEN json_extract(value, '$.policy.request_json') END AS r"
    f" FROM state_meta WHERE key = '{POLICY_PREFIX}' || target_session_id))")


_CRON_ADMISSION_SQL = (
    f"principal_id = '{CRON_PRINCIPAL_ID}' AND request_id >= :request_lo AND request_id < :request_hi"
    f" AND ((instr(substr(request_id, {len(CRON_REQUEST_PREFIX) + 1}), ':') = length(:request_lo) - {len(CRON_REQUEST_PREFIX)}"
    " AND instr(substr(request_id, length(:request_lo) + 1), ':') = 0)"
    f" OR {_CRON_POLICY_JOB_SQL} = :job_id)")


_CRON_RUN_IDS_SQL = (
    "SELECT id FROM sessions WHERE source = 'cron' AND id >= :prefix AND id < :prefix_hi"
    " AND substr(id, length(:prefix) + 1) GLOB '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]'"
    f" UNION ALL SELECT target_session_id FROM session_admissions WHERE {_CRON_ADMISSION_SQL}")


_CRON_EXECUTION_SQL = (
    "(SELECT substr(request_id, length(:request_lo) + 1) FROM session_admissions"
    f" WHERE target_session_id = s.id AND {_CRON_ADMISSION_SQL} LIMIT 1)")


def _compact_session_cols(home) -> str:
    """``s.``-prefixed SELECT list of every SCHEMA_SQL ``sessions`` column except
    prompt storage internals (the compact_rows projection)."""
    from persistence.sqlite.conversation._store.sessions import _SESSION_COMPACT_EXCLUDED
    global _session_compact_cols_sql
    if _session_compact_cols_sql is None:
        declared = _parse_schema_columns(SCHEMA_SQL, home)["sessions"]
        _session_compact_cols_sql = ", ".join(
            f"s.{name}" for name in declared if name not in _SESSION_COMPACT_EXCLUDED
        )
    return _session_compact_cols_sql


def _rich_row(row) -> dict[str, Any]:
    """Session row dict with ``_preview_raw`` shaped into ``preview``."""
    s = _session_row_dict(row)
    s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
    return s


def _read_rows(holder: StoreHolder, sql: str, params=()) -> list:
    """Pure-read query via ``_read_ctx()`` (never the writer lock: turn persistence must not convoy)."""
    with holder._read_ctx() as conn:
        return conn.execute(sql, params).fetchall()


def distinct_session_cwds(holder: StoreHolder, include_archived: bool = False) -> list[dict[str, Any]]:
    """Distinct non-empty session cwds with usage stats, for repo discovery. Aggregates
    across ALL history; children/branches count (a worktree session is a real
    workspace signal)."""
    where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
    if not include_archived:
        where += " AND archived = 0"
    rows = _read_rows(holder,
        "SELECT cwd AS cwd, COUNT(*) AS sessions, MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
        f"FROM sessions WHERE {where} GROUP BY cwd"
    )
    return [{"cwd": r["cwd"], "sessions": int(r["sessions"] or 0), "last_active": float(r["last_active"] or 0)}
            for r in rows]


def list_cron_job_runs(holder: StoreHolder, job_id: str, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """Run sessions of one cron job, newest first, in the ``list_sessions_rich`` row shape,
    stamped with ``cron_job_id`` and ``cron_execution_id`` (None for a pre-unified run).

    Pre-unified runs are flat ``cron_{job_id}_{timestamp}`` sessions: a ``[prefix,
    prefix_hi)`` id range. Unified-runtime runs keep their runtime-minted ``local-*`` id, which
    must not change (exact-retry recovery re-derives it), but the scheduler principal admitted
    each under ``cron:<job>:<execution>``: a range of ``idx_session_admissions_cron_request``.
    Both scans scale with this job's runs, never with total cron history or with every local
    session's policy receipt, and skip ``list_sessions_rich``'s compression-chain CTE.

    Limitation (legacy runs had it too): only a run's FIRST session is listed. A run that
    compressed continues in a child session this does not return, and the row's liveness
    (``scheduler_owned``) and detail identity follow that first session, not the tip."""
    prefix, prefix_hi = prefix_range(f"cron_{job_id}_")
    request_lo, request_hi = cron_request_range(job_id)
    query = _rich_select(
        f"s.*, {_CRON_EXECUTION_SQL} AS cron_execution_id", f"s.source = 'cron' AND s.id IN ({_CRON_RUN_IDS_SQL})",
        "\n            ORDER BY s.started_at DESC, s.id DESC\n            LIMIT :limit OFFSET :offset",
        prompt_select=f",\n                {_PROMPT_RESOLVED_SQL}",
    )
    rows = _read_rows(holder, query, {"prefix": prefix, "prefix_hi": prefix_hi, "request_lo": request_lo,
                                   "request_hi": request_hi, "job_id": job_id, "limit": limit, "offset": offset})
    return [{**_rich_row(row), "cron_job_id": job_id} for row in rows]


def cron_run_identity(holder: StoreHolder, session_id: str) -> Optional[dict[str, Any]]:
    """``{"cron_job_id", "cron_execution_id"}`` of a cron run session, else None: a legacy id
    names its job (no execution), a unified-runtime run is one probe of its scheduler admission
    (and, when its request id is ambiguous, of the job its creation receipt's policy names)."""
    legacy_job = legacy_cron_run_job(session_id)
    if legacy_job is not None:
        return {"cron_job_id": legacy_job, "cron_execution_id": None}
    rows = _read_rows(holder,
        "SELECT request_id, " + _CRON_POLICY_JOB_SQL + " AS policy_job FROM session_admissions"
        " WHERE principal_id = ? AND target_session_id = ? ORDER BY seq LIMIT 1", (CRON_PRINCIPAL_ID, session_id))
    run = None
    if rows:
        run = cron_run_from_request(rows[0]["request_id"])
        if run is None and isinstance(rows[0]["policy_job"], str):
            run = cron_run_from_request(rows[0]["request_id"], rows[0]["policy_job"])
    return {"cron_job_id": run[0], "cron_execution_id": run[1]} if run else None


def _get_session_rich_row(holder: StoreHolder, session_id: str, compact_rows: bool = False) -> Optional[dict[str, Any]]:
    """One session with the ``list_sessions_rich`` enriched columns, or None.
    ``compact_rows=True`` omits the ``system_prompt`` blob. Public alias:
    :meth:`get_session_rich_row` (web server hydration)."""
    return _get_session_rich_rows_batch(holder, [session_id], compact_rows=compact_rows).get(session_id)


get_session_rich_row = _get_session_rich_row


def _get_session_rich_rows_batch(holder: StoreHolder, session_ids, compact_rows: bool = False) -> dict[str, dict[str, Any]]:
    """Enriched rows for many sessions in one query, keyed by id; missing ids are absent
    (a page of compression tips resolves in one round trip)."""
    ids = [sid for sid in session_ids if sid]
    if not ids:
        return {}
    # Old SQLite caps bound variables at 999 (SQLITE_MAX_VARIABLE_NUMBER); limit=10000
    # callers exist. Chunk here — the single choke point.
    _CHUNK = 900
    if len(ids) > _CHUNK:
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), _CHUNK):
            result.update(_get_session_rich_rows_batch(holder, ids[start:start + _CHUNK], compact_rows=compact_rows))
        return result
    # Same read-your-writes guarantee as list_sessions_rich.
    holder.flush_token_counts()
    query = _rich_select(
        _compact_session_cols(holder.db_path.parent) if compact_rows else "s.*", f"s.id IN ({','.join('?' for _ in ids)})",
        prompt_select=None if compact_rows else f", {_PROMPT_RESOLVED_SQL}",
    )
    return {s["id"]: s for s in map(_rich_row, _read_rows(holder, query, ids))}


def list_skill_scaffolded_sessions(holder: StoreHolder, limit: int = 200) -> list[dict[str, Any]]:
    """Titled sessions whose first user turn was a ``/skill`` invocation or a gateway
    auto-load scaffold (their titles describe the expanded skill body, not the
    request). Returns ``id``, ``title`` and the first-turn ``content`` so callers can
    re-derive what was typed. Newest first."""
    rows = _read_rows(holder, """
                SELECT s.id, s.title, m.content
                FROM sessions s
                JOIN messages m ON m.id = (
                    SELECT m2.id FROM messages m2
                    WHERE m2.session_id = s.id AND m2.role = 'user'
                      AND m2.content IS NOT NULL
                    ORDER BY m2.timestamp, m2.id LIMIT 1
                )
                WHERE s.title IS NOT NULL AND (m.content LIKE ? OR m.content LIKE ?)
                ORDER BY s.started_at DESC
                LIMIT ?
                """, (SKILL_SCAFFOLD_SQL_LIKE, AUTO_LOAD_SCAFFOLD_SQL_LIKE, int(limit)))
    return [dict(row) for row in rows]


_SQLITE_INT_MIN = -(1 << 63)


_SQLITE_INT_MAX = (1 << 63) - 1


_session_compact_cols_sql: Optional[str] = None
