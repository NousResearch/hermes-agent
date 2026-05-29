"""Project usage ledger for dashboard-level board/task cost visibility.

The ledger is a derived, idempotently backfilled SQLite database under the
active Hermes home.  It correlates:

* session usage stored in ``state.db`` (tokens/cost/model/provider), and
* Kanban board/task/run metadata stored in each board DB.

The source databases remain authoritative; this module only materializes a
query-friendly view for dashboards and future reporting jobs.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home
from hermes_state import apply_wal_with_fallback
from hermes_cli import kanban_db

log = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_entries (
    source_type          TEXT NOT NULL,
    source_id            TEXT NOT NULL,
    board_slug           TEXT,
    board_name           TEXT,
    task_id              TEXT,
    task_title           TEXT,
    task_status          TEXT,
    run_id               INTEGER,
    run_status           TEXT,
    run_outcome          TEXT,
    session_id           TEXT,
    session_title        TEXT,
    session_source       TEXT,
    user_id              TEXT,
    model                TEXT,
    billing_provider     TEXT,
    billing_mode         TEXT,
    input_tokens         INTEGER NOT NULL DEFAULT 0,
    output_tokens        INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens   INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens     INTEGER NOT NULL DEFAULT 0,
    api_call_count       INTEGER NOT NULL DEFAULT 0,
    tool_call_count      INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd   REAL,
    actual_cost_usd      REAL,
    cost_status          TEXT,
    started_at           REAL,
    ended_at             REAL,
    metadata             TEXT,
    backfilled_at        REAL NOT NULL,
    PRIMARY KEY (source_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_usage_board ON usage_entries(board_slug);
CREATE INDEX IF NOT EXISTS idx_usage_task ON usage_entries(task_id);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_entries(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_started ON usage_entries(started_at DESC);

CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def ledger_path() -> Path:
    """Return ``<HERMES_HOME>/usage/project_usage.db``."""
    return get_hermes_home() / "usage" / "project_usage.db"


def state_db_path() -> Path:
    return get_hermes_home() / "state.db"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    p = path or ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    apply_wal_with_fallback(conn, db_label=f"project_usage.db ({p.name})")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    existing = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if existing is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (LEDGER_SCHEMA_VERSION,))
    else:
        conn.execute("UPDATE schema_version SET version = ?", (LEDGER_SCHEMA_VERSION,))
    return conn


@contextlib.contextmanager
def _ledger_write_txn(conn: sqlite3.Connection):
    """Batch ledger writes in one explicit transaction.

    Ledger connections are intentionally opened in autocommit mode so simple
    dashboard operations stay straightforward. Backfill is different: it can
    perform hundreds of `_upsert_entry` calls, and without an explicit
    transaction each one commits independently. Reuse a caller's active
    transaction if present; otherwise wrap the batch in BEGIN IMMEDIATE.
    """
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _json_loads(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_state_session(conn: sqlite3.Connection, session_id: Optional[str]) -> Optional[sqlite3.Row]:
    if not session_id:
        return None
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


def _session_values(row: Optional[sqlite3.Row]) -> dict[str, Any]:
    if row is None:
        return {
            "session_id": None,
            "session_title": None,
            "session_source": None,
            "user_id": None,
            "model": None,
            "billing_provider": None,
            "billing_mode": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "api_call_count": 0,
            "tool_call_count": 0,
            "estimated_cost_usd": None,
            "actual_cost_usd": None,
            "cost_status": None,
            "started_at": None,
            "ended_at": None,
        }
    return {
        "session_id": row["id"],
        "session_title": row["title"],
        "session_source": row["source"],
        "user_id": row["user_id"],
        "model": row["model"],
        "billing_provider": row["billing_provider"],
        "billing_mode": row["billing_mode"],
        "input_tokens": int(row["input_tokens"] or 0),
        "output_tokens": int(row["output_tokens"] or 0),
        "cache_read_tokens": int(row["cache_read_tokens"] or 0),
        "cache_write_tokens": int(row["cache_write_tokens"] or 0),
        "reasoning_tokens": int(row["reasoning_tokens"] or 0),
        "api_call_count": int(row["api_call_count"] or 0),
        "tool_call_count": int(row["tool_call_count"] or 0),
        "estimated_cost_usd": row["estimated_cost_usd"],
        "actual_cost_usd": row["actual_cost_usd"],
        "cost_status": row["cost_status"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
    }


def _zero_usage_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return session identity/timing metadata with additive usage fields zeroed.

    A single Hermes worker session can appear on multiple task_run rows when a
    task retries or records multiple terminal runs. The source session totals are
    cumulative for the whole session, so summing them once per run inflates board
    and project totals. Keep the duplicate run visible for audit/drilldown while
    counting the session's additive token/cost fields only once.
    """
    out = dict(values)
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "api_call_count",
        "tool_call_count",
    ):
        out[key] = 0
    out["estimated_cost_usd"] = 0.0
    out["actual_cost_usd"] = 0.0
    return out


def _upsert_entry(conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
    cols = [
        "source_type", "source_id", "board_slug", "board_name", "task_id",
        "task_title", "task_status", "run_id", "run_status", "run_outcome",
        "session_id", "session_title", "session_source", "user_id", "model",
        "billing_provider", "billing_mode", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "api_call_count", "tool_call_count", "estimated_cost_usd", "actual_cost_usd",
        "cost_status", "started_at", "ended_at", "metadata", "backfilled_at",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c not in {"source_type", "source_id"}])
    conn.execute(
        f"INSERT INTO usage_entries ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(source_type, source_id) DO UPDATE SET {updates}",
        tuple(entry.get(c) for c in cols),
    )


def backfill(*, ledger_conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
    """Backfill the ledger from current ``state.db`` and all Kanban boards.

    Returns a compact summary with the number of session-only and task-run
    entries materialized.  The operation is safe to call repeatedly.
    """
    owns_conn = ledger_conn is None
    lconn = ledger_conn or connect()
    now = time.time()
    state_path = state_db_path()
    session_entries = 0
    run_entries = 0
    linked_sessions: set[str] = set()
    counted_task_session_usage: set[tuple[str, str]] = set()
    board_errors: list[str] = []

    if not state_path.exists():
        with _ledger_write_txn(lconn):
            lconn.execute("INSERT OR REPLACE INTO ledger_meta(key, value) VALUES ('last_backfill_error', ?)", (f"missing state db: {state_path}",))
        if owns_conn:
            lconn.close()
        return {"session_entries": 0, "run_entries": 0, "boards": 0, "ledger_path": str(ledger_path())}

    sconn: Optional[sqlite3.Connection] = None
    try:
        sconn = sqlite3.connect(str(state_path), timeout=30)
        sconn.row_factory = sqlite3.Row
        with _ledger_write_txn(lconn):
            # First materialize all board/task/run usage entries.  A run can link
            # to a worker session via task_runs.metadata.worker_session_id or to an
            # originating session via tasks.session_id.
            boards = kanban_db.list_boards(include_archived=False)
            for board in boards:
                slug = board.get("slug") or kanban_db.DEFAULT_BOARD
                try:
                    kanban_db.init_db(board=slug)
                    kconn = kanban_db.connect(board=slug)
                except Exception as exc:
                    msg = f"{slug}: {exc}"
                    board_errors.append(msg)
                    log.warning("project usage backfill skipped board %s: %s", slug, exc)
                    continue
                try:
                    rows = kconn.execute(
                        """
                        SELECT
                            r.id AS run_id,
                            r.task_id AS task_id,
                            r.status AS run_status,
                            r.outcome AS run_outcome,
                            r.started_at AS run_started_at,
                            r.ended_at AS run_ended_at,
                            r.metadata AS run_metadata,
                            t.title AS task_title,
                            t.status AS task_status,
                            t.session_id AS task_session_id
                        FROM task_runs r
                        LEFT JOIN tasks t ON t.id = r.task_id
                        ORDER BY r.id
                        """
                    ).fetchall()
                    for row in rows:
                        meta = _json_loads(row["run_metadata"])
                        session_id = (
                            meta.get("worker_session_id")
                            or meta.get("session_id")
                            or row["task_session_id"]
                        )
                        if session_id:
                            linked_sessions.add(str(session_id))
                        session_row = _get_state_session(sconn, session_id)
                        values = _session_values(session_row)
                        if session_id:
                            session_key = (str(row["task_id"]), str(session_id))
                            if session_key in counted_task_session_usage:
                                values = _zero_usage_values(values)
                            else:
                                counted_task_session_usage.add(session_key)
                        _upsert_entry(lconn, {
                            **values,
                            "source_type": "task_run",
                            "source_id": f"{slug}:{row['run_id']}",
                            "board_slug": slug,
                            "board_name": board.get("name") or slug,
                            "task_id": row["task_id"],
                            "task_title": row["task_title"],
                            "task_status": row["task_status"],
                            "run_id": row["run_id"],
                            "run_status": row["run_status"],
                            "run_outcome": row["run_outcome"],
                            "started_at": values.get("started_at") or row["run_started_at"],
                            "ended_at": values.get("ended_at") or row["run_ended_at"],
                            "metadata": json.dumps(meta, sort_keys=True) if meta else None,
                            "backfilled_at": now,
                        })
                        run_entries += 1
                except Exception as exc:
                    msg = f"{slug}: {exc}"
                    board_errors.append(msg)
                    log.warning("project usage backfill skipped board %s: %s", slug, exc)
                finally:
                    kconn.close()

            # Remove stale unassigned rows for sessions that are now attributable to
            # board task_runs. Backfill is idempotent and sessions can become linked
            # after an earlier pass materialized them as source_type='session'.
            if linked_sessions:
                placeholders = ", ".join(["?"] * len(linked_sessions))
                lconn.execute(
                    f"DELETE FROM usage_entries WHERE source_type = 'session' AND source_id IN ({placeholders})",
                    tuple(linked_sessions),
                )

            # Also materialize raw sessions that were not attributable to a board.
            for row in sconn.execute("SELECT * FROM sessions ORDER BY started_at"):
                if row["id"] in linked_sessions:
                    continue
                values = _session_values(row)
                _upsert_entry(lconn, {
                    **values,
                    "source_type": "session",
                    "source_id": row["id"],
                    "board_slug": None,
                    "board_name": None,
                    "task_id": None,
                    "task_title": None,
                    "task_status": None,
                    "run_id": None,
                    "run_status": None,
                    "run_outcome": None,
                    "metadata": None,
                    "backfilled_at": now,
                })
                session_entries += 1

            lconn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES ('last_backfill_at', ?)",
                (str(now),),
            )
            if board_errors:
                lconn.execute(
                    "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES ('last_backfill_error', ?)",
                    ("; ".join(board_errors),),
                )
            else:
                lconn.execute("DELETE FROM ledger_meta WHERE key = 'last_backfill_error'")
            return {
                "session_entries": session_entries,
                "run_entries": run_entries,
                "boards": len(boards),
                "ledger_path": str(ledger_path()),
            }
    finally:
        if sconn is not None:
            sconn.close()
        if owns_conn:
            lconn.close()



def session_usage_snapshot(session_id: Optional[str]) -> dict[str, Any]:
    """Return a compact token/cost snapshot for a session id from state.db.

    Intended for embedding into Kanban task_run metadata at terminal
    transitions.  Missing state/session rows return only ``session_id`` so the
    ledger can still join/backfill once the session row appears.
    """
    if not session_id:
        return {}
    snapshot: dict[str, Any] = {"session_id": str(session_id)}
    p = state_db_path()
    if not p.exists():
        return snapshot
    conn = sqlite3.connect(str(p), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        row = _get_state_session(conn, str(session_id))
        if row is None:
            return snapshot
        vals = _session_values(row)
        for key in (
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "reasoning_tokens", "api_call_count",
            "tool_call_count", "estimated_cost_usd", "actual_cost_usd",
            "cost_status", "model", "billing_provider", "billing_mode",
            "started_at", "ended_at",
        ):
            snapshot[key] = vals.get(key)
        return snapshot
    finally:
        conn.close()


def stamp_usage_metadata(metadata: Optional[dict[str, Any]], session_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Return metadata with worker_session_id + usage_snapshot merged in."""
    if not session_id:
        return metadata
    out = dict(metadata or {})
    # Trusted dispatcher/tool metadata wins over any user-supplied/spoofed value.
    out["worker_session_id"] = str(session_id)
    out["usage_snapshot"] = session_usage_snapshot(str(session_id))
    return out


def stamp_worker_usage_metadata(
    task_id: str,
    metadata: Optional[dict[str, Any]],
    *,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Optional[dict[str, Any]]:
    """Best-effort usage metadata stamping for the active worker task.

    Both the human/CLI Kanban path and native Kanban tool path close task runs.
    Keep task completion/blocking reliable even when usage state is absent,
    locked, or corrupt by falling back to the trusted worker_session_id only.
    """
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return metadata
    session_id = os.environ.get("HERMES_SESSION_ID")
    if not session_id:
        return metadata
    try:
        return stamp_usage_metadata(metadata, session_id)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        fallback = dict(metadata or {})
        fallback["worker_session_id"] = str(session_id)
        return fallback


def _totals_where(board: Optional[str] = None, task_id: Optional[str] = None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if board:
        clauses.append("board_slug = ?")
        params.append(board)
    if task_id:
        clauses.append("task_id = ?")
        params.append(task_id)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def get_task_rollups(
    *,
    board: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    refresh: bool = True,
) -> list[dict[str, Any]]:
    """Return usage rollups for specific task ids without the dashboard top-N cap."""
    if not task_ids:
        return []
    conn = connect()
    try:
        if refresh:
            backfill(ledger_conn=conn)
        placeholders = ", ".join(["?"] * len(task_ids))
        where = f"WHERE task_id IN ({placeholders})"
        params: list[Any] = list(task_ids)
        if board:
            where += " AND board_slug = ?"
            params.append(board)
        rows = conn.execute(
            f"""
            SELECT
                board_slug,
                board_name,
                task_id,
                MAX(task_title) AS task_title,
                MAX(task_status) AS task_status,
                COUNT(*) AS runs,
                COUNT(DISTINCT session_id) AS sessions,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
                COALESCE(SUM(actual_cost_usd), 0.0) AS actual_cost_usd,
                MIN(started_at) AS first_started_at,
                MAX(ended_at) AS last_ended_at
            FROM usage_entries
            {where}
            GROUP BY board_slug, board_name, task_id
            ORDER BY estimated_cost_usd DESC, input_tokens + output_tokens DESC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_summary(*, board: Optional[str] = None, task_id: Optional[str] = None, refresh: bool = True) -> dict[str, Any]:
    """Return dashboard-ready per-board totals and task drilldown rows."""
    conn = connect()
    try:
        if refresh:
            backfill(ledger_conn=conn)
        where, params = _totals_where(board, task_id)
        totals = conn.execute(
            f"""
            SELECT
                COUNT(*) AS entries,
                COUNT(DISTINCT session_id) AS sessions,
                COUNT(DISTINCT task_id) AS tasks,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(api_call_count), 0) AS api_call_count,
                COALESCE(SUM(tool_call_count), 0) AS tool_call_count,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
                COALESCE(SUM(actual_cost_usd), 0.0) AS actual_cost_usd
            FROM usage_entries
            {where}
            """,
            params,
        ).fetchone()
        board_where = "WHERE board_slug = ?" if board else ""
        board_params: list[Any] = [board] if board else []
        board_rows = conn.execute(
            f"""
            SELECT
                COALESCE(board_slug, '__unassigned__') AS board_slug,
                COALESCE(board_name, 'Unassigned sessions') AS board_name,
                COUNT(*) AS entries,
                COUNT(DISTINCT task_id) AS tasks,
                COUNT(DISTINCT session_id) AS sessions,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
                COALESCE(SUM(actual_cost_usd), 0.0) AS actual_cost_usd,
                MAX(started_at) AS latest_started_at
            FROM usage_entries
            {board_where}
            GROUP BY COALESCE(board_slug, '__unassigned__'), COALESCE(board_name, 'Unassigned sessions')
            ORDER BY estimated_cost_usd DESC, input_tokens + output_tokens DESC
            """,
            board_params,
        ).fetchall()
        task_where = "WHERE task_id IS NOT NULL"
        task_params: list[Any] = []
        if board:
            task_where += " AND board_slug = ?"
            task_params.append(board)
        if task_id:
            task_where += " AND task_id = ?"
            task_params.append(task_id)
        task_rows = conn.execute(
            f"""
            SELECT
                board_slug,
                board_name,
                task_id,
                MAX(task_title) AS task_title,
                MAX(task_status) AS task_status,
                COUNT(*) AS runs,
                COUNT(DISTINCT session_id) AS sessions,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
                COALESCE(SUM(actual_cost_usd), 0.0) AS actual_cost_usd,
                MIN(started_at) AS first_started_at,
                MAX(ended_at) AS last_ended_at
            FROM usage_entries
            {task_where}
            GROUP BY board_slug, board_name, task_id
            ORDER BY estimated_cost_usd DESC, input_tokens + output_tokens DESC
            LIMIT 500
            """,
            task_params,
        ).fetchall()
        run_rows: list[sqlite3.Row] = []
        if task_id:
            run_where = "task_id = ?"
            run_params: list[Any] = [task_id]
            if board:
                run_where += " AND board_slug = ?"
                run_params.append(board)
            run_rows = conn.execute(
                f"""
                SELECT * FROM usage_entries
                WHERE {run_where}
                ORDER BY COALESCE(started_at, 0) DESC, run_id DESC
                LIMIT 200
                """,
                run_params,
            ).fetchall()
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM ledger_meta")}
        return {
            "ledger_path": str(ledger_path()),
            "last_backfill_at": float(meta["last_backfill_at"]) if meta.get("last_backfill_at") else None,
            "last_backfill_error": meta.get("last_backfill_error"),
            "totals": dict(totals) if totals else {},
            "boards": [dict(r) for r in board_rows],
            "tasks": [dict(r) for r in task_rows],
            "runs": [dict(r) for r in run_rows],
        }
    finally:
        conn.close()
