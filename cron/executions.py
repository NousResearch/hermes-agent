"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

EXECUTIONS_FILE = get_hermes_home().resolve() / "cron" / "executions.db"
_DEFAULT_EXECUTIONS_FILE = EXECUTIONS_FILE
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown")
_ERROR_CODES = frozenset({
    "delivery_failed",
    "dispatch_rejected",
    "empty_response",
    "execution_failed",
    "executor_dispatch_failed",
    "legacy_error_redacted",
    "scheduler_failed",
    "scheduler_restarted",
})
_RESULT_KINDS = frozenset({
    "delivery_failed",
    "dispatch_rejected",
    "empty_response",
    "execution_failed",
    "executor_dispatch_failed",
    "output_produced",
    "scheduler_failed",
    "scheduler_restarted",
    "silent",
})
_SOURCE_KINDS = frozenset({"builtin", "direct", "external", "unknown"})
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


def _source_kind(source: Any) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "chronos":
        return "external"
    return normalized if normalized in _SOURCE_KINDS else "unknown"


def _bounded_exit_code(value: Any) -> Optional[int]:
    if type(value) is not int:
        return None
    return value if -(2**31) <= value <= (2**31 - 1) else None


def _current_executions_file() -> Path:
    """Resolve the active profile ledger without breaking test overrides."""
    if EXECUTIONS_FILE != _DEFAULT_EXECUTIONS_FILE:
        return Path(EXECUTIONS_FILE).resolve()
    return get_hermes_home().resolve() / "cron" / "executions.db"


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply retryable privacy migrations before any rows are returned."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS schema_meta (
             key TEXT PRIMARY KEY,
             value TEXT NOT NULL
           )"""
    )
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    version = int(row[0]) if row is not None else 1
    if version >= 5:
        return

    conn.commit()
    try:
        conn.execute("BEGIN IMMEDIATE")
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(executions)")
        }
        additions = {
            "result_kind": "TEXT",
            "exit_code": "INTEGER",
            "delivery_requested": "INTEGER NOT NULL DEFAULT 0",
            "delivery_attempted": "INTEGER NOT NULL DEFAULT 0",
            "delivery_failed": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE executions ADD COLUMN {name} {definition}"
                )

        error_placeholders = ",".join("?" for _ in _ERROR_CODES)
        conn.execute(
            f"""UPDATE executions SET error='legacy_error_redacted'
                 WHERE error IS NOT NULL AND error NOT IN ({error_placeholders})""",
            tuple(sorted(_ERROR_CODES)),
        )
        conn.execute(
            "UPDATE executions SET source='external' WHERE lower(source)='chronos'"
        )
        source_placeholders = ",".join("?" for _ in _SOURCE_KINDS)
        conn.execute(
            f"""UPDATE executions SET source='unknown'
                 WHERE source NOT IN ({source_placeholders})""",
            tuple(sorted(_SOURCE_KINDS)),
        )
        result_placeholders = ",".join("?" for _ in _RESULT_KINDS)
        conn.execute(
            f"""UPDATE executions SET result_kind=CASE status
                   WHEN 'completed' THEN 'output_produced'
                   WHEN 'failed' THEN 'execution_failed'
                   WHEN 'unknown' THEN 'scheduler_restarted'
                   ELSE NULL END
                 WHERE result_kind IS NULL
                    OR result_kind NOT IN ({result_placeholders})""",
            tuple(sorted(_RESULT_KINDS)),
        )
        conn.execute(
            """UPDATE executions SET exit_code=NULL
                 WHERE typeof(exit_code) != 'integer'
                    OR exit_code < ? OR exit_code > ?""",
            (-(2**31), 2**31 - 1),
        )
        for column in (
            "delivery_requested",
            "delivery_attempted",
            "delivery_failed",
        ):
            conn.execute(
                f"""UPDATE executions SET {column}=CASE
                       WHEN typeof({column})='integer' AND {column} != 0 THEN 1
                       ELSE 0 END"""
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint and checkpoint[0]:
        raise sqlite3.OperationalError(
            "privacy migration checkpoint was busy; migration remains retryable"
        )

    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO schema_meta(key, value) VALUES('schema_version', '5')
               ON CONFLICT(key) DO UPDATE SET value=excluded.value"""
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _secure_storage(executions_file: Path) -> None:
    """Enforce owner-only modes without changing the process-global umask."""
    executions_file.parent.chmod(0o700)
    for path in (
        executions_file,
        Path(f"{executions_file}-wal"),
        Path(f"{executions_file}-shm"),
    ):
        if path.exists():
            path.chmod(0o600)


def _connect() -> sqlite3.Connection:
    executions_file = _current_executions_file()
    executions_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    executions_file.parent.chmod(0o700)
    conn = sqlite3.connect(executions_file, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA secure_delete=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT,
             result_kind TEXT,
             exit_code INTEGER,
             delivery_requested INTEGER NOT NULL DEFAULT 0,
             delivery_attempted INTEGER NOT NULL DEFAULT 0,
             delivery_failed INTEGER NOT NULL DEFAULT 0
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    try:
        _migrate_schema(conn)
        _secure_storage(executions_file)
    except Exception:
        conn.close()
        raise
    return conn


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?)""",
            (execution_id, str(job_id), _source_kind(source), _PROCESS_ID, pid,
             _process_start_time(pid), now),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    return _record(row)  # type: ignore[return-value]


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            """UPDATE executions SET status='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        return _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())


def finish_execution(
    execution_id: str,
    *,
    success: bool,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
    result_kind: Optional[str] = None,
    exit_code: Optional[int] = None,
    delivery_requested: bool = False,
    delivery_attempted: bool = False,
    delivery_failed: bool = False,
) -> Optional[Dict[str, Any]]:
    """Write a terminal categorical result once.

    ``error`` remains accepted for caller compatibility and local logging, but
    arbitrary text never crosses the durable-ledger boundary.
    """
    del error
    now = _hermes_now().isoformat()
    terminal_success = bool(success and not delivery_failed)
    status = "completed" if terminal_success else "failed"
    if result_kind not in _RESULT_KINDS:
        if delivery_failed:
            result_kind = "delivery_failed"
        elif terminal_success:
            result_kind = "output_produced"
        else:
            result_kind = "execution_failed"
    detail = None
    if not terminal_success:
        detail = error_code if error_code in _ERROR_CODES else result_kind
        if detail not in _ERROR_CODES:
            detail = "execution_failed"
    with _lock, _connect() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, result_kind=?, exit_code=?,
                   delivery_requested=?, delivery_attempted=?, delivery_failed=?
               WHERE id=? AND status IN ('claimed','running')""",
            (
                status,
                now,
                detail,
                result_kind,
                _bounded_exit_code(exit_code),
                int(bool(delivery_requested)),
                int(bool(delivery_attempted)),
                int(bool(delivery_failed)),
                execution_id,
            ),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        return _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    with _lock, _connect() as conn:
        rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error='scheduler_restarted',
                       result_kind='scheduler_restarted'
                   WHERE id=? AND status IN ('claimed','running')""",
                (now, row["id"]),
            )
            changed += cur.rowcount
        if changed:
            _prune_unlocked(conn)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
