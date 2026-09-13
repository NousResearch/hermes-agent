"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue. Interrupted attempts
become ``unknown`` only after their exact owner process is proved gone. Terminal states are
immutable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so dashboard operations
# that temporarily enter another profile cannot leak that profile's records into the import-time
# home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
SUPERSEDED_STATUS = "superseded"
_TERMINAL_STATES = ("completed", "failed", "unknown", SUPERSEDED_STATUS)
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


class DuplicateFireAttempt(RuntimeError):
    """A second fire of an occurrence (``job_id`` + ``scheduled_instant``) that already completed.

    One occurrence runs the job's side effects once. The ledger — the seam every fire crosses to
    record its attempt — refuses the repeat, so a restored slot, a replayed catch-up or a duplicate
    dispatch cannot do the work a second time. Non-terminal, ``failed`` and ``unknown`` siblings are
    NOT duplicates: those are the retries the ledger is designed to hold.
    """

    def __init__(self, job_id: str, scheduled_instant: str, existing_id: str,
                 existing_status: str) -> None:
        super().__init__(
            f"Occurrence {scheduled_instant} of job {job_id!r} already completed as attempt "
            f"{existing_id} (status {existing_status}); a second fire for the same occurrence "
            "is refused"
        )
        self.job_id = job_id
        self.scheduled_instant = scheduled_instant
        self.existing_id = existing_id
        self.existing_status = existing_status


# --- executions ledger --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


_EXECUTIONS_DDL = """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown','superseded')),
             handoff_pending INTEGER NOT NULL DEFAULT 0,
             handoff_started_at REAL,
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""

_EXECUTIONS_COLUMNS = (
    "id", "job_id", "source", "process_id", "pid", "process_started_at", "status",
    "handoff_pending", "handoff_started_at", "claimed_at", "started_at", "finished_at", "error",
    "delivery_outcome", "scheduled_instant",
)


def _create_execution_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant) WHERE status='completed'"
    )


def _add_optional_columns(conn: sqlite3.Connection) -> None:
    """Columns added after the first release, on BOTH the fresh and the rebuilt table."""
    from hermes_cli.sqlite_util import add_column_if_missing

    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(conn, "executions", "handoff_started_at", "handoff_started_at REAL")
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")


def _adopt_status_vocabulary(conn: sqlite3.Connection) -> None:
    """Rebuild a ledger written before ``superseded`` was a status.

    The status vocabulary lives in the table's CHECK constraint, and SQLite cannot ALTER one — so a
    profile whose ``executions.db`` predates this build is rebuilt in place once, copying every
    column and row verbatim. Terminal outcomes are data, not derivable state: they are moved, never
    recomputed. Idempotent — a table already carrying the vocabulary is left untouched.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='executions'"
    ).fetchone()
    ddl = ""
    if row is not None:
        ddl = (row["sql"] if isinstance(row, sqlite3.Row) else row[0]) or ""
    if SUPERSEDED_STATUS in ddl:
        return
    present = [
        str(column[1]) for column in conn.execute("PRAGMA table_info(executions)").fetchall()
    ]
    shared = [column for column in _EXECUTIONS_COLUMNS if column in present]
    columns = ", ".join(shared)
    conn.execute("ALTER TABLE executions RENAME TO executions_legacy")
    conn.execute(_EXECUTIONS_DDL)
    _add_optional_columns(conn)
    conn.execute(f"INSERT INTO executions ({columns}) SELECT {columns} FROM executions_legacy")
    conn.execute("DROP TABLE executions_legacy")
    _create_execution_indexes(conn)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_EXECUTIONS_DDL)
    _add_optional_columns(conn)
    _create_execution_indexes(conn)
    _adopt_status_vocabulary(conn)


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _fetch(conn: sqlite3.Connection, execution_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


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
    placeholders = ", ".join("?" for _ in _TERMINAL_STATES)
    conn.execute(
        f"""DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ({placeholders})
             ORDER BY finished_at DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (*_TERMINAL_STATES, max(0, int(MAX_TERMINAL_EXECUTIONS))),
    )


def _refuse_completed_occurrence(conn: sqlite3.Connection, job_id: str, instant: str) -> None:
    """Refuse a second fire of an occurrence that ALREADY COMPLETED.

    A sibling attempt that is merely non-terminal (``claimed``/``running``), ``failed`` or
    ``unknown`` is NOT a duplicate: the ledger deliberately holds several attempts per occurrence —
    a retry after a transient failure, and the restore-once of a slot whose owner is provably gone
    (#107485, ``cron/occurrences.py``). ``completed`` is the one state that proves this occurrence's
    work is already done, and firing it again would run the job's side effects twice.
    """
    rows = conn.execute(
        "SELECT id, status FROM executions WHERE job_id=? AND scheduled_instant=? "
        "ORDER BY claimed_at, id",
        (str(job_id), instant),
    ).fetchall()
    for row in rows:
        record = dict(row)
        if str(record.get("status") or "") == "completed":
            raise DuplicateFireAttempt(
                str(job_id), instant, str(record["id"]), "completed")


def create_execution(
    job_id: str, *, source: str, scheduled_instant: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch.

    Raised ``DuplicateFireAttempt``: this occurrence (``job_id`` + ``scheduled_instant``) is already
    accounted for by a live or completed attempt. No row is written for the refused fire.
    """
    from cron.occurrences import scheduled_instant as canonical_instant

    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    instant = canonical_instant(scheduled_instant)
    with _transaction() as conn:
        if instant is not None:
            _refuse_completed_occurrence(conn, str(job_id), instant)
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, scheduled_instant)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, instant),
        )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def _close_refused_attempt(execution_id: str) -> None:
    """Close an attempt whose fire was refused before it could run anything.

    Its own transaction: the caller's raised refusal rolls its transaction back, so the row must be
    terminalized outside it or the refused attempt would sit non-terminal forever.
    """
    with _transaction() as conn:
        conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?
               WHERE id=? AND status='claimed' AND process_id=? AND pid=?""",
            (SUPERSEDED_STATUS, _hermes_now().isoformat(),
             "Superseded: this occurrence is already accounted for by another attempt.",
             execution_id, _PROCESS_ID, os.getpid()),
        )
        _emit_execution_state(_fetch(conn, execution_id))


def set_execution_occurrence(execution_id: str, instant: Optional[str]) -> None:
    """Bind the store-claimed snapshot before a provider hands it to a worker.

    Refuses an occurrence that is already accounted for (a live or completed attempt for the same
    ``job_id`` + instant): the attempt is closed as ``superseded`` and ``DuplicateFireAttempt`` is
    raised, so the caller's dispatch fails closed without running the job a second time.
    """
    from cron.occurrences import scheduled_instant

    canonical = scheduled_instant(instant)
    try:
        with _transaction() as conn:
            row = _fetch(conn, execution_id)
            if row is None:
                raise RuntimeError("Cron occurrence could not be bound before dispatch")
            if canonical is not None:
                _refuse_completed_occurrence(conn, str(row["job_id"]), canonical)
            cur = conn.execute(
                "UPDATE executions SET scheduled_instant=? WHERE id=? AND status='claimed' "
                "AND handoff_pending=0 AND process_id=? AND pid=?",
                (canonical, execution_id, _PROCESS_ID, os.getpid()),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Cron occurrence could not be bound before dispatch")
    except DuplicateFireAttempt:
        _close_refused_attempt(execution_id)
        raise


def mark_execution_handoff_pending(execution_id: str) -> Optional[Dict[str, Any]]:
    """Fence restart recovery while an external worker is adopting a claim."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET handoff_pending=1, handoff_started_at=?
               WHERE id=? AND status='claimed'
                 AND process_id=? AND pid=?""",
            (time.time(), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def adopt_claimed_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Atomically transfer and start an attempt in its worker process.

    The dispatching gateway creates the row before spawning a restart-safe
    worker.  Adoption is the single ``claimed`` → ``running`` gate: only the
    winner may acknowledge ownership or run side effects.
    """
    pid = os.getpid()
    process_started_at = _process_start_time(pid)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=?, process_started_at=?,
                   status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=1""",
            (_PROCESS_ID, pid, process_started_at, now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None, superseded: bool = False,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten.

    ``superseded``: the attempt stopped running but its outcome does not describe the job — a later
    fire for the same job, or a completed sibling of the same occurrence, already owns that. It is
    recorded as its own terminal status rather than as another ``failed`` row, so a delivered or
    inconclusive attempt is never reported as a plain error.
    """
    now = _hermes_now().isoformat()
    if superseded:
        status = SUPERSEDED_STATUS
        detail = str(error) if error else "Superseded by a later attempt for this job."
    else:
        status = "completed" if success else "failed"
        detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, handoff_pending=0,
                   handoff_started_at=NULL, delivery_outcome=?
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, detail, delivery_outcome, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    return record


# Newest-attempt ordering. `claimed_at` is WALL CLOCK and `id` is a random uuid4, so the previous
# `ORDER BY claimed_at DESC, id DESC` ordered two fires that landed inside the same clock tick
# ARBITRARILY: whichever uuid happened to sort higher looked "newest", so the attempt trusted to
# own the job record could be the OLDER one (WH-CREATED-5A4D2A184BBA AC4). `rowid` is SQLite's
# monotonically increasing insert counter, so equal timestamps fall back to true insertion order.
# Every recency decision (newest_attempt_id -> attempt_is_newest -> attempt_owns_job_record /
# job_status_write_blocked) inherits this tiebreak.
_ATTEMPT_RECENCY_ORDER = "ORDER BY claimed_at DESC, rowid DESC"


def newest_attempt_id(job_id: str) -> Optional[str]:
    """Id of this job's newest attempt, or ``None`` when it has none."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT id FROM executions WHERE job_id=? " + _ATTEMPT_RECENCY_ORDER + " LIMIT 1",
            (str(job_id),),
        ).fetchone()
    return str(row["id"]) if row is not None else None


def occurrence_rows(job_id: str, scheduled_instant: Optional[str]) -> List[Dict[str, Any]]:
    """Every attempt recorded for one exact occurrence (``None`` has no identity → no rows)."""
    from cron.occurrences import scheduled_instant as canonical_instant

    instant = canonical_instant(scheduled_instant)
    if instant is None:
        return []
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions WHERE job_id=? AND scheduled_instant=? "
            "ORDER BY claimed_at, id",
            (str(job_id), instant),
        ).fetchall()
    return [dict(row) for row in rows]


def occurrence_completed(job_id: str, scheduled_instant: Optional[str]) -> bool:
    """True when some attempt already COMPLETED this exact occurrence."""
    return any(row.get("status") == "completed" for row in occurrence_rows(job_id, scheduled_instant))


def occurrence_winner(job_id: str, scheduled_instant: Optional[str]) -> Optional[str]:
    """The status that describes the occurrence: its best surviving outcome.

    ``completed`` outranks everything — an occurrence that ran cannot be reported as an error just
    because a sibling attempt stopped inconclusively.
    """
    outcomes = [str(row.get("status") or "") for row in occurrence_rows(job_id, scheduled_instant)]
    for candidate in ("completed", SUPERSEDED_STATUS, "failed", "unknown"):
        if candidate in outcomes:
            return candidate
    return None


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: List[Dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at
               FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            handoff_started_at = row["handoff_started_at"]
            if (
                row["handoff_pending"]
                and handoff_started_at is not None
                and time.time() - float(handoff_started_at)
                < HANDOFF_ADOPTION_GRACE_SECONDS
            ):
                continue
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?,
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _fetch(conn, row["id"])
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50, before_claimed_at: Optional[str] = None,
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
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Return one exact execution attempt, or ``None`` when it is absent."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
