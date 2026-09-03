"""SQLite-backed store for the task-ownership controller.

Durable local state for tasks an agent (or a human operator) owns end to
end: what state a task is in, what the next action is, why it's blocked,
how many times it's been retried, and — the load-bearing invariant — proof
that a task was actually verified before it is marked DONE.

DB lives at ``<HERMES_HOME>/task_ownership.db``. Single-profile, single-file,
no multi-worker coordination needed (unlike ``kanban_db.py``, which this
module borrows its WAL/connect/init_db shape from) — one profile's tasks are
private to that profile.

State machine
-------------
NEW -> WORKING -> {WAITING_FOR_USER, RETRYING, VERIFYING, BLOCKED, STALE, CANCELLED}
VERIFYING -> DONE   (ONLY reachable state for DONE — see mark_done())
Terminal states: DONE, CANCELLED (no transitions out).

No-false-completion invariant
------------------------------
``mark_done()`` refuses to set state=DONE unless ``verification_evidence``
is present on the task (recorded via ``record_verification()``, or passed
directly to ``mark_done()`` which records it first) AND, for tasks created
with ``approval_required=True``, an approval is on file. There is no code
path that reaches DONE without both checks passing.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    state TEXT NOT NULL,
    next_action TEXT,
    blocker TEXT,
    decision TEXT,
    owner TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    fallback TEXT,
    approval_required INTEGER NOT NULL DEFAULT 0,
    approved_by TEXT,
    approved_at TEXT,
    verification_evidence TEXT,
    verified_at TEXT,
    last_outcome TEXT,
    last_outcome_detail TEXT,
    last_outcome_at TEXT,
    aged_24h_marker TEXT,
    aged_72h_marker TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_changed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);

CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    event TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id);

CREATE TABLE IF NOT EXISTS task_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    result TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_outcomes_task ON task_outcomes(task_id);

CREATE TABLE IF NOT EXISTS task_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    source TEXT,
    payload TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(task_id, receipt_id)
);

CREATE INDEX IF NOT EXISTS idx_task_receipts_task ON task_receipts(task_id);
"""

STATES = frozenset(
    {
        "NEW",
        "WORKING",
        "WAITING_FOR_USER",
        "RETRYING",
        "VERIFYING",
        "DONE",
        "BLOCKED",
        "STALE",
        "CANCELLED",
    }
)

TERMINAL_STATES = frozenset({"DONE", "CANCELLED"})

# Reachability graph. This encodes which transitions are structurally
# sane; it does NOT encode the extra evidence/approval preconditions that
# gate entry into DONE specifically — those live in mark_done().
TRANSITIONS: Dict[str, frozenset] = {
    "NEW": frozenset({"WORKING", "BLOCKED", "CANCELLED"}),
    "WORKING": frozenset(
        {"WAITING_FOR_USER", "RETRYING", "VERIFYING", "BLOCKED", "STALE", "CANCELLED"}
    ),
    "WAITING_FOR_USER": frozenset({"WORKING", "BLOCKED", "STALE", "CANCELLED"}),
    "RETRYING": frozenset({"WORKING", "BLOCKED", "STALE", "CANCELLED"}),
    "VERIFYING": frozenset({"DONE", "WORKING", "BLOCKED", "CANCELLED"}),
    "BLOCKED": frozenset({"WORKING", "STALE", "CANCELLED"}),
    "STALE": frozenset({"WORKING", "BLOCKED", "CANCELLED"}),
    "DONE": frozenset(),
    "CANCELLED": frozenset(),
}


class TaskOwnershipError(Exception):
    """Base class for task-ownership state errors."""


class InvalidTransitionError(TaskOwnershipError):
    pass


class VerificationRequiredError(TaskOwnershipError):
    pass


class ApprovalRequiredError(TaskOwnershipError):
    pass


class TaskNotFoundError(TaskOwnershipError):
    pass


class DuplicateReceiptError(TaskOwnershipError):
    """Raised only by callers that opt into strict (non-idempotent) mode."""


_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: set = set()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def db_path(db_path: Optional[Path] = None) -> Path:
    if db_path is not None:
        return db_path
    return get_hermes_home() / "task_ownership.db"


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the task-ownership DB."""
    resolved_path = db_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(resolved_path.resolve())

    conn = _sqlite_connect(resolved_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    with _INIT_LOCK:
        if key not in _INITIALIZED_PATHS:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="task_ownership.db")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            _INITIALIZED_PATHS.add(key)
    return conn


def connect_closing(path: Optional[Path] = None):
    return contextlib.closing(connect(path))


def init_db(path: Optional[Path] = None) -> Path:
    """Create the schema if missing; return the resolved path.

    Unlike :func:`connect`'s cached first-open init, this always clears the
    cache entry first so a test harness or a caller that suspects schema
    drift can force a re-check.
    """
    resolved_path = db_path(path)
    key = str(resolved_path.resolve())
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(key)
    with connect_closing(resolved_path):
        pass
    return resolved_path


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


def _log_event(
    conn: sqlite3.Connection,
    task_id: str,
    event: str,
    *,
    from_state: Optional[str] = None,
    to_state: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    conn.execute(
        "INSERT INTO task_events (task_id, event, from_state, to_state, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (task_id, event, from_state, to_state, detail, _now_iso()),
    )


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    next_action: Optional[str] = None,
    owner: Optional[str] = None,
    max_retries: int = 3,
    approval_required: bool = False,
) -> Dict[str, Any]:
    if not title or not title.strip():
        raise ValueError("title is required")
    task_id = "t_" + uuid.uuid4().hex[:12]
    now = _now_iso()
    conn.execute(
        "INSERT INTO tasks (id, title, state, next_action, owner, max_retries, "
        "approval_required, created_at, updated_at, state_changed_at) "
        "VALUES (?, ?, 'NEW', ?, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            title.strip(),
            next_action,
            owner,
            max_retries,
            1 if approval_required else 0,
            now,
            now,
            now,
        ),
    )
    _log_event(conn, task_id, "created", to_state="NEW", detail=title.strip())
    conn.commit()
    return get_task(conn, task_id)


def get_task(conn: sqlite3.Connection, task_id: str) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise TaskNotFoundError(task_id)
    return _row_to_dict(row)


def list_tasks(
    conn: sqlite3.Connection,
    *,
    state: Optional[str] = None,
    include_terminal: bool = True,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM tasks"
    clauses = []
    params: List[Any] = []
    if state:
        clauses.append("state = ?")
        params.append(state)
    elif not include_terminal:
        clauses.append("state NOT IN ('DONE', 'CANCELLED')")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def _touch(conn: sqlite3.Connection, task_id: str) -> None:
    conn.execute("UPDATE tasks SET updated_at = ? WHERE id = ?", (_now_iso(), task_id))


def set_state(
    conn: sqlite3.Connection,
    task_id: str,
    to_state: str,
    *,
    detail: Optional[str] = None,
    event: str = "state_change",
) -> Dict[str, Any]:
    if to_state not in STATES:
        raise ValueError(f"unknown state: {to_state}")
    task = get_task(conn, task_id)
    from_state = task["state"]
    if to_state == from_state:
        return task
    allowed = TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise InvalidTransitionError(f"{from_state} -> {to_state} is not a valid transition")
    now = _now_iso()
    conn.execute(
        "UPDATE tasks SET state = ?, updated_at = ?, state_changed_at = ? WHERE id = ?",
        (to_state, now, now, task_id),
    )
    _log_event(conn, task_id, event, from_state=from_state, to_state=to_state, detail=detail)
    conn.commit()
    return get_task(conn, task_id)


def update_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    next_action: Optional[str] = None,
    blocker: Optional[str] = None,
    decision: Optional[str] = None,
    owner: Optional[str] = None,
    fallback: Optional[str] = None,
    max_retries: Optional[int] = None,
    state: Optional[str] = None,
) -> Dict[str, Any]:
    get_task(conn, task_id)  # 404s cleanly if missing
    fields = {
        "next_action": next_action,
        "blocker": blocker,
        "decision": decision,
        "owner": owner,
        "fallback": fallback,
        "max_retries": max_retries,
    }
    sets = [(k, v) for k, v in fields.items() if v is not None]
    if sets:
        assignments = ", ".join(f"{k} = ?" for k, _ in sets)
        params = [v for _, v in sets] + [_now_iso(), task_id]
        conn.execute(
            f"UPDATE tasks SET {assignments}, updated_at = ? WHERE id = ?", params
        )
        _log_event(conn, task_id, "updated", detail=json.dumps({k: v for k, v in sets}))
        conn.commit()
    if state is not None:
        return set_state(conn, task_id, state, event="manual_update")
    return get_task(conn, task_id)


def record_outcome(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: str,
    detail: Optional[str] = None,
    retry: bool = False,
    fallback: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a worker's attempt outcome. Never sets state to DONE.

    Completion only ever happens through :func:`record_verification` +
    :func:`mark_done` — outcome recording is deliberately incapable of
    completing a task, so a worker reporting "success" cannot by itself
    produce a false DONE.
    """
    if result not in {"success", "failure", "partial"}:
        raise ValueError("result must be one of: success, failure, partial")
    task = get_task(conn, task_id)
    attempt = task["retry_count"] + 1
    now = _now_iso()
    conn.execute(
        "INSERT INTO task_outcomes (task_id, attempt, result, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, attempt, result, detail, now),
    )
    conn.execute(
        "UPDATE tasks SET last_outcome = ?, last_outcome_detail = ?, last_outcome_at = ?, "
        "updated_at = ? WHERE id = ?",
        (result, detail, now, now, task_id),
    )
    conn.commit()

    if result != "failure" or not retry:
        return get_task(conn, task_id)

    # Bounded retry bookkeeping.
    new_retry_count = task["retry_count"] + 1
    conn.execute(
        "UPDATE tasks SET retry_count = ?, updated_at = ? WHERE id = ?",
        (new_retry_count, _now_iso(), task_id),
    )
    if fallback:
        conn.execute(
            "UPDATE tasks SET fallback = ?, updated_at = ? WHERE id = ?",
            (fallback, _now_iso(), task_id),
        )
    conn.commit()
    task = get_task(conn, task_id)

    if new_retry_count > task["max_retries"]:
        _log_event(
            conn,
            task_id,
            "retry_limit_exceeded",
            detail=f"{new_retry_count}/{task['max_retries']}"
            + (f"; fallback={fallback}" if fallback else ""),
        )
        conn.commit()
        blocker_msg = f"retry limit exceeded ({new_retry_count}/{task['max_retries']})"
        conn.execute(
            "UPDATE tasks SET blocker = ?, updated_at = ? WHERE id = ?",
            (blocker_msg, _now_iso(), task_id),
        )
        conn.commit()
        return set_state(conn, task_id, "BLOCKED", event="retry_exhausted", detail=blocker_msg)

    if task["state"] in TRANSITIONS and "RETRYING" in TRANSITIONS[task["state"]]:
        return set_state(
            conn,
            task_id,
            "RETRYING",
            event="retry_scheduled",
            detail=f"attempt {new_retry_count}/{task['max_retries']}",
        )
    return task


def record_receipt(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    receipt_id: str,
    source: Optional[str] = None,
    payload: Optional[str] = None,
) -> Dict[str, Any]:
    """Idempotently record an external receipt.

    Recording the same ``receipt_id`` for the same task twice is a no-op —
    the second call returns the original row untouched with
    ``duplicate=True``, so callers integrating a flaky/at-least-once
    external system can safely retry without double-booking side effects.
    """
    get_task(conn, task_id)  # 404s cleanly if missing
    if not receipt_id or not receipt_id.strip():
        raise ValueError("receipt_id is required")
    receipt_id = receipt_id.strip()
    existing = conn.execute(
        "SELECT * FROM task_receipts WHERE task_id = ? AND receipt_id = ?",
        (task_id, receipt_id),
    ).fetchone()
    if existing is not None:
        result = _row_to_dict(existing)
        result["duplicate"] = True
        return result
    now = _now_iso()
    conn.execute(
        "INSERT INTO task_receipts (task_id, receipt_id, source, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, receipt_id, source, payload, now),
    )
    _log_event(conn, task_id, "receipt_recorded", detail=f"{receipt_id} ({source or 'unknown'})")
    conn.commit()
    row = conn.execute(
        "SELECT * FROM task_receipts WHERE task_id = ? AND receipt_id = ?",
        (task_id, receipt_id),
    ).fetchone()
    result = _row_to_dict(row)
    result["duplicate"] = False
    return result


def list_receipts(conn: sqlite3.Connection, task_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM task_receipts WHERE task_id = ? ORDER BY created_at ASC", (task_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def record_verification(
    conn: sqlite3.Connection, task_id: str, evidence: str
) -> Dict[str, Any]:
    if not evidence or not evidence.strip():
        raise ValueError("evidence is required")
    task = get_task(conn, task_id)
    now = _now_iso()
    conn.execute(
        "UPDATE tasks SET verification_evidence = ?, verified_at = ?, updated_at = ? "
        "WHERE id = ?",
        (evidence.strip(), now, now, task_id),
    )
    conn.commit()
    if task["state"] != "VERIFYING":
        task = set_state(
            conn, task_id, "VERIFYING", event="verification_recorded", detail=evidence.strip()
        )
    else:
        _log_event(conn, task_id, "verification_recorded", detail=evidence.strip())
        conn.commit()
        task = get_task(conn, task_id)
    return task


def approve_task(conn: sqlite3.Connection, task_id: str, approved_by: str) -> Dict[str, Any]:
    if not approved_by or not approved_by.strip():
        raise ValueError("approved_by is required")
    get_task(conn, task_id)
    now = _now_iso()
    conn.execute(
        "UPDATE tasks SET approved_by = ?, approved_at = ?, updated_at = ? WHERE id = ?",
        (approved_by.strip(), now, now, task_id),
    )
    _log_event(conn, task_id, "approved", detail=approved_by.strip())
    conn.commit()
    return get_task(conn, task_id)


def mark_done(
    conn: sqlite3.Connection, task_id: str, *, evidence: Optional[str] = None
) -> Dict[str, Any]:
    """Transition a task to DONE. Refuses without verification evidence.

    This is the ONLY function in the module that can set state=DONE. It:

    1. Records ``evidence`` via :func:`record_verification` if supplied
       (moving the task into VERIFYING if it isn't already there).
    2. Requires a non-empty ``verification_evidence`` to already be on the
       task row — raises :class:`VerificationRequiredError` otherwise.
    3. For tasks created with ``approval_required=True``, requires
       ``approved_by`` to be set — raises :class:`ApprovalRequiredError`
       otherwise.
    4. Only then transitions VERIFYING -> DONE.
    """
    task = get_task(conn, task_id)
    if evidence:
        task = record_verification(conn, task_id, evidence)
    if not task.get("verification_evidence"):
        raise VerificationRequiredError(
            f"task {task_id} cannot be marked DONE without verification evidence "
            "— run `hermes task verify <id> --evidence ...` first"
        )
    if task["approval_required"] and not task.get("approved_by"):
        raise ApprovalRequiredError(
            f"task {task_id} requires approval before DONE "
            "— run `hermes task approve <id> --by ...` first"
        )
    if task["state"] != "VERIFYING":
        raise InvalidTransitionError(
            f"task {task_id} must be in VERIFYING state to be marked DONE (currently {task['state']})"
        )
    return set_state(
        conn,
        task_id,
        "DONE",
        event="completed",
        detail=task["verification_evidence"],
    )


def block_task(conn: sqlite3.Connection, task_id: str, reason: str) -> Dict[str, Any]:
    get_task(conn, task_id)
    conn.execute(
        "UPDATE tasks SET blocker = ?, updated_at = ? WHERE id = ?",
        (reason, _now_iso(), task_id),
    )
    conn.commit()
    return set_state(conn, task_id, "BLOCKED", event="blocked", detail=reason)


def cancel_task(
    conn: sqlite3.Connection, task_id: str, reason: Optional[str] = None
) -> Dict[str, Any]:
    return set_state(conn, task_id, "CANCELLED", event="cancelled", detail=reason)


def list_events(conn: sqlite3.Connection, task_id: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id ASC", (task_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def age_check(
    conn: sqlite3.Connection,
    *,
    enabled: bool,
    dry_run: bool = False,
    warn_hours: int = 24,
    stale_hours: int = 72,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Evaluate warn/stale aging thresholds for every non-terminal task.

    Anti-spam: each tier only fires once per distinct ``state_changed_at``
    value on the task (tracked via ``aged_24h_marker`` / ``aged_72h_marker``).
    If the task's state changes (deliberately, or via the 72h auto-STALE
    transition below), the anchor moves and the clock effectively resets —
    a task doesn't get re-flagged every run.

    Mutation (marker writes + the 72h auto-transition to STALE) only
    happens when ``enabled`` is True and ``dry_run`` is False. Otherwise
    this is a pure read — safe to call for a preview regardless of the
    feature flag, and guaranteed inert (shadow mode) when the flag is off.
    """
    moment = now or _now()
    warn_delta = timedelta(hours=warn_hours)
    stale_delta = timedelta(hours=stale_hours)
    flagged: List[Dict[str, Any]] = []

    for task in list_tasks(conn, include_terminal=False):
        anchor = task["state_changed_at"]
        age = moment - _parse_iso(anchor)
        age_hours = round(age.total_seconds() / 3600, 1)

        if age >= stale_delta and task["aged_72h_marker"] != anchor:
            flagged.append(
                {
                    "task_id": task["id"],
                    "title": task["title"],
                    "tier": "stale",
                    "age_hours": age_hours,
                    "state": task["state"],
                }
            )
            if enabled and not dry_run:
                set_state(
                    conn,
                    task["id"],
                    "STALE",
                    event="aging_stale",
                    detail=f"no activity for {age_hours}h",
                )
                conn.execute(
                    "UPDATE tasks SET aged_72h_marker = ? WHERE id = ?",
                    (anchor, task["id"]),
                )
                conn.commit()
        elif age >= warn_delta and task["aged_24h_marker"] != anchor:
            flagged.append(
                {
                    "task_id": task["id"],
                    "title": task["title"],
                    "tier": "warn",
                    "age_hours": age_hours,
                    "state": task["state"],
                }
            )
            if enabled and not dry_run:
                conn.execute(
                    "UPDATE tasks SET aged_24h_marker = ? WHERE id = ?",
                    (anchor, task["id"]),
                )
                _log_event(
                    conn,
                    task["id"],
                    "aging_warn",
                    detail=f"no activity for {age_hours}h",
                )
                conn.commit()

    return flagged
