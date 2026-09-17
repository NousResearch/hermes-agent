"""Durable idempotency reservations for API server runs."""

import hmac
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from hermes_cli.sqlite_util import add_column_if_missing


# Keep the extracted store's log records on the API server logger.
logger = logging.getLogger("gateway.platforms.api_server")

TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "interrupted", "superseded", "unrecoverable"}
)

_SELECT_BY_KEY = (
    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
    "FROM run_idempotency WHERE scope=? AND idempotency_key=?")
_EXTEND_RETENTION_BY_KEY = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND idempotency_key=? AND fingerprint=?")
_EXTEND_RETENTION_BY_RUN = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND run_id=?")
# Columns added after the first schema shipped; applied when missing.
_MIGRATIONS = {
    "owner_pid": "INTEGER NOT NULL DEFAULT 0",
    "owner_started": "INTEGER NOT NULL DEFAULT 0",
    "retention_until": "REAL NOT NULL DEFAULT 0",
    "acknowledged_at": "REAL"}

_APPROVAL_MIGRATIONS = {
    "tool_json": "TEXT",
    "dispatch_state": "TEXT NOT NULL DEFAULT 'pending'",
    "result_json": "TEXT",
    "applied_at": "REAL",
    "dispatch_started_at": "REAL",
    "completed_at": "REAL",
}


def _encode_status(status: Dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True, separators=(",", ":"))


def _record(run_id, status_json, owner_pid, owner_started, updated_at) -> dict[str, Any]:
    return {
        "run_id": run_id, "status": json.loads(status_json), "owner_pid": int(owner_pid or 0),
        "owner_started": int(owner_started or 0), "updated_at": float(updated_at or 0)}


def _outcome(row, fingerprint):
    """Classify a stored ``(scope, key)`` row against the caller's fingerprint."""
    return ("reused" if hmac.compare_digest(row[0], fingerprint) else "conflict"), _record(*row[1:])


class RunIdempotencyStore:
    """Durable, tenant-scoped reservations for ``POST /v1/runs``: a unique ``(scope, key)`` row
    inserted inside ``BEGIN IMMEDIATE`` so separate workers cannot both admit one request. Only
    fingerprints and public run status are stored — never request bodies or credentials."""

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60

    @property
    def durable(self) -> bool:
        """Whether reservations survive this process."""
        return self._db_path is not None
    def __init__(self, db_path: str = None):
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "runs_idempotency.db")
            except Exception:
                db_path = ":memory:"
        self._db_path = None if db_path == ":memory:" else db_path
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        except Exception as exc:
            # Docker may create the container object before `docker run` fails to start it (e.g. exit code
            # 125 when the daemon isn't ready, or a timeout mid-pull). That orphan is left in "Created"
            # state — which the exited-only orphan reaper (reap_orphan_containers, status=exited) never
            # catches, so it leaks permanently. Remove it by its known name before re-raising. See #7439.
            logger.warning(
                "Run idempotency storage is unavailable; falling back to "
                "process memory, so replay will not survive a restart: %s", exc)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0,
                owner_started INTEGER NOT NULL DEFAULT 0,
                retention_until REAL NOT NULL DEFAULT 0,
                acknowledged_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )"""
        )
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(run_idempotency)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in columns:
                add_column_if_missing(self._conn, "run_idempotency", column, f"{column} {ddl}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS run_idempotency_run_id ON run_idempotency(run_id)")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, sequence)
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_approvals (
                run_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                choice TEXT,
                applied INTEGER,
                resolved INTEGER,
                tool_json TEXT,
                dispatch_state TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT,
                applied_at REAL,
                dispatch_started_at REAL,
                completed_at REAL,
                created_at REAL NOT NULL,
                responded_at REAL,
                PRIMARY KEY (run_id, request_id)
            )"""
        )
        approval_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(run_approvals)")
        }
        for column, ddl in _APPROVAL_MIGRATIONS.items():
            if column not in approval_columns:
                add_column_if_missing(self._conn, "run_approvals", column, f"{column} {ddl}")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_launches (
                run_id TEXT PRIMARY KEY,
                launch_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_recovery (
                parent_run_id TEXT PRIMARY KEY,
                successor_run_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self._tighten_permissions()

    def _tighten_permissions(self) -> None:
        for suffix in ("", "-wal", "-shm") if self._db_path else ():
            candidate = Path(self._db_path + suffix)
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict run idempotency store permissions", exc_info=True)

    @contextmanager
    def _immediate_txn(self):
        """Hold the lock inside ``BEGIN IMMEDIATE``; the body commits, errors roll back."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._conn.rollback()
                raise

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str, status: Dict[str, Any], *,
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0):
        """Atomically reserve a key; return ``(outcome, stored_record)``."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is not None:
                if retention_until:
                    self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
                self._conn.commit()
                return _outcome(row, fingerprint)
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, now, now))
            self._conn.commit()
            return "created", _record(run_id, encoded, owner_pid, owner_started, now) | {"status": status}

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._immediate_txn():
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            self._conn.commit()
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        for stale_scope, stale_key, stale_status in stale:
            try:
                terminal = json.loads(stale_status).get("status") in TERMINAL_STATUSES
            except Exception:
                terminal = False
            if terminal:
                run_row = self._conn.execute(
                    "SELECT run_id FROM run_idempotency WHERE scope=? AND idempotency_key=?",
                    (stale_scope, stale_key),
                ).fetchone()
                if run_row is not None:
                    self._conn.execute("DELETE FROM run_events WHERE run_id=?", (run_row[0],))
                    self._conn.execute("DELETE FROM run_approvals WHERE run_id=?", (run_row[0],))
                    self._conn.execute("DELETE FROM run_launches WHERE run_id=?", (run_row[0],))
                    self._conn.execute(
                        "DELETE FROM run_recovery WHERE parent_run_id=? OR successor_run_id=?",
                        (run_row[0], run_row[0]),
                    )
                self._conn.execute(
                    "DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (stale_scope, stale_key))

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0) -> dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_RUN, (retention_until, scope, run_id))
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id)).fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            changed = self._conn.execute(_EXTEND_RETENTION_BY_RUN, (checked_until, scope, run_id)).rowcount
            self._conn.commit()
        return changed == 1

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone()
        return row is not None

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), time.time(), run_id))
            self._conn.commit()

    def save_run_launch(self, run_id: str, launch: Dict[str, Any]) -> Dict[str, Any]:
        """Freeze the non-secret inputs needed to continue a run after process loss."""
        encoded = _encode_status(dict(launch))
        now = time.time()
        with self._immediate_txn():
            if self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE run_id=?", (run_id,)
            ).fetchone() is None:
                raise KeyError(run_id)
            self._conn.execute(
                """INSERT OR IGNORE INTO run_launches(run_id,launch_json,created_at,updated_at)
                   VALUES(?,?,?,?)""",
                (run_id, encoded, now, now),
            )
            row = self._conn.execute(
                "SELECT launch_json FROM run_launches WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None or not hmac.compare_digest(str(row[0]), encoded):
                raise ValueError("run launch conflicts with an existing launch")
            self._conn.commit()
        return json.loads(row[0])

    def append_event(self, run_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append one event and return its stable, run-local sequence envelope.

        Events are admitted only for durable run rows. Callers can therefore
        expose the cursor under the same authenticated scope as run status.
        """
        now = time.time()
        with self._immediate_txn():
            if self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE run_id=?", (run_id,)
            ).fetchone() is None:
                raise KeyError(run_id)
            stored = self._append_event_locked(run_id, event, now)
            self._conn.commit()
        return stored

    def _append_event_locked(
        self, run_id: str, event: Dict[str, Any], created_at: float
    ) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id=?",
            (run_id,),
        ).fetchone()
        sequence = int(row[0])
        stored = dict(event)
        stored["sequence"] = sequence
        self._conn.execute(
            "INSERT INTO run_events(run_id,sequence,event_json,created_at) VALUES(?,?,?,?)",
            (run_id, sequence, _encode_status(stored), created_at),
        )
        return stored

    def update_status_and_append_event(
        self, run_id: str, status: Dict[str, Any], event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Commit a status transition and its replay event as one boundary."""
        now = time.time()
        with self._immediate_txn():
            changed = self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), now, run_id),
            ).rowcount
            if changed != 1:
                raise KeyError(run_id)
            stored = self._append_event_locked(run_id, event, now)
            self._conn.commit()
        return stored

    def events_after(self, scope: str, run_id: str, sequence: int) -> list[Dict[str, Any]]:
        """Return events after *sequence* only when *scope* owns the run."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.event_json
                     FROM run_events AS e
                     JOIN run_idempotency AS r ON r.run_id=e.run_id
                    WHERE r.scope=? AND e.run_id=? AND e.sequence>?
                    ORDER BY e.sequence""",
                (scope, run_id, max(0, int(sequence or 0))),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_approval_request(self, run_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a redacted approval request without rewriting an existing one."""
        request_id = str(request.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("approval request_id is required")
        encoded = _encode_status(dict(request))
        now = time.time()
        with self._immediate_txn():
            if self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE run_id=?", (run_id,)
            ).fetchone() is None:
                raise KeyError(run_id)
            row = self._save_approval_request_locked(
                run_id, request_id, encoded, now
            )
            self._conn.commit()
        return {
            "run_id": run_id,
            "request_id": request_id,
            "request": json.loads(row[0]),
            "state": str(row[1]),
        }

    def _save_approval_request_locked(
        self, run_id: str, request_id: str, encoded: str, created_at: float
    ):
        self._conn.execute(
            """INSERT OR IGNORE INTO run_approvals(
                   run_id,request_id,request_json,state,created_at
               ) VALUES(?,?,?,'pending',?)""",
            (run_id, request_id, encoded, created_at),
        )
        row = self._conn.execute(
            "SELECT request_json,state FROM run_approvals WHERE run_id=? AND request_id=?",
            (run_id, request_id),
        ).fetchone()
        if row is None or not hmac.compare_digest(str(row[0]), encoded):
            raise ValueError("approval request_id conflicts with an existing request")
        return row

    def record_approval_request(
        self,
        run_id: str,
        status: Dict[str, Any],
        event: Dict[str, Any],
        *,
        tool: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Commit pending status, request, and replay event as one boundary."""
        request_id = str(event.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("approval request_id is required")
        now = time.time()
        with self._immediate_txn():
            changed = self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), now, run_id),
            ).rowcount
            if changed != 1:
                raise KeyError(run_id)
            self._save_approval_request_locked(
                run_id, request_id, _encode_status(dict(event)), now
            )
            if tool is not None:
                encoded_tool = _encode_status(dict(tool))
                row = self._conn.execute(
                    "SELECT tool_json FROM run_approvals WHERE run_id=? AND request_id=?",
                    (run_id, request_id),
                ).fetchone()
                if row is None:
                    raise KeyError(request_id)
                if row[0] is None:
                    self._conn.execute(
                        "UPDATE run_approvals SET tool_json=? WHERE run_id=? AND request_id=?",
                        (encoded_tool, run_id, request_id),
                    )
                elif not hmac.compare_digest(str(row[0]), encoded_tool):
                    raise ValueError("approval tool conflicts with an existing request")
            stored = self._append_event_locked(run_id, event, now)
            self._conn.commit()
        return stored

    def pending_approval(
        self, scope: str, run_id: str, request_id: str = ""
    ) -> Dict[str, Any] | None:
        """Load one unresolved approval through the run's authenticated scope."""
        return self.approval_for_run(scope, run_id, request_id, pending_only=True)

    def approval_for_run(
        self,
        scope: str,
        run_id: str,
        request_id: str = "",
        *,
        pending_only: bool = False,
    ) -> Dict[str, Any] | None:
        """Load one approval request or receipt through the run's scope."""
        query = (
            """SELECT a.request_id,a.request_json,a.state,a.choice,a.applied,a.resolved
                 FROM run_approvals AS a
                 JOIN run_idempotency AS r ON r.run_id=a.run_id
                WHERE r.scope=? AND a.run_id=?"""
        )
        params: list[Any] = [scope, run_id]
        if pending_only:
            query += " AND a.state='pending'"
        if request_id:
            query += " AND a.request_id=?"
            params.append(request_id)
        query += " ORDER BY a.created_at,a.request_id LIMIT 1"
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        if row is None:
            return None
        result = {
            "run_id": run_id,
            "request_id": str(row[0]),
            "request": json.loads(row[1]),
            "state": str(row[2]),
        }
        if result["state"] == "resolved":
            result["receipt"] = {
                "run_id": run_id,
                "request_id": str(row[0]),
                "choice": str(row[3]),
                "applied": bool(row[4]),
                "resolved": int(row[5] or 0),
            }
        return result

    def approval_dispatch_state(
        self, scope: str, run_id: str, request_id: str
    ) -> str | None:
        """Return the private execution fence state for one scoped approval."""
        with self._lock:
            row = self._conn.execute(
                """SELECT a.dispatch_state
                     FROM run_approvals AS a
                     JOIN run_idempotency AS r ON r.run_id=a.run_id
                    WHERE r.scope=? AND a.run_id=? AND a.request_id=?""",
                (scope, run_id, request_id),
            ).fetchone()
        return None if row is None else str(row[0])

    def recovery_snapshot(self, scope: str, run_id: str) -> Dict[str, Any] | None:
        """Return private recovery inputs without changing the dispatch fence.

        The API handler uses this immediately before successor reservation to
        reconcile a canonical SessionDB tool row that may have been committed
        just before the gateway process died.
        """
        with self._lock:
            approval = self._conn.execute(
                """SELECT a.request_id,a.choice,a.dispatch_state,a.tool_json,a.result_json
                     FROM run_approvals AS a
                     JOIN run_idempotency AS r ON r.run_id=a.run_id
                    WHERE r.scope=? AND a.run_id=? AND a.state='resolved'
                      AND a.tool_json IS NOT NULL
                    ORDER BY a.responded_at DESC,a.created_at DESC LIMIT 1""",
                (scope, run_id),
            ).fetchone()
            launch = self._conn.execute(
                "SELECT launch_json FROM run_launches WHERE run_id=?", (run_id,)
            ).fetchone()
        if approval is None or launch is None:
            return None
        return {
            **self._recovery_plan_values(approval, launch[0]),
            "dispatch_state": str(approval[2]),
        }

    def resolve_approval(
        self,
        scope: str,
        run_id: str,
        request_id: str,
        choice: str,
        *,
        applied: bool,
        resolved: int,
    ) -> tuple[str, Dict[str, Any] | None]:
        """Persist a decision receipt; identical retries replay the first receipt."""
        with self._immediate_txn():
            row = self._conn.execute(
                """SELECT a.state,a.choice,a.applied,a.resolved
                     FROM run_approvals AS a
                     JOIN run_idempotency AS r ON r.run_id=a.run_id
                    WHERE r.scope=? AND a.run_id=? AND a.request_id=?""",
                (scope, run_id, request_id),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return "missing", None
            state, stored_choice, stored_applied, stored_resolved = row
            if state != "pending":
                receipt = {
                    "run_id": run_id,
                    "request_id": request_id,
                    "choice": str(stored_choice),
                    "applied": bool(stored_applied),
                    "resolved": int(stored_resolved or 0),
                }
                self._conn.commit()
                return ("replayed" if hmac.compare_digest(str(stored_choice), choice) else "conflict"), receipt
            receipt = {
                "run_id": run_id,
                "request_id": request_id,
                "choice": choice,
                "applied": bool(applied),
                "resolved": max(0, int(resolved)),
            }
            self._conn.execute(
                """UPDATE run_approvals
                      SET state='resolved',choice=?,applied=?,resolved=?,responded_at=?
                    WHERE run_id=? AND request_id=? AND state='pending'""",
                (choice, int(applied), receipt["resolved"], time.time(), run_id, request_id),
            )
            self._conn.commit()
            return "created", receipt

    def mark_approval_applied(self, run_id: str, request_id: str, *, resolved: int) -> bool:
        """Record that a durable decision was delivered to the native waiter."""
        now = time.time()
        with self._lock:
            changed = self._conn.execute(
                """UPDATE run_approvals
                      SET applied=1,resolved=MAX(COALESCE(resolved,0),?),applied_at=?
                    WHERE run_id=? AND request_id=? AND state='resolved'""",
                (max(0, int(resolved)), now, run_id, request_id),
            ).rowcount
            self._conn.commit()
        return changed == 1

    def mark_tool_dispatching(self, run_id: str, request_id: str) -> bool:
        """Close the safe replay window before the approved tool can have effects."""
        now = time.time()
        with self._lock:
            changed = self._conn.execute(
                """UPDATE run_approvals
                      SET dispatch_state='dispatching',dispatch_started_at=?
                    WHERE run_id=? AND request_id=? AND state='resolved'
                      AND tool_json IS NOT NULL AND dispatch_state='pending'""",
                (now, run_id, request_id),
            ).rowcount
            self._conn.commit()
        return changed == 1

    def reset_tool_dispatching(self, run_id: str, request_id: str) -> bool:
        """Re-open a fence only when delivery proved that no waiter was woken."""
        with self._lock:
            changed = self._conn.execute(
                """UPDATE run_approvals
                      SET dispatch_state='pending',dispatch_started_at=NULL
                    WHERE run_id=? AND request_id=? AND dispatch_state='dispatching'
                      AND result_json IS NULL""",
                (run_id, request_id),
            ).rowcount
            self._conn.commit()
        return changed == 1

    def mark_tool_completed(
        self, run_id: str, tool_call_id: str, result: Dict[str, Any]
    ) -> bool:
        """Commit the exact tool result before any continuation is announced."""
        encoded_result = _encode_status(dict(result))
        now = time.time()
        with self._immediate_txn():
            rows = self._conn.execute(
                """SELECT request_id,tool_json,dispatch_state,result_json
                     FROM run_approvals
                    WHERE run_id=? AND tool_json IS NOT NULL
                    ORDER BY created_at DESC""",
                (run_id,),
            ).fetchall()
            match = next(
                (
                    row
                    for row in rows
                    if str(json.loads(row[1]).get("tool_call_id") or "") == tool_call_id
                ),
                None,
            )
            if match is None:
                self._conn.commit()
                return False
            request_id, _, dispatch_state, stored_result = match
            if dispatch_state == "completed":
                self._conn.commit()
                return bool(
                    stored_result is not None
                    and hmac.compare_digest(str(stored_result), encoded_result)
                )
            if dispatch_state != "dispatching":
                self._conn.commit()
                return False
            self._conn.execute(
                """UPDATE run_approvals
                      SET dispatch_state='completed',result_json=?,completed_at=?
                    WHERE run_id=? AND request_id=? AND dispatch_state='dispatching'""",
                (encoded_result, now, run_id, request_id),
            )
            self._conn.execute(
                "UPDATE run_recovery SET state='tool_completed' WHERE parent_run_id=?",
                (run_id,),
            )
            self._conn.commit()
            return True

    def mark_tool_skipped(
        self, run_id: str, request_id: str, result: Dict[str, Any]
    ) -> bool:
        """Commit a durable non-execution result for a denied frozen call."""
        encoded_result = _encode_status(dict(result))
        now = time.time()
        with self._immediate_txn():
            changed = self._conn.execute(
                """UPDATE run_approvals
                      SET dispatch_state='completed',result_json=?,completed_at=?
                    WHERE run_id=? AND request_id=? AND state='resolved'
                      AND choice='deny' AND dispatch_state='pending'""",
                (encoded_result, now, run_id, request_id),
            ).rowcount
            if changed:
                self._conn.execute(
                    "UPDATE run_recovery SET state='tool_completed' WHERE parent_run_id=?",
                    (run_id,),
                )
            self._conn.commit()
        return changed == 1

    def reserve_recovery_successor(
        self,
        scope: str,
        parent_run_id: str,
        *,
        successor_run_id: str,
        owner_pid: int = 0,
        owner_started: int = 0,
    ) -> Dict[str, Any]:
        """Atomically reserve the sole continuation for a decided, frozen approval.

        A dispatch with no committed result is deliberately not replayable: the
        external effect may already have occurred even though this store cannot
        prove it.
        """
        now = time.time()
        with self._immediate_txn():
            parent = self._conn.execute(
                """SELECT idempotency_key,fingerprint,retention_until,status_json
                     FROM run_idempotency WHERE scope=? AND run_id=?""",
                (scope, parent_run_id),
            ).fetchone()
            if parent is None:
                self._conn.commit()
                raise KeyError(parent_run_id)
            existing = self._conn.execute(
                "SELECT successor_run_id,request_id,state FROM run_recovery WHERE parent_run_id=?",
                (parent_run_id,),
            ).fetchone()
            if existing is not None:
                plan = self._recovery_plan_locked(parent_run_id, existing[1])
                dynamic = self._conn.execute(
                    """SELECT dispatch_state,result_json FROM run_approvals
                         WHERE run_id=? AND request_id=?""",
                    (parent_run_id, existing[1]),
                ).fetchone()
                state = str(existing[2])
                if dynamic is not None and dynamic[0] == "dispatching" and dynamic[1] is None:
                    prior_state = state
                    state = "unrecoverable"
                    failure = {
                        "run_id": str(existing[0]),
                        "status": "unrecoverable",
                        "intervention_reason": "tool_effect_uncertain",
                        "parent_run_id": parent_run_id,
                    }
                    self._conn.execute(
                        "UPDATE run_idempotency SET status_json=?,updated_at=? WHERE run_id=?",
                        (_encode_status(failure), now, existing[0]),
                    )
                    self._conn.execute(
                        "UPDATE run_recovery SET state=? WHERE parent_run_id=?",
                        (state, parent_run_id),
                    )
                    if prior_state != "unrecoverable":
                        self._append_event_locked(
                            str(existing[0]),
                            {
                                "event": "run.unrecoverable",
                                "run_id": str(existing[0]),
                                "timestamp": now,
                                "parent_run_id": parent_run_id,
                                "intervention_reason": "tool_effect_uncertain",
                            },
                            now,
                        )
                elif dynamic is not None and dynamic[0] == "completed":
                    state = "tool_completed"
                    self._conn.execute(
                        "UPDATE run_recovery SET state=? WHERE parent_run_id=?",
                        (state, parent_run_id),
                    )
                self._conn.commit()
                return {
                    **plan,
                    "created": False,
                    "scope": scope,
                    "parent_run_id": parent_run_id,
                    "successor_run_id": str(existing[0]),
                    "state": state,
                }
            approval = self._conn.execute(
                """SELECT request_id,choice,dispatch_state,tool_json,result_json
                     FROM run_approvals
                    WHERE run_id=? AND state='resolved' AND tool_json IS NOT NULL
                    ORDER BY responded_at DESC,created_at DESC LIMIT 1""",
                (parent_run_id,),
            ).fetchone()
            launch = self._conn.execute(
                "SELECT launch_json FROM run_launches WHERE run_id=?", (parent_run_id,)
            ).fetchone()
            if approval is None or launch is None:
                self._conn.commit()
                raise ValueError("run has no frozen decided approval to recover")
            request_id, choice, dispatch_state, _, result_json = approval
            try:
                parent_status = str(json.loads(parent[3]).get("status") or "")
            except Exception:
                parent_status = ""
            if parent_status == "unrecoverable":
                self._conn.commit()
                return {
                    **self._recovery_plan_values(approval, launch[0]),
                    "created": False,
                    "scope": scope,
                    "parent_run_id": parent_run_id,
                    "successor_run_id": None,
                    "state": "unrecoverable",
                }
            if dispatch_state == "dispatching" and result_json is None:
                status = {
                    "run_id": parent_run_id,
                    "status": "unrecoverable",
                    "intervention_reason": "tool_effect_uncertain",
                }
                self._conn.execute(
                    "UPDATE run_idempotency SET status_json=?,updated_at=? WHERE run_id=?",
                    (_encode_status(status), now, parent_run_id),
                )
                self._append_event_locked(
                    parent_run_id,
                    {
                        "event": "run.unrecoverable",
                        "run_id": parent_run_id,
                        "timestamp": now,
                        "intervention_reason": "tool_effect_uncertain",
                    },
                    now,
                )
                self._conn.commit()
                return {
                    **self._recovery_plan_values(approval, launch[0]),
                    "created": False,
                    "scope": scope,
                    "parent_run_id": parent_run_id,
                    "successor_run_id": None,
                    "state": "unrecoverable",
                }
            state = "tool_completed" if dispatch_state == "completed" else "recovery_pending"
            successor_status = {
                "run_id": successor_run_id,
                "status": "recovery_pending",
                "parent_run_id": parent_run_id,
            }
            recovery_key = f"\x00recovery:{parent_run_id}:{request_id}"
            self._conn.execute(
                """INSERT INTO run_idempotency(
                       scope,idempotency_key,fingerprint,run_id,status_json,
                       owner_pid,owner_started,retention_until,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    scope,
                    recovery_key,
                    f"recovery:{parent_run_id}",
                    successor_run_id,
                    _encode_status(successor_status),
                    int(owner_pid or 0),
                    int(owner_started or 0),
                    float(parent[2] or 0),
                    now,
                    now,
                ),
            )
            self._conn.execute(
                "INSERT INTO run_launches(run_id,launch_json,created_at,updated_at) VALUES(?,?,?,?)",
                (successor_run_id, launch[0], now, now),
            )
            self._conn.execute(
                """INSERT INTO run_recovery(
                       parent_run_id,successor_run_id,request_id,state,created_at
                   ) VALUES(?,?,?,?,?)""",
                (parent_run_id, successor_run_id, request_id, state, now),
            )
            parent_status = {
                "run_id": parent_run_id,
                "status": "superseded",
                "successor_run_id": successor_run_id,
            }
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?,updated_at=? WHERE run_id=?",
                (_encode_status(parent_status), now, parent_run_id),
            )
            self._append_event_locked(
                parent_run_id,
                {
                    "event": "run.superseded",
                    "run_id": parent_run_id,
                    "timestamp": now,
                    "successor_run_id": successor_run_id,
                },
                now,
            )
            self._append_event_locked(
                successor_run_id,
                {
                    "event": "run.recovery_pending",
                    "run_id": successor_run_id,
                    "timestamp": now,
                    "parent_run_id": parent_run_id,
                },
                now,
            )
            plan = self._recovery_plan_values(approval, launch[0])
            self._conn.commit()
            return {
                **plan,
                "created": True,
                "scope": scope,
                "parent_run_id": parent_run_id,
                "successor_run_id": successor_run_id,
                "state": state,
            }

    def _recovery_plan_locked(self, parent_run_id: str, request_id: str) -> Dict[str, Any]:
        approval = self._conn.execute(
            """SELECT request_id,choice,dispatch_state,tool_json,result_json
                 FROM run_approvals WHERE run_id=? AND request_id=?""",
            (parent_run_id, request_id),
        ).fetchone()
        launch = self._conn.execute(
            "SELECT launch_json FROM run_launches WHERE run_id=?", (parent_run_id,)
        ).fetchone()
        if approval is None or launch is None:
            raise ValueError("stored recovery plan is incomplete")
        return self._recovery_plan_values(approval, launch[0])

    @staticmethod
    def _recovery_plan_values(approval, launch_json: str) -> Dict[str, Any]:
        request_id, choice, _, tool_json, result_json = approval
        plan = {
            "launch": json.loads(launch_json),
            "tool": json.loads(tool_json),
            "decision": {"request_id": str(request_id), "choice": str(choice)},
        }
        if result_json is not None:
            plan["tool_result"] = json.loads(result_json)
        return plan

    def close(self) -> None:
        with self._lock:
            self._conn.close()
