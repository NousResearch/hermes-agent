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

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})

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
    "acknowledged_at": "REAL",
    "stop_requested": "INTEGER NOT NULL DEFAULT 0",
    "room_authority_key": "TEXT",
    "room_authority_epoch": "INTEGER"}


def _encode_status(status: Dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True, separators=(",", ":"))


def _record(run_id, status_json, owner_pid, owner_started, updated_at) -> dict[str, Any]:
    return {
        "run_id": run_id, "status": json.loads(status_json), "owner_pid": int(owner_pid or 0),
        "owner_started": int(owner_started or 0), "updated_at": float(updated_at or 0)}


def _outcome(row, fingerprint):
    """Classify a stored ``(scope, key)`` row against the caller's fingerprint."""
    record = _record(*row[1:])
    # A cancellation fences the identity itself, including a delayed, changed payload.
    matches = record["status"].get("admission_cancelled") or hmac.compare_digest(row[0], fingerprint)
    return ("reused" if matches else "conflict"), record


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
        from gateway.platforms.api_server_run_authority import initialize
        initialize(self._conn)
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
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0,
                cancel_if_missing: bool = False, room_authority=None):
        """Reserve admission, or atomically fence an absent key and stop its existing run.

        Cancellation addresses the scoped identity, not a payload fingerprint. Its durable
        intent also covers an owner still between reservation and scheduling.
        """
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is not None:
                if retention_until:
                    self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
                if cancel_if_missing:
                    self._conn.execute(
                        "UPDATE run_idempotency SET stop_requested=1 WHERE scope=? AND idempotency_key=?",
                        (scope, key))
                self._conn.commit()
                return ("reused", _record(*row[1:])) if cancel_if_missing else _outcome(row, fingerprint)
            from gateway.platforms.api_server_run_authority import superseded
            if superseded(self._conn, room_authority):
                self._conn.commit()
                return "authority_retired", None
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, now, now))
            if room_authority is not None:
                from gateway.platforms.api_server_run_authority import canonical
                room_authority = canonical(self._conn, room_authority)
                self._conn.execute("UPDATE run_idempotency SET room_authority_key=?,room_authority_epoch=? WHERE run_id=?",
                                   (*room_authority[:2], run_id))
            self._conn.commit()
            return "created", _record(run_id, encoded, owner_pid, owner_started, now) | {"status": status}

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0, room_authority=None):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._immediate_txn():
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            from gateway.platforms.api_server_run_authority import superseded
            retired = row is None and superseded(self._conn, room_authority)
            self._conn.commit()
        if retired:
            return "authority_retired", None
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def accepts_room_authority(self, authority, previous=None, namespace=None, claims=None, previous_home=None):
        from gateway.platforms.api_server_run_authority import namespace_matches, successor, superseded
        from gateway.platforms.api_server_room_origins import accepts
        with self._lock:
            candidate = successor(self._conn, authority, previous)
            return (namespace_matches(self._conn, namespace, candidate) and not superseded(self._conn, candidate)
                    and (claims is None or accepts(self._conn, claims, previous_home)))

    def knows_room_target(self, claims):
        from gateway.platforms.api_server_room_origins import retained
        with self._lock:
            return retained(self._conn, claims) is not None

    def knows_room_authority(self, authority):
        from gateway.platforms.api_server_run_authority import canonical
        with self._lock:
            return self._conn.execute("SELECT 1 FROM run_room_authorities WHERE authority_key=?",
                                      (canonical(self._conn, authority)[0],)).fetchone() is not None

    def permits_room_retirement(self, authority):
        from gateway.platforms.api_server_run_authority import retirement_allowed
        with self._lock:
            return retirement_allowed(self._conn, authority)

    def room_authority_retired(self, authority):
        from gateway.platforms.api_server_run_authority import canonical, retirement_allowed
        with self._lock:
            if not retirement_allowed(self._conn, authority):
                return False
            authority = canonical(self._conn, authority)
            row = self._conn.execute(
                "SELECT retired_through FROM run_room_authorities WHERE authority_key=?", (authority[0],)).fetchone()
        return row is not None and authority[1] <= row[0]

    def room_origin_home(self, claims):
        from gateway.platforms.api_server_run_authority import origin_home, room_authority
        with self._lock:
            return origin_home(self._conn, room_authority(claims), claims["home_install_id"])

    def observe_room_authority(self, scope, authority, previous=None, previous_home=None, namespace=None, claims=None):
        from gateway.platforms.api_server_run_authority import observe
        with self._immediate_txn():
            current = observe(self._conn, scope, authority, previous, previous_home, namespace, claims)
            self._conn.commit()
        return current

    def retire_room_authority(self, scope, authority):
        from gateway.platforms.api_server_run_authority import retire
        with self._immediate_txn():
            retire(self._conn, scope, authority)
            self._conn.commit()

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json, stop_requested
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        for stale_scope, stale_key, stale_status, stop_requested in stale:
            try:
                status = json.loads(stale_status)
                # A renewed grant may still carry the same generation after normal replay TTL.
                # Never turn proof of non-admission back into an admissible absent key.
                terminal = (status.get("status") in TERMINAL_STATUSES
                            and not status.get("admission_cancelled") and not stop_requested)
            except Exception:
                terminal = False
            if terminal:
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

    def request_stop(self, scope: str, run_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET stop_requested=1 WHERE scope=? AND run_id=?", (scope, run_id))
            self._conn.commit()

    def stop_requested(self, run_id: str) -> bool:
        """Read cancellation intent across processes without changing the public run status."""
        with self._lock:
            row = self._conn.execute(
                "SELECT stop_requested FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()
        return bool(row and row[0])

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), time.time(), run_id))
            row = self._conn.execute("SELECT room_authority_key FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()
            if row and row[0]:
                from gateway.platforms.api_server_run_authority import compact
                compact(self._conn, row[0])
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
