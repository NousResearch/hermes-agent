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
    "room_authority_epoch": "INTEGER",
    # Public Idempotency-Keys are not legacy room rows. Room reserves stamp an authority;
    # a reserve without one is a client key and must stay eligible for prune.
    "client_replay": "INTEGER NOT NULL DEFAULT 0"}


def _encode_status(status: dict[str, Any]) -> str:
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


def _room_generation_key(key: str) -> bool:
    """True for ``room:{task}:{generation}`` identities the room client sends."""
    if not isinstance(key, str) or key.count(":") < 2 or not key.startswith("room:"):
        return False
    generation = key.rsplit(":", 1)[-1]
    return generation.isdigit() and int(generation) >= 1


def terminal_observation_expired(updated_at, retention_until, acknowledged_at, now):
    """A settled receipt outlives both its recovery window and verified Status grant."""
    if acknowledged_at is not None and acknowledged_at >= updated_at:
        if acknowledged_at + RunIdempotencyStore.ACKNOWLEDGED_RETENTION_SECONDS <= now:
            return True
    return max(retention_until, updated_at + RunIdempotencyStore.RETENTION_SECONDS) <= now


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
    def __init__(self, db_path: str | None = None):
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

    def _lineage_attempt(self, room_authority, key):
        """Find this generation under the canonical room authority, across epoch/home scopes.

        Returns ``("row", scope, record_row)``, ``("fence", None, fenced_row)``, or ``(None, None, None)``.
        ``record_row`` matches ``_SELECT_BY_KEY`` (fingerprint, run_id, status_json, owners, updated_at).
        """
        if room_authority is None:
            return None, None, None
        from gateway.platforms.api_server_run_authority import canonical
        authority_key = canonical(self._conn, room_authority)[0]
        # Prefer a real receipt over an absent barrier when a pre-fix duplicate pair exists.
        found = self._conn.execute(
            """SELECT scope, fingerprint, run_id, status_json, owner_pid, owner_started, updated_at
               FROM run_idempotency WHERE room_authority_key=? AND idempotency_key=?
               ORDER BY (fingerprint = '') ASC, updated_at DESC""",
            (authority_key, key)).fetchone()
        if found is not None:
            return "row", found[0], found[1:]
        fenced = self._conn.execute(
            """SELECT run_id, status_json FROM run_room_fenced_attempts
               WHERE authority_key=? AND idempotency_key=?""",
            (authority_key, key)).fetchone()
        if fenced is not None:
            return "fence", None, fenced
        return None, None, None

    def _replay_lineage(self, kind, lineage_scope, payload, key, *, cancel_if_missing):
        """Replay or stop a generation already owned by this authority lineage."""
        if kind == "fence":
            return "reused", _record(payload[0], payload[1], 0, 0, time.time())
        if cancel_if_missing:
            self._stop_generation(lineage_scope, key)
        # The successor grant's body hash differs because its authority fields changed.
        # The idempotency key is still this generation: return the stored run.
        return "reused", _record(*payload[1:])

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str, status: dict[str, Any], *,
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0,
                cancel_if_missing: bool = False, room_authority=None, _authorize=None):
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
                    self._stop_generation(scope, key)
                self._conn.commit()
                return ("reused", _record(*row[1:])) if cancel_if_missing else _outcome(row, fingerprint)
            from gateway.platforms.api_server_run_authority import superseded
            if superseded(self._conn, room_authority):
                self._conn.commit()
                return "authority_retired", None
            if _authorize is not None and not _authorize():
                self._conn.commit()
                return "authority_retired", None
            kind, lineage_scope, payload = self._lineage_attempt(room_authority, key)
            if kind is not None:
                outcome, record = self._replay_lineage(
                    kind, lineage_scope, payload, key, cancel_if_missing=cancel_if_missing)
                self._conn.commit()
                return outcome, record
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,created_at,updated_at,client_replay"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, now, now, 0 if room_authority is not None else 1))
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
            lineage = None if row is not None or retired else self._lineage_attempt(room_authority, key)
            self._conn.commit()
        if retired:
            return "authority_retired", None
        if row is None and lineage is not None and lineage[0] is not None:
            return self._replay_lineage(lineage[0], lineage[1], lineage[2], key, cancel_if_missing=False)
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

    def commit_room_invitation(self, claims, previous, previous_home, commit_reservation):
        """Grant writer precedes this lock; keep admissions fenced through both commits.

        The callback validates legacy reservations, writes the grant, and commits its
        writer before returning. A failed grant commit never changes this store's floor.
        """
        from gateway.platforms.api_server_run_authority import (
            canonical, namespace_matches, observe, room_authority, room_namespace, room_run_scope,
            successor, superseded)
        from gateway.platforms.api_server_room_origins import accepts, retained
        authority, namespace = room_authority(claims), room_namespace(claims)
        with self._immediate_txn():
            candidate = successor(self._conn, authority, previous)
            if (not namespace_matches(self._conn, namespace, candidate) or superseded(self._conn, candidate)
                    or not accepts(self._conn, claims, previous_home)):
                raise ValueError("room authority has already advanced")
            known = retained(self._conn, claims) is not None or self._conn.execute(
                "SELECT 1 FROM run_room_authorities WHERE authority_key=?",
                (canonical(self._conn, authority)[0],)).fetchone() is not None
            commit_reservation(known)
            if not observe(self._conn, room_run_scope(claims), authority, previous, previous_home, namespace, claims):
                raise ValueError("room authority has already advanced")
            self._conn.commit()

    def retire_room_authority(self, scope, authority):
        from gateway.platforms.api_server_run_authority import retire
        with self._immediate_txn():
            retire(self._conn, scope, authority)
            self._conn.commit()

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, run_id, status_json, stop_requested, room_authority_key,
                      updated_at, retention_until, acknowledged_at, client_replay
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        retired_keys = set()
        for (stale_scope, stale_key, stale_run, stale_status, stop_requested, authority_key,
             updated_at, row_retention, acknowledged_at, client_replay) in stale:
            # A room row written before room_authority_key existed cannot be fenced yet.
            # Deleting it makes the same generation admissible. Leave it until a successor
            # observe stamps the predecessor scope. A client key that merely looks like
            # ``room:{id}:{generation}`` is client_replay and follows the normal TTL.
            if not authority_key and not client_replay and _room_generation_key(stale_key):
                continue
            if authority_key:
                retired_keys.add(authority_key)
            try:
                status = json.loads(stale_status)
                # A renewed grant may still carry the same generation after normal replay TTL.
                # Never turn proof of non-admission back into an admissible absent key.
                # A stamped room receipt also outlives the grant's status horizon: compact
                # already waits for terminal_observation_expired, and prune must not discard
                # the only copy of the reply sooner than that.
                terminal = (status.get("status") in TERMINAL_STATUSES
                            and not status.get("admission_cancelled") and not stop_requested)
                if terminal and authority_key and not terminal_observation_expired(
                        float(updated_at or 0), float(row_retention or 0), acknowledged_at, now):
                    terminal = False
            except Exception:
                terminal = False
            if terminal:
                if authority_key:
                    from gateway.platforms.api_server_run_authority import remember_fenced_attempt
                    remember_fenced_attempt(self._conn, authority_key, stale_key, stale_run, stale_status)
                self._conn.execute(
                    "DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (stale_scope, stale_key))
        from gateway.platforms.api_server_run_authority import compact
        for authority_key in retired_keys:
            compact(self._conn, authority_key, now=now)


    def _stop_generation(self, scope, key):
        """Set stop_requested on every row of this generation, including a pre-fix duplicate."""
        self._conn.execute(
            "UPDATE run_idempotency SET stop_requested=1 WHERE scope=? AND idempotency_key=?",
            (scope, key))
        self._conn.execute(
            """UPDATE run_idempotency SET stop_requested=1
               WHERE idempotency_key=? AND room_authority_key IS NOT NULL
                 AND room_authority_key=(
                   SELECT room_authority_key FROM run_idempotency
                   WHERE scope=? AND idempotency_key=?)""",
            (key, scope, key))

    def _current_authority_key(self, room_authority):
        """Canonical key for a grant that is still allowed to see this room, else None."""
        if room_authority is None:
            return None
        from gateway.platforms.api_server_run_authority import canonical, superseded
        if superseded(self._conn, room_authority):
            return None
        return canonical(self._conn, room_authority)[0]

    def _scope_owning_run(self, scope, run_id, room_authority):
        """Scope that stores ``run_id``, following the canonical authority when the grant moved."""
        if self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone():
            return scope
        authority_key = self._current_authority_key(room_authority)
        if authority_key is None:
            return None
        found = self._conn.execute(
            "SELECT scope FROM run_idempotency WHERE room_authority_key=? AND run_id=?",
            (authority_key, run_id)).fetchone()
        return None if found is None else found[0]

    def _fenced_run_status(self, run_id, room_authority):
        authority_key = self._current_authority_key(room_authority)
        if authority_key is None:
            return None
        row = self._conn.execute(
            """SELECT status_json FROM run_room_fenced_attempts
               WHERE authority_key=? AND run_id=?""",
            (authority_key, run_id)).fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, row[0], 0, 0, time.time()).items() if k != "run_id"}

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0,
                       room_authority=None) -> dict[str, Any] | None:
        """Load one durable run status for this grant, including a successor of the admitting scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            stored = self._scope_owning_run(scope, run_id, room_authority)
            if stored is None:
                return self._fenced_run_status(run_id, room_authority)
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_RUN, (retention_until, stored, run_id))
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (stored, run_id)).fetchone()
            if row is None:
                # The row can be pruned by another process after the scope lookup and
                # before this read. The fence written first is the remaining receipt.
                return self._fenced_run_status(run_id, room_authority)
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float, *, room_authority=None) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            stored = self._scope_owning_run(scope, run_id, room_authority)
            if stored is None:
                return False
            changed = self._conn.execute(
                _EXTEND_RETENTION_BY_RUN, (checked_until, stored, run_id)).rowcount
            self._conn.commit()
        return changed == 1

    def owns_run(self, scope: str, run_id: str, *, room_authority=None) -> bool:
        with self._lock:
            if self._scope_owning_run(scope, run_id, room_authority) is not None:
                return True
            return self._fenced_run_status(run_id, room_authority) is not None

    def request_stop(self, scope: str, run_id: str, *, room_authority=None) -> None:
        with self._lock:
            stored = self._scope_owning_run(scope, run_id, room_authority)
            if stored is not None:
                identity = self._conn.execute(
                    "SELECT idempotency_key, room_authority_key FROM run_idempotency WHERE scope=? AND run_id=?",
                    (stored, run_id)).fetchone()
                if identity is None:
                    self._conn.execute(
                        "UPDATE run_idempotency SET stop_requested=1 WHERE scope=? AND run_id=?",
                        (stored, run_id))
                elif identity[1]:
                    # A pre-fix pair shares the authority key and idempotency key under two
                    # run ids. The receipt route must flag every one of them.
                    self._conn.execute(
                        """UPDATE run_idempotency SET stop_requested=1
                           WHERE idempotency_key=? AND room_authority_key=?""",
                        (identity[0], identity[1]))
                else:
                    self._conn.execute(
                        "UPDATE run_idempotency SET stop_requested=1 WHERE scope=? AND run_id=?",
                        (stored, run_id))
            self._conn.commit()

    def stop_requested(self, run_id: str) -> bool:
        """Read cancellation intent across processes without changing the public run status."""
        with self._lock:
            row = self._conn.execute(
                "SELECT stop_requested FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()
        return bool(row and row[0])

    def update_status(self, run_id: str, status: dict[str, Any]) -> None:
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
