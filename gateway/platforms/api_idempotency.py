"""Durable idempotency receipts for API-server agent turns.

The in-memory ``_IdempotencyCache`` in ``api_server.py`` (5-minute TTL, LRU)
is a best-effort duplicate suppressor for stateless OpenAI-format endpoints.
It cannot make any promise across a gateway restart: a client that retries
after a crash has no way to know whether the previous request already ran
tools with real side effects (terminal, APIs, messaging).

This module is the durable complement for endpoints that drive *persisted*
sessions, where a repeated turn repeats side effects. It stores a receipt in
a profile-aware SQLite database **before** the agent is constructed, and only
flips it to ``completed`` after the successful response has been serialized:

    running    key + fingerprint durably reserved; execution may be in
               progress, or the owning process may have died mid-turn.
    completed  the exact JSON response (and its response headers) recorded
               for replay.

Crash semantics are intentionally at-most-once / fail-closed: a ``running``
receipt found by a *different* store instance (i.e. after a process restart,
or from a sibling gateway process sharing the profile) is reported as
``uncertain`` and the caller must not re-execute. Operator reconciliation —
inspecting the persisted session transcript, then either retrying with a new
key or explicitly releasing the receipt via :meth:`DurableIdempotencyStore.release`
— is preferred over silently repeating agent tool calls.

Storage conventions match ``ResponseStore`` / ``state.db``: profile-aware
location under ``get_hermes_home()``, ``apply_wal_with_fallback`` for
NFS/SMB/FUSE mounts, owner-only file permissions. Unlike ``ResponseStore``
there is **no** in-memory fallback — a receipt that cannot be persisted
must prevent execution, so open/write failures raise
:class:`IdempotencyStoreUnavailable` instead of degrading.

Receipts may contain the final agent response for replay; treat the database
with the same privacy discipline as session state (it is chmod 0600 and must
never be logged wholesale).
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Route identity for receipts created by POST /api/sessions/{id}/chat.  The
# scope column keeps the table generic so other endpoints can adopt durable
# receipts later without a schema change.
SESSION_CHAT_SCOPE = "session_chat"

DEFAULT_RETENTION_HOURS = 24.0
# Floor for configured retention.  The whole point of the receipt is to
# outlive external retry windows; a sub-hour retention would quietly turn
# "replay" into "re-execute" for slow retriers.
MIN_RETENTION_HOURS = 1.0
# Bounded size: completed receipts beyond this count are evicted oldest-first
# (they are also pruned by retention age).  running receipts are NEVER
# auto-evicted — see _prune_completed_locked.
MAX_COMPLETED_RECEIPTS = 10_000
# Pathology guard so unreconciled running receipts cannot grow the store
# without bound.  Steady-state running count tracks in-flight turns; reaching
# this many means crash-orphans are accumulating and the operator needs to
# reconcile.  New keyed requests fail closed (503) rather than reserving.
MAX_ACTIVE_RUNNING_RECEIPTS = 1_000


class IdempotencyStoreUnavailable(RuntimeError):
    """The durable receipt store cannot guarantee a receipt.

    Raised on open/read/write/corruption failures.  Callers must fail closed
    for keyed requests (no agent execution) and leave unkeyed requests
    untouched.
    """


@dataclass(frozen=True)
class IdempotencyDecision:
    """Outcome of :meth:`DurableIdempotencyStore.reserve`.

    kind:
        ``reserved``           this call atomically created the ``running``
                               receipt; the caller owns execution.
        ``in_progress_local``  a ``running`` receipt owned by *this* store
                               instance exists; the caller should coalesce
                               with the process-local in-flight execution.
        ``replay``             a ``completed`` receipt with the same
                               fingerprint exists; return the stored response
                               without executing.
        ``conflict``           the key exists with a different fingerprint.
        ``uncertain``          a ``running`` receipt from another store
                               instance (process restart / sibling process),
                               or an unreadable completed receipt.  Never
                               re-execute automatically.
    """

    kind: str
    response_body: Optional[str] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_status: int = 200


class DurableIdempotencyStore:
    """SQLite-backed receipt store keyed by (scope, principal, session, key).

    Concurrency: the primary-key ``INSERT OR IGNORE`` makes the ``running``
    reservation atomic across threads and across processes sharing the
    profile's database file.  ``instance_id`` (fresh per store instance, i.e.
    per process in practice) distinguishes "running here, coalesce" from
    "running somewhere I cannot see, fail closed".
    """

    def __init__(self, db_path, *, retention_hours: float = DEFAULT_RETENTION_HOURS):
        self.instance_id = uuid.uuid4().hex
        self._retention_seconds = max(float(retention_hours), MIN_RETENTION_HOURS) * 3600.0
        self._lock = threading.Lock()
        self._db_path = Path(db_path)
        conn: Optional[sqlite3.Connection] = None
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label=self._db_path.name)
            # The reservation gates real side effects, so a commit must mean
            # "on stable storage" even at the cost of an fsync per receipt.
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS idempotency_receipts (
                    scope TEXT NOT NULL,
                    principal_hash TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('running', 'completed')),
                    owner_instance_id TEXT NOT NULL,
                    response_body TEXT,
                    response_headers TEXT,
                    response_status INTEGER,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    PRIMARY KEY (scope, principal_hash, session_id, idempotency_key)
                )"""
            )
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise IdempotencyStoreUnavailable(
                f"cannot open idempotency store at {self._db_path.name}: {exc}"
            ) from exc
        self._conn = conn
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars.

        Same rationale as ResponseStore: receipts can embed full agent
        responses, so other local users on a shared box must not read them.
        """
        for candidate in (
            self._db_path,
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug(
                    "Failed to restrict idempotency store permissions for %s",
                    candidate,
                    exc_info=True,
                )

    def _rollback_quietly(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            pass

    def _prune_completed_locked(self, now: float) -> None:
        """Retention + size bound for ``completed`` receipts only.

        ``running`` receipts are deliberately untouched regardless of age or
        count: an orphaned reservation is exactly the evidence an operator
        needs to reconcile an ambiguous turn, and evicting it would let a
        retry silently re-execute.  Caller holds the lock and commits.
        """
        cutoff = now - self._retention_seconds
        self._conn.execute(
            "DELETE FROM idempotency_receipts"
            " WHERE state='completed' AND completed_at IS NOT NULL AND completed_at < ?",
            (cutoff,),
        )
        row = self._conn.execute(
            "SELECT COUNT(*) FROM idempotency_receipts WHERE state='completed'"
        ).fetchone()
        excess = (row[0] if row else 0) - MAX_COMPLETED_RECEIPTS
        if excess > 0:
            self._conn.execute(
                "DELETE FROM idempotency_receipts WHERE rowid IN ("
                " SELECT rowid FROM idempotency_receipts WHERE state='completed'"
                " ORDER BY completed_at ASC LIMIT ?)",
                (excess,),
            )

    def reserve(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> IdempotencyDecision:
        """Atomically reserve ``running`` or classify the existing receipt.

        Must be called (and must succeed) before any agent construction or
        tool execution.  Raises :class:`IdempotencyStoreUnavailable` when the
        reservation cannot be durably recorded — callers fail closed.
        """
        now = time.time()
        with self._lock:
            try:
                self._prune_completed_locked(now)
                running_row = self._conn.execute(
                    "SELECT COUNT(*) FROM idempotency_receipts WHERE state='running'"
                ).fetchone()
                if running_row and running_row[0] >= MAX_ACTIVE_RUNNING_RECEIPTS:
                    self._rollback_quietly()
                    raise IdempotencyStoreUnavailable(
                        f"{running_row[0]} unreconciled running receipts —"
                        " refusing new reservations until the operator reconciles"
                    )
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO idempotency_receipts"
                    " (scope, principal_hash, session_id, idempotency_key,"
                    "  fingerprint, state, owner_instance_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
                    (scope, principal, session_id, idempotency_key, fingerprint, self.instance_id, now),
                )
                reserved = cur.rowcount == 1
                self._conn.commit()
                if reserved:
                    return IdempotencyDecision(kind="reserved")
                row = self._conn.execute(
                    "SELECT fingerprint, state, owner_instance_id,"
                    " response_body, response_headers, response_status"
                    " FROM idempotency_receipts"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?",
                    (scope, principal, session_id, idempotency_key),
                ).fetchone()
            except IdempotencyStoreUnavailable:
                raise
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise IdempotencyStoreUnavailable(
                    f"idempotency reservation failed: {exc}"
                ) from exc

        if row is None:
            # The row vanished between INSERT OR IGNORE and SELECT — only
            # possible via a concurrent explicit release.  Fail closed.
            return IdempotencyDecision(kind="uncertain")
        stored_fp, state, owner, body, headers_json, status = row
        if stored_fp != fingerprint:
            return IdempotencyDecision(kind="conflict")
        if state == "completed":
            if not body:
                logger.warning(
                    "Completed idempotency receipt has no stored response"
                    " (scope=%s session=%s) — reporting uncertain",
                    scope,
                    session_id,
                )
                return IdempotencyDecision(kind="uncertain")
            try:
                headers = json.loads(headers_json) if headers_json else {}
                if not isinstance(headers, dict):
                    raise ValueError("stored headers are not an object")
                headers = {str(k): str(v) for k, v in headers.items()}
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Completed idempotency receipt has unreadable headers"
                    " (scope=%s session=%s) — reporting uncertain",
                    scope,
                    session_id,
                )
                return IdempotencyDecision(kind="uncertain")
            return IdempotencyDecision(
                kind="replay",
                response_body=body,
                response_headers=headers,
                response_status=int(status) if status else 200,
            )
        if owner == self.instance_id:
            return IdempotencyDecision(kind="in_progress_local")
        return IdempotencyDecision(kind="uncertain")

    def complete(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
        response_body: str,
        response_headers: Dict[str, str],
        response_status: int = 200,
    ) -> bool:
        """Durably record the successful response for replay.

        Never raises: by the time this runs the agent has already executed,
        so the response in hand must still be delivered to the caller.  On
        failure the receipt stays ``running`` and later retries with the same
        key fail closed as ``uncertain`` — at-most-once is preserved at the
        cost of losing replay for this key.
        """
        now = time.time()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE idempotency_receipts"
                    " SET state='completed', response_body=?, response_headers=?,"
                    "     response_status=?, completed_at=?"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?"
                    "   AND state='running'",
                    (
                        response_body,
                        json.dumps(response_headers or {}),
                        int(response_status),
                        now,
                        scope,
                        principal,
                        session_id,
                        idempotency_key,
                    ),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self._rollback_quietly()
            logger.warning(
                "Failed to persist completed idempotency receipt"
                " (scope=%s session=%s): %s — the turn executed; retries with"
                " this key will fail closed as uncertain",
                scope,
                session_id,
                exc,
            )
            return False
        if cur.rowcount != 1:
            logger.warning(
                "Idempotency receipt missing or not running at completion"
                " (scope=%s session=%s) — replay will not be available",
                scope,
                session_id,
            )
            return False
        return True

    def release(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
    ) -> bool:
        """Explicit reconciliation escape hatch: delete a receipt.

        This is never called automatically.  It exists for operators (and
        tests) who have inspected the persisted session transcript, decided
        whether the ambiguous turn actually ran, and want to clear the held
        key.  Returns True when a receipt was removed.
        """
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM idempotency_receipts"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?",
                    (scope, principal, session_id, idempotency_key),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise IdempotencyStoreUnavailable(
                f"idempotency release failed: {exc}"
            ) from exc
        return cur.rowcount > 0

    def get_receipt(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
    ) -> Optional[Dict[str, object]]:
        """Read one receipt row (introspection for tests/reconciliation)."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT fingerprint, state, owner_instance_id, response_body,"
                    " response_headers, response_status, created_at, completed_at"
                    " FROM idempotency_receipts"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?",
                    (scope, principal, session_id, idempotency_key),
                ).fetchone()
        except sqlite3.Error as exc:
            raise IdempotencyStoreUnavailable(
                f"idempotency read failed: {exc}"
            ) from exc
        if row is None:
            return None
        keys = (
            "fingerprint",
            "state",
            "owner_instance_id",
            "response_body",
            "response_headers",
            "response_status",
            "created_at",
            "completed_at",
        )
        return dict(zip(keys, row))

    def count_receipts(self, *, state: Optional[str] = None) -> int:
        """Count receipts, optionally filtered by state (test/ops helper)."""
        query = "SELECT COUNT(*) FROM idempotency_receipts"
        params: tuple = ()
        if state is not None:
            query += " WHERE state=?"
            params = (state,)
        try:
            with self._lock:
                row = self._conn.execute(query, params).fetchone()
        except sqlite3.Error as exc:
            raise IdempotencyStoreUnavailable(
                f"idempotency read failed: {exc}"
            ) from exc
        return row[0] if row else 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
