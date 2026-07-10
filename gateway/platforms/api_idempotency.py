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
               progress, or the owning process may have died ambiguously.
    completed  terminal evidence that the turn executed.  While the replay
               payload is retained, retries replay the exact JSON response;
               after the payload expires (retention age, byte cap, or a
               privacy enforcement failure) the row remains as a compact
               tombstone and retries are refused with ``response_expired``.

Crash semantics are intentionally at-most-once / fail-closed:

* A ``running`` receipt found by a *different* store instance (process
  restart, sibling gateway process) is reported ``uncertain`` and the caller
  must not re-execute.
* Terminal evidence is **never deleted automatically**.  Replay payload
  bytes expire after ``retention_hours``; the (scope, principal, session
  incarnation, key, fingerprint) row survives as a tombstone so an aged
  retry can never reserve fresh and execute a second time.
* When the bounded receipt capacity is reached, reservations for NEW keys
  fail closed instead of evicting evidence.
* Cleanup is explicit only: :meth:`DurableIdempotencyStore.release` (CAS,
  dead-owner reconciliation of a single receipt) and
  :meth:`DurableIdempotencyStore.purge_completed` (operator-invoked bulk
  purge of terminal rows older than the replay window).

Storage conventions match ``ResponseStore`` / ``state.db``: profile-aware
location under ``get_hermes_home()``, ``apply_wal_with_fallback`` for
NFS/SMB/FUSE mounts.  Unlike ``ResponseStore`` there is **no** in-memory
fallback — a receipt that cannot be persisted must prevent execution, so
open/write failures raise :class:`IdempotencyStoreUnavailable` instead of
degrading.

Privacy: receipts may contain the final agent response for replay, so on
POSIX the database (and its SQLite sidecars) must be owner-only.  Permission
enforcement is a **gate**, not best-effort: if 0600 cannot be applied and
verified, opening/reserving raises :class:`IdempotencyStoreUnavailable`
(keyed execution stays at zero), and a completion falls back to a no-replay
tombstone rather than writing response bytes into a non-private file.
Windows/NTFS has no POSIX mode semantics; enforcement is explicitly skipped
there rather than falsely claimed.
"""

import json
import logging
import os
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
# Floor for configured retention.  Retention bounds how long the *replay
# payload* stays available; a sub-hour window would quietly turn "replay"
# into "response_expired" for slow retriers.  (Terminal evidence itself
# never expires — see _expire_completed_payloads_locked.)
MIN_RETENTION_HOURS = 1.0
# Pathology guard so unreconciled running receipts cannot grow the store
# without bound.  Steady-state running count tracks in-flight turns; reaching
# this many means crash-orphans are accumulating and the operator needs to
# reconcile.  New keyed requests fail closed (503) rather than reserving.
MAX_ACTIVE_RUNNING_RECEIPTS = 1_000
# Hard bound on total rows (running + completed + tombstones).  When full,
# reservations for keys that have no existing receipt fail closed —
# at-most-once evidence is never evicted to make room.  Requests for keys
# that already have a receipt are still classified normally.  Expired
# tombstones are small (payload columns NULL), so this bounds the store at
# roughly a few MB; operators reclaim space explicitly via purge_completed().
MAX_TOTAL_RECEIPTS = 25_000
# Cap on persisted replay-payload size (UTF-8 bytes of the response JSON).
# Session-chat replies are normally well under this; data-URL-inlined media
# can exceed it.  An oversized live response is still returned to the
# original caller, but the receipt is terminalized WITHOUT a payload
# (no-replay tombstone) so retries get ``response_expired`` instead of the
# store growing without bound — and never a second execution.
MAX_PERSISTED_RESPONSE_BYTES = 1_048_576


class IdempotencyStoreUnavailable(RuntimeError):
    """The durable receipt store cannot guarantee a receipt.

    Raised on open/read/write/corruption failures, on POSIX permission
    enforcement failure, and on capacity exhaustion for new keys.  Callers
    must fail closed for keyed requests (no agent execution) and leave
    unkeyed requests untouched.
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
                               fingerprint exists and its payload is still
                               retained; return the stored response without
                               executing.
        ``response_expired``   a ``completed`` receipt with the same
                               fingerprint exists but its replay payload is
                               gone (retention age, byte cap, privacy
                               fallback, or unreadable).  The turn already
                               executed — never re-execute, never replay.
        ``conflict``           the key exists with a different fingerprint
                               or a different session incarnation.
        ``uncertain``          a ``running`` receipt from another store
                               instance (process restart / sibling process).
                               Never re-execute automatically.
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
    "running somewhere I cannot see, fail closed", and — together with the
    reserved fingerprint — is the compare-and-swap token ``complete()`` and
    ``release()`` require, so a stale execution can never overwrite a
    receipt it no longer owns.

    Session incarnation: rows record the immutable creation identity of the
    target session (derived from the persisted ``sessions.started_at``, which
    is written once at INSERT and never updated).  A receipt minted against a
    session that was later deleted and recreated under the same textual ID
    can therefore never replay into the new session — the incarnation (also
    folded into the fingerprint) no longer matches and the request is
    classified ``conflict``.
    """

    def __init__(self, db_path, *, retention_hours: float = DEFAULT_RETENTION_HOURS):
        self.instance_id = uuid.uuid4().hex
        self._retention_seconds = max(float(retention_hours), MIN_RETENTION_HOURS) * 3600.0
        self._lock = threading.Lock()
        self._db_path = Path(db_path)
        conn: Optional[sqlite3.Connection] = None
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # Pre-create the DB file owner-only so there is no window where a
            # permissive umask exposes it (sqlite would create it with the
            # process default otherwise; WAL/SHM sidecars inherit the main
            # database file's mode, so locking this down first also covers
            # sidecars created later on write).
            if os.name == "posix" and not self._db_path.exists():
                fd = os.open(str(self._db_path), os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
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
                    session_incarnation TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (scope, principal_hash, session_id, idempotency_key)
                )"""
            )
            # Migration for stores created before the incarnation column
            # existed: preserve rows, add the column.  Pre-migration rows
            # carry '' which can never equal a real incarnation, so they
            # classify as conflict (safe: no cross-incarnation replay).
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(idempotency_receipts)").fetchall()
            }
            if "session_incarnation" not in columns:
                conn.execute(
                    "ALTER TABLE idempotency_receipts"
                    " ADD COLUMN session_incarnation TEXT NOT NULL DEFAULT ''"
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
        try:
            self._enforce_owner_only_permissions()
        except IdempotencyStoreUnavailable:
            try:
                conn.close()
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # Privacy gate
    # ------------------------------------------------------------------

    def _enforce_owner_only_permissions(self) -> None:
        """Enforce and verify owner-only modes on the DB and SQLite sidecars.

        The receipts can embed full agent responses, so this is a gate, not
        best-effort: on POSIX a chmod failure or a post-chmod verification
        failure raises :class:`IdempotencyStoreUnavailable` and the caller
        fails closed.  Windows/NTFS does not enforce POSIX modes — there we
        skip rather than pretend (see CONTRIBUTING.md cross-platform rules).
        """
        if os.name != "posix":
            return
        for candidate in (
            self._db_path,
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ):
            try:
                if not candidate.exists():
                    continue
                candidate.chmod(0o600)
                mode = candidate.stat().st_mode & 0o777
            except OSError as exc:
                raise IdempotencyStoreUnavailable(
                    f"cannot enforce owner-only permissions on {candidate.name}: {exc}"
                ) from exc
            if mode & 0o077:
                raise IdempotencyStoreUnavailable(
                    f"{candidate.name} remains group/world-accessible"
                    f" (mode {oct(mode)}) after chmod"
                )

    def _permissions_private(self) -> bool:
        """Non-raising wrapper used on the completion path."""
        try:
            self._enforce_owner_only_permissions()
        except IdempotencyStoreUnavailable as exc:
            logger.warning("Idempotency store privacy enforcement failed: %s", exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rollback_quietly(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            pass

    def _expire_completed_payloads_locked(self, now: float) -> None:
        """Expire replay payloads past retention — keep terminal evidence.

        Only the response bytes are dropped; the row itself (scope,
        principal, session incarnation, key, fingerprint, completed_at)
        survives as a tombstone so an aged retry classifies as
        ``response_expired`` instead of reserving fresh and executing a
        second time.  ``running`` receipts are deliberately untouched
        regardless of age.  Caller holds the lock and commits.
        """
        cutoff = now - self._retention_seconds
        self._conn.execute(
            "UPDATE idempotency_receipts"
            " SET response_body=NULL, response_headers=NULL"
            " WHERE state='completed' AND completed_at IS NOT NULL"
            "   AND completed_at < ? AND response_body IS NOT NULL",
            (cutoff,),
        )

    def _guard_capacity_locked(self) -> None:
        """Fail closed for NEW reservations when the store is at capacity.

        Evidence is never evicted to make room.  Raises after rollback so
        the caller's 503 carries no partial state.
        """
        running_row = self._conn.execute(
            "SELECT COUNT(*) FROM idempotency_receipts WHERE state='running'"
        ).fetchone()
        if running_row and running_row[0] >= MAX_ACTIVE_RUNNING_RECEIPTS:
            self._rollback_quietly()
            raise IdempotencyStoreUnavailable(
                f"{running_row[0]} unreconciled running receipts —"
                " refusing new reservations until the operator reconciles"
            )
        total_row = self._conn.execute(
            "SELECT COUNT(*) FROM idempotency_receipts"
        ).fetchone()
        if total_row and total_row[0] >= MAX_TOTAL_RECEIPTS:
            self._rollback_quietly()
            raise IdempotencyStoreUnavailable(
                f"receipt store at capacity ({total_row[0]} rows) — refusing"
                " new reservations; reclaim space explicitly with"
                " purge_completed()"
            )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def reserve(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        session_incarnation: str,
        idempotency_key: str,
        fingerprint: str,
    ) -> IdempotencyDecision:
        """Atomically reserve ``running`` or classify the existing receipt.

        Must be called (and must succeed) before any agent construction or
        tool execution.  Raises :class:`IdempotencyStoreUnavailable` when the
        reservation cannot be durably recorded, when privacy enforcement
        fails, or when the store is at capacity for new keys — callers fail
        closed.
        """
        now = time.time()
        with self._lock:
            self._enforce_owner_only_permissions()
            try:
                self._expire_completed_payloads_locked(now)
                row = self._read_receipt_row_locked(
                    scope, principal, session_id, idempotency_key
                )
                if row is None:
                    self._guard_capacity_locked()
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO idempotency_receipts"
                        " (scope, principal_hash, session_id, idempotency_key,"
                        "  fingerprint, state, owner_instance_id, created_at,"
                        "  session_incarnation)"
                        " VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                        (
                            scope,
                            principal,
                            session_id,
                            idempotency_key,
                            fingerprint,
                            self.instance_id,
                            now,
                            session_incarnation,
                        ),
                    )
                    reserved = cur.rowcount == 1
                    self._conn.commit()
                    if reserved:
                        return IdempotencyDecision(kind="reserved")
                    # Lost a cross-process race: another writer inserted the
                    # row between our SELECT and INSERT.  Classify theirs.
                    row = self._read_receipt_row_locked(
                        scope, principal, session_id, idempotency_key
                    )
                else:
                    self._conn.commit()  # persist any payload expiry
            except IdempotencyStoreUnavailable:
                raise
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise IdempotencyStoreUnavailable(
                    f"idempotency reservation failed: {exc}"
                ) from exc

        if row is None:
            # The row vanished between INSERT OR IGNORE and re-read — only
            # possible via a concurrent explicit release.  Fail closed.
            return IdempotencyDecision(kind="uncertain")
        return self._classify_existing_row(
            row, fingerprint=fingerprint, session_incarnation=session_incarnation,
            scope=scope, session_id=session_id,
        )

    def _read_receipt_row_locked(self, scope, principal, session_id, idempotency_key):
        return self._conn.execute(
            "SELECT fingerprint, state, owner_instance_id,"
            " response_body, response_headers, response_status, session_incarnation"
            " FROM idempotency_receipts"
            " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?",
            (scope, principal, session_id, idempotency_key),
        ).fetchone()

    def _classify_existing_row(
        self, row, *, fingerprint: str, session_incarnation: str, scope: str, session_id: str
    ) -> IdempotencyDecision:
        stored_fp, state, owner, body, headers_json, status, stored_incarnation = row
        # Incarnation is folded into the fingerprint as well; the explicit
        # comparison is defense-in-depth so a receipt from a deleted-and-
        # recreated session can never be mistaken for the current one.
        if stored_incarnation != session_incarnation or stored_fp != fingerprint:
            return IdempotencyDecision(kind="conflict")
        if state == "completed":
            if not body:
                # Terminal tombstone: retention elapsed, byte cap exceeded,
                # or privacy fallback dropped the payload.  Execution
                # evidence stands — never replay, never re-execute.
                return IdempotencyDecision(kind="response_expired")
            try:
                headers = json.loads(headers_json) if headers_json else {}
                if not isinstance(headers, dict):
                    raise ValueError("stored headers are not an object")
                headers = {str(k): str(v) for k, v in headers.items()}
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Completed idempotency receipt has unreadable headers"
                    " (scope=%s session=%s) — treating replay payload as expired",
                    scope,
                    session_id,
                )
                return IdempotencyDecision(kind="response_expired")
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
        fingerprint: str,
        response_body: Optional[str],
        response_headers: Dict[str, str],
        response_status: int = 200,
    ) -> bool:
        """Durably terminalize the reservation this store instance owns.

        Compare-and-swap: the UPDATE only matches a ``running`` row that
        still carries the *original* ``fingerprint`` AND this instance's
        ``owner_instance_id``.  A stale execution (its receipt released by an
        operator and the key re-reserved — possibly with a new fingerprint by
        a new owner) can therefore never overwrite the replacement receipt;
        it logs and returns False instead.

        Payload policy: a response larger than ``MAX_PERSISTED_RESPONSE_BYTES``
        — or any response when owner-only permissions cannot be enforced at
        completion time — is terminalized WITHOUT the payload (no-replay
        tombstone).  The live caller still receives the real response;
        retries get ``response_expired``.

        Never raises: by the time this runs the agent has already executed,
        so the response in hand must still be delivered.  On write failure
        the receipt stays ``running`` and later retries with the same key
        fail closed as ``uncertain`` — at-most-once is preserved at the cost
        of losing replay for this key.

        Returns True only when the replay payload was durably persisted.
        """
        now = time.time()
        body = response_body
        payload_persistable = body is not None
        if body is not None and len(body.encode("utf-8")) > MAX_PERSISTED_RESPONSE_BYTES:
            logger.warning(
                "Idempotency replay payload exceeds %d bytes"
                " (scope=%s session=%s) — terminalizing without replay payload",
                MAX_PERSISTED_RESPONSE_BYTES,
                scope,
                session_id,
            )
            body = None
            payload_persistable = False
        if body is not None and not self._permissions_private():
            logger.warning(
                "Idempotency store is not private — terminalizing without"
                " replay payload (scope=%s session=%s)",
                scope,
                session_id,
            )
            body = None
            payload_persistable = False
        headers_json = json.dumps(response_headers or {}) if body is not None else None
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE idempotency_receipts"
                    " SET state='completed', response_body=?, response_headers=?,"
                    "     response_status=?, completed_at=?"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?"
                    "   AND state='running' AND fingerprint=? AND owner_instance_id=?",
                    (
                        body,
                        headers_json,
                        int(response_status),
                        now,
                        scope,
                        principal,
                        session_id,
                        idempotency_key,
                        fingerprint,
                        self.instance_id,
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
                "Idempotency receipt no longer matches this execution's"
                " reservation at completion (scope=%s session=%s) — refusing"
                " to overwrite; replay will not be available for this run",
                scope,
                session_id,
            )
            return False
        return payload_persistable

    def release(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
        fingerprint: str,
        owner_instance_id: str,
    ) -> bool:
        """Dead-owner reconciliation ONLY: delete one exact inspected receipt.

        This is never called automatically.  It exists for an operator who
        has (1) confirmed the owning process is dead, (2) inspected the
        persisted session transcript to decide whether the ambiguous turn
        ran, and (3) read the receipt via :meth:`get_receipt`.  The DELETE
        is compare-and-swap on the inspected ``fingerprint`` and
        ``owner_instance_id``: if the receipt changed since inspection (for
        example the key was already released and re-reserved by a live
        owner), the stale release matches nothing and returns False.
        """
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM idempotency_receipts"
                    " WHERE scope=? AND principal_hash=? AND session_id=? AND idempotency_key=?"
                    "   AND fingerprint=? AND owner_instance_id=?",
                    (scope, principal, session_id, idempotency_key, fingerprint, owner_instance_id),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise IdempotencyStoreUnavailable(
                f"idempotency release failed: {exc}"
            ) from exc
        return cur.rowcount > 0

    def purge_completed(self, *, older_than_hours: float) -> int:
        """Explicitly purge terminal receipts older than ``older_than_hours``.

        The ONLY way completed rows/tombstones leave the store, and it is
        never called automatically — an operator invokes it to reclaim
        capacity once the purged keys are older than every client retry
        window.  Ages below the store's replay retention are refused so
        evidence younger than the replay window cannot be destroyed by a
        typo.  ``running`` receipts are never touched.  Returns the number
        of rows removed.
        """
        min_hours = self._retention_seconds / 3600.0
        if older_than_hours < min_hours:
            raise ValueError(
                f"older_than_hours must be >= the replay retention ({min_hours}h)"
            )
        cutoff = time.time() - older_than_hours * 3600.0
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM idempotency_receipts"
                    " WHERE state='completed' AND completed_at IS NOT NULL"
                    "   AND completed_at < ?",
                    (cutoff,),
                )
                self._conn.commit()
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise IdempotencyStoreUnavailable(
                f"idempotency purge failed: {exc}"
            ) from exc
        return cur.rowcount

    def get_receipt(
        self,
        *,
        scope: str,
        principal: str,
        session_id: str,
        idempotency_key: str,
    ) -> Optional[Dict[str, object]]:
        """Read one receipt row (operator inspection for release(), tests)."""
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT fingerprint, state, owner_instance_id, response_body,"
                    " response_headers, response_status, created_at, completed_at,"
                    " session_incarnation"
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
            "session_incarnation",
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
