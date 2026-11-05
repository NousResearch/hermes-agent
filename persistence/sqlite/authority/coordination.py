"""``CoordinationRepository`` on SQLite: the compression lock, the turn lease, the hygiene streak, session handoffs and
the delegation ledger of the bound profile's ``state.db`` (the delivery-obligation verbs come from
``DeliveryObligationVerbs``), each verb one transaction of the store or of a ledger through the store holder.

The lease verbs resolve the conversation key and judge a holder's liveness with the conversation store's bodies. The
housekeeping repositories import the store's write-guard checks (``_guarded_ids``, ``_write_guards_reject``) from
here."""

from __future__ import annotations

import json
import logging
import threading
import time
import weakref
from collections.abc import Sequence
from typing import Any, Optional

from persistence import DelegationId, JsonValue, PersistenceError, SessionId, StoreContention, StoreReplaced
from persistence.authority.coordination import (
    DELEGATION_CLAIM_STALE_SECONDS, DELEGATION_FAILURE_WINDOW_SECONDS, DELEGATION_MAX_DELIVERY_ATTEMPTS,
    DELEGATION_MAX_DURABLE_PENDING, DELEGATION_MAX_RETAINED_TERMINAL, DELEGATION_ORPHAN_STALE_SECONDS,
    DELEGATION_REPLAY_MAX_AGE_SECONDS, DELEGATION_RETENTION_SECONDS, TURN_LEASE_SWEEP_GRACE_SECONDS, DelegationRecord,
    DelegationRecovery, DeliveryState, FinishedDelegationRow, HandoffState, HandoffStatus, InflightDelegation,
    OrphanCompletion, OwnerStamp, PendingCompletion, PendingHandoff, RecordedOwner,
)

from .._backend import SQLiteStore, register_repository, translated
from ..conversation._store.compression import _session_turn_lease_key_on_conn
from ..conversation._store.core import _compression_lock_holder_process_is_dead
from ..conversation._store.write_guards import _guarded_ids, _write_guards_reject
from ..gateway._delivery_ledger import LEDGER_PATIENCE_S, DeliveryObligationVerbs

__all__ = ["SQLiteCoordinationRepository"]

# The log records keep main's logger names: the store's and the delegation tool's.
logger = logging.getLogger("hermes_state")
_delegation_logger = logging.getLogger("tools.async_delegation")

# What the compression-lock writes log and answer as their empty value: a store failure (translated) or a torn-down
# backend. A replaced store is neither: ``StoreReplaced`` propagates so the caller re-acquires the store.
_LOCK_FAILURES = (PersistenceError, RuntimeError)

# Per store holder: the writer connection ``_ledger_setup`` last ran on and the setup steps it ran there.
_setup_done: weakref.WeakKeyDictionary[Any, tuple[Any, set[Any]]] = weakref.WeakKeyDictionary()
_setup_lock = threading.Lock()

_HANDOFF_FAIL_SQL = "UPDATE sessions SET handoff_state = 'failed', handoff_error = ? WHERE "

def _claim_lease_row(conn, table: str, key_col: str, key: str, holder: str, now: float, expires_at: float,
                     stale) -> tuple[bool, Optional[str]]:
    """Single-transaction lease claim: DELETE a stale holder's row (``stale(holder,
    expires_at)``), INSERT OR IGNORE ours, then SELECT to confirm ownership (INSERT OR
    IGNORE gives no rowcount signal). Returns ``(acquired, reclaimed_holder)``."""
    reclaimed_holder = None
    row = conn.execute(f"SELECT holder, expires_at FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
    if row is not None and stale(row["holder"], row["expires_at"]):
        conn.execute(f"DELETE FROM {table} WHERE {key_col} = ? AND holder = ?", (key, row["holder"]))
        reclaimed_holder = row["holder"]
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({key_col}, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (key, holder, now, expires_at))
    owner = conn.execute(f"SELECT holder FROM {table} WHERE {key_col} = ?", (key,)).fetchone()
    return owner is not None and owner["holder"] == holder, reclaimed_holder


# The in-memory offer of each returned completion stays with the caller.
def _replay_pending(conn, rows, now: float) -> tuple[PendingCompletion, ...]:
    """Each pending ``(delegation_id, event_json, completed_at, dispatched_at)`` row as a completion to offer, or
    terminally dropped past ``DELEGATION_REPLAY_MAX_AGE_SECONDS``."""
    offered = []
    for delegation_id, payload, completed_at, dispatched_at in rows:
        age_basis = completed_at or dispatched_at
        if age_basis and (now - age_basis) > DELEGATION_REPLAY_MAX_AGE_SECONDS:
            conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL,
                          updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'""", (now, delegation_id))
            _delegation_logger.warning("Async delegation %s: pending completion is %.1fh old "
                                       "(cap %.1fh); terminally dropping the replay (result remains queryable).",
                                       delegation_id, (now - age_basis) / 3600.0,
                                       DELEGATION_REPLAY_MAX_AGE_SECONDS / 3600.0)
            continue
        offered.append(PendingCompletion(delegation_id=DelegationId(delegation_id), event=json.loads(payload),
                                         completed_at=completed_at, dispatched_at=dispatched_at))
    return tuple(offered)


def _past_replay_age(now: float, completed_at: Optional[float], dispatched_at: Optional[float]) -> bool:
    age_basis = completed_at or dispatched_at
    return bool(age_basis) and (now - age_basis) > DELEGATION_REPLAY_MAX_AGE_SECONDS


# A JSON object column as a dict: ``{}`` when empty, undecodable or not an object.
def _json_object(raw: Optional[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _exhausted_warning(delegation_id: str) -> None:
    _delegation_logger.warning("Async delegation %s exhausted its %d delivery attempts; "
                               "marking terminally dropped (result remains queryable).",
                               delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS)


class SQLiteCoordinationRepository(DeliveryObligationVerbs):
    """Compression lock, turn lease, hygiene streak, handoffs and the two ledgers of the bound profile's
    ``state.db``. The ledger verbs run through the registered writer with the ledgers' 10 s lock wait."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def _ledger_write(self, fn):
        """One ledger transaction on the store's writer (BEGIN IMMEDIATE) with the ledgers' lock wait."""
        return self._store.holder()._execute_write(fn, patience_s=LEDGER_PATIENCE_S)

    def _ledger_read(self, read):
        """``read(conn)`` on a read connection with the ledgers' lock wait."""
        with self._store.holder()._read_ctx_with_timeout(LEDGER_PATIENCE_S) as conn:
            return read(conn)

    def _ledger_setup(self, step) -> None:
        """``step(conn)`` (a ledger's table step) once per writer connection of the store, outside any transaction,
        waiting out lock contention for the ledgers' lock wait. A writer reopened on a file restored in place runs it
        again."""
        holder = self._store.holder()
        conn, done = _setup_done.get(holder, (None, set()))
        if conn is holder._conn and step in done:
            return
        with _setup_lock:
            deadline = time.monotonic() + LEDGER_PATIENCE_S
            while True:
                try:
                    with translated(), holder._lock:
                        holder._raise_if_db_replaced()
                        if holder._conn is None:
                            holder._reopen_after_close_locked(context="write")
                        conn, done = _setup_done.get(holder, (None, set()))
                        if conn is not holder._conn:
                            conn, done = holder._conn, set()
                            _setup_done[holder] = (conn, done)
                        if step not in done:
                            step(conn)
                            done.add(step)
                    return
                except StoreContention:
                    if not holder._sleep_before_write_retry(deadline, LEDGER_PATIENCE_S):
                        raise

    def _write_sql_logged(self, op: str, session_id: str, sql: str, params) -> None:
        """``_write_sql`` that logs a store failure instead of raising."""
        try:
            with translated():
                self._store.holder()._write_sql(sql, params)
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            logger.warning("%s(%s) failed: %s", op, session_id, exc)

    # ── compression lock ────────────────────────────────────────────────────

    def try_acquire_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds
        def _do(conn):
            return _claim_lease_row(
                conn, "compression_locks", "session_id", session_id, holder, now, expires_at,
                lambda h, e: e < now or _compression_lock_holder_process_is_dead(h))

        try:
            with translated():
                acquired, reclaimed_holder = self._store.holder()._execute_write(_do)
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            # False makes the caller skip compression: safe when the lock subsystem is broken.
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False
        if reclaimed_holder:
            logger.warning("Reclaimed stale compression lock for session=%s (holder=%s)",
                           session_id, reclaimed_holder)
        return bool(acquired)

    def refresh_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        """Ownership is decided by ``holder`` alone, not ``expires_at``: a live owner whose refresher stalled past
        its TTL revives its still-unclaimed row; a reclaim cannot interleave because SQLite serialises writes."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + ttl_seconds
        try:
            with translated():
                return self._store.holder()._write_rowcount(
                    "UPDATE compression_locks SET expires_at = ? WHERE session_id = ? AND holder = ?",
                    (expires_at, session_id, holder)) > 0
        except StoreReplaced:
            raise
        except _LOCK_FAILURES as exc:
            logger.warning("refresh_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def release_compression_lock(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id:
            return
        self._write_sql_logged(
            "release_compression_lock", session_id,
            "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?",
            (session_id, holder))

    def get_compression_lock_holder(self, *, session_id: SessionId) -> Optional[str]:
        if not session_id:
            return None
        with translated():
            row = self._store.holder()._read_one(
                "SELECT holder FROM compression_locks WHERE session_id = ? AND expires_at >= ?",
                (session_id, time.time()))
        return None if row is None else row[0]

    # ── turn lease ──────────────────────────────────────────────────────────

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                               patience_s: Optional[float] = None) -> bool:
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))
        def _do(conn):
            # Sweep rows that expired long ago: a holder that died without releasing leaves its row behind, and
            # nothing else revisits a conversation nobody resumes. The grace keeps the recent expiries a live
            # owner can renew from a starved refresher.
            conn.execute("DELETE FROM session_turn_leases WHERE expires_at < ?",
                         (now - TURN_LEASE_SWEEP_GRACE_SECONDS,))
            conversation_id = _session_turn_lease_key_on_conn(self._store.holder(), conn, session_id)
            return _claim_lease_row(
                conn, "session_turn_leases", "conversation_id", conversation_id, holder, now, expires_at,
                lambda h, e: float(e) <= now or _compression_lock_holder_process_is_dead(h),
            )[0]
        with translated():
            return bool(self._store.holder()._execute_write(_do, patience_s=patience_s))

    def refresh_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0,
                           patience_s: Optional[float] = None) -> bool:
        """``expires_at`` is stamped before the write-lock wait, so a caller timestamp taken before this call never
        exceeds it."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))
        def _do(conn):
            conversation_id = _session_turn_lease_key_on_conn(self._store.holder(), conn, session_id)
            return conn.execute(
                "UPDATE session_turn_leases SET expires_at = ? "
                "WHERE conversation_id = ? AND holder = ?", (expires_at, conversation_id, holder),
            ).rowcount > 0
        with translated():
            return bool(self._store.holder()._execute_write(_do, patience_s=patience_s))

    def release_turn_lease(self, *, session_id: SessionId, holder: str) -> None:
        if not session_id or not holder:
            return
        def _do(conn):
            conversation_id = _session_turn_lease_key_on_conn(self._store.holder(), conn, session_id)
            conn.execute(
                "DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                (conversation_id, holder))
        with translated():
            self._store.holder()._execute_write(_do)

    def turn_lease_expires_at(self, *, session_id: SessionId, holder: str) -> Optional[float]:
        if not session_id or not holder:
            return None
        with translated(), self._store.holder()._read_ctx() as conn:
            row = conn.execute(
                "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",
                (_session_turn_lease_key_on_conn(self._store.holder(), conn, session_id), holder),
            ).fetchone()
        return float(row[0]) if row else None

    # ── hygiene streak ──────────────────────────────────────────────────────

    def increment_hygiene_failure_streak(self, *, session_key: str) -> int:
        if not session_key:
            return 1
        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_hygiene_state (session_key, failure_streak)
                   VALUES (?, 1)
                   ON CONFLICT(session_key) DO UPDATE SET
                       failure_streak = gateway_hygiene_state.failure_streak + 1""",
                (session_key,),
            )
            row = conn.execute(
                "SELECT failure_streak FROM gateway_hygiene_state WHERE session_key = ?", (session_key,),
            ).fetchone()
            return int(row[0])
        with translated():
            return self._store.holder()._execute_write(_do)

    def reset_hygiene_failure_streak(self, *, session_key: str) -> None:
        if not session_key:
            return
        with translated():
            self._store.holder()._write_sql("DELETE FROM gateway_hygiene_state WHERE session_key = ?", (session_key,))

    # ── handoffs ────────────────────────────────────────────────────────────

    def request_handoff(self, *, session_id: SessionId, platform: str) -> bool:
        with translated():
            return self._store.holder()._write_rowcount(
                "UPDATE sessions SET handoff_state = 'pending',     handoff_platform = ?, "
                "    handoff_error = NULL WHERE id = ? AND (handoff_state IS NULL "
                "                  OR handoff_state IN ('completed', 'failed'))",
                (platform, session_id),
            ) > 0

    def claim_handoff(self, *, session_id: SessionId) -> bool:
        with translated():
            return self._store.holder()._write_rowcount(
                "UPDATE sessions SET handoff_state = 'running' WHERE id = ? AND handoff_state = 'pending'",
                (session_id,),
            ) > 0

    def complete_handoff(self, *, session_id: SessionId) -> None:
        with translated():
            self._store.holder()._write_sql(
                "UPDATE sessions SET handoff_state = 'completed', handoff_error = NULL WHERE id = ?",
                (session_id,))

    def fail_handoff(self, *, session_id: SessionId, error: str,
                     only_states: Optional[tuple[HandoffStatus, ...]] = None) -> bool:
        """``only_states`` makes the write a compare-and-swap on ``handoff_state``: a waiter that gives up passes
        ``(pending,)``, since the watcher owns the terminal state of a row it claimed."""
        states = tuple(only_states) if only_states else ()
        sql = _HANDOFF_FAIL_SQL + "id = ?" + (
            f" AND handoff_state IN ({', '.join('?' for _ in states)})" if states else "")
        with translated():
            return self._store.holder()._write_rowcount(sql, (error[:500], session_id, *states)) > 0

    def reclaim_stale_running_handoffs(self, *, error: str) -> tuple[SessionId, ...]:
        """A ``running`` row at watcher startup belongs to a previous gateway that died mid-dispatch; it is failed,
        not re-queued, because the dead gateway may already have dispatched the turn."""
        def _do(conn):
            cur = conn.execute("SELECT id FROM sessions WHERE handoff_state = 'running'")
            ids = [r[0] for r in cur.fetchall()]
            if ids:
                conn.execute(_HANDOFF_FAIL_SQL + "handoff_state = 'running'", (error[:500],))
            return ids
        try:
            with translated():
                ids = self._store.holder()._execute_write(_do) or []
        except Exception:
            # Swallow but never silently: a persistently failing reclaim leaves poisonous 'running' rows.
            logger.warning(
                "reclaim_stale_running_handoffs failed; stranded 'running' "
                "handoff rows (if any) were left in place", exc_info=True)
            return ()
        return tuple(SessionId(sid) for sid in ids)

    def get_handoff(self, *, session_id: SessionId) -> Optional[HandoffState]:
        try:
            row = self._store.holder()._read_one(
                "SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = ?",
                (session_id,))
            if not row or row["handoff_state"] is None:
                return None
            return HandoffState(state=HandoffStatus(row["handoff_state"]), platform=row["handoff_platform"],
                                error=row["handoff_error"])
        except Exception:  # health: allow BLE001 -- handoff state is advisory; a failed read answers no handoff
            return None

    def list_pending_handoffs(self) -> tuple[PendingHandoff, ...]:
        try:
            rows = self._store.holder()._read_all(
                "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved FROM sessions s "
                "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
                "WHERE s.handoff_state = 'pending' "
                "ORDER BY s.started_at ASC")
            return tuple(PendingHandoff(session_id=SessionId(r["id"]), platform=r["handoff_platform"],
                                        title=r["title"], started_at=float(r["started_at"])) for r in rows)
        except Exception:  # health: allow BLE001 -- the watcher polls again; a failed read answers none pending
            return ()

    def has_pending_handoffs(self) -> bool:
        with translated():
            return self._store.holder()._read_one(
                "SELECT 1 FROM sessions WHERE handoff_state = 'pending' LIMIT 1") is not None

    # ── delegation ledger ───────────────────────────────────────────────────

    # A dispatch replaces any earlier row of its id: running, its delivery pending, no attempts spent.
    def record_delegation_dispatch(self, *, delegation_id: DelegationId, origin_session: str,
                                   origin_ui_session_id: str, parent_session_id: Optional[SessionId],
                                   dispatched_at: float, owner: OwnerStamp, task: JsonValue,
                                   origin_session_id: str = '') -> None:
        now = time.time()
        def _do(conn):
            conn.execute("""INSERT OR REPLACE INTO async_delegations
                   (delegation_id, origin_session, origin_ui_session_id,
                    parent_session_id, state, dispatched_at, updated_at,
                    delivery_state, delivery_attempts, owner_pid,
                    owner_started_at, task_json, origin_session_id)
                   VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
                (delegation_id, origin_session, origin_ui_session_id,
                 parent_session_id, dispatched_at, now, owner.pid, owner.started_at,
                 json.dumps(task), origin_session_id))
        with translated():
            self._ledger_write(_do)

    def prune_delegation_ledger(self) -> None:
        """Bound terminal history, preferring delivered records for deletion."""
        cutoff = time.time() - DELEGATION_RETENTION_SECONDS
        def _do(conn):
            conn.execute(
                "DELETE FROM async_delegations WHERE delivery_state='delivered' AND updated_at < ?", (cutoff,))
            terminal_count = conn.execute(
                "SELECT COUNT(*) FROM async_delegations WHERE state NOT IN ('running','finalizing')").fetchone()[0]
            if terminal_count > DELEGATION_MAX_RETAINED_TERMINAL:
                conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state NOT IN ('running','finalizing')
                         ORDER BY CASE delivery_state WHEN 'delivered' THEN 0 ELSE 1 END,
                                  updated_at ASC LIMIT ?
                       )""", (terminal_count - DELEGATION_MAX_RETAINED_TERMINAL,))
            pending_count = conn.execute("""SELECT COUNT(*) FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'""").fetchone()[0]
            if pending_count > DELEGATION_MAX_DURABLE_PENDING:
                conn.execute("""DELETE FROM async_delegations WHERE delegation_id IN (
                         SELECT delegation_id FROM async_delegations
                         WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                         ORDER BY updated_at ASC LIMIT ?
                       )""", (pending_count - DELEGATION_MAX_DURABLE_PENDING,))
        with translated():
            self._ledger_write(_do)

    # The finished state, event and result; the completion then waits for delivery.
    def record_delegation_completion(self, *, delegation_id: DelegationId, state: str, completed_at: float,
                                     event: JsonValue, result: JsonValue) -> None:
        now = time.time()
        def _do(conn):
            conn.execute("""UPDATE async_delegations SET state=?, completed_at=?, updated_at=?,
                   event_json=?, result_json=?, delivery_state='pending'
                   WHERE delegation_id=?""",
                (state, completed_at, now,
                 json.dumps(event), json.dumps(result), delegation_id))
        with translated():
            self._ledger_write(_do)

    def record_unit_child(self, *, delegation_id: DelegationId, entry: JsonValue) -> None:
        """Merge one finished child of a still-running unit into the stored partial by ``task_index``; the caller
        redacts ``entry``, and the stored partial holds only entries redacted that way."""
        def _do(conn):
            row = conn.execute("SELECT result_json FROM async_delegations WHERE delegation_id=? AND state='running'",
                               (delegation_id,)).fetchone()
            if row is None:
                return
            partial = json.loads(row[0] or "{}") or {}
            results = [r for r in partial.get("results") or [] if r.get("task_index") != entry.get("task_index")]
            results.append(entry)
            conn.execute("UPDATE async_delegations SET result_json=?, updated_at=? WHERE delegation_id=? AND state='running'",
                         (json.dumps({"results": results, "partial": True}), time.time(), delegation_id))
        with translated():
            self._ledger_write(_do)

    # Each recovery lands only while its row keeps the state and ``updated_at`` the caller read; the caller probes
    # owners and builds each event outside the transaction.
    def recover_abandoned_delegations(self, *, recoveries: Sequence[DelegationRecovery]) -> int:
        now = time.time()
        def _do(conn):
            recovered = 0
            for recovery in recoveries:
                cursor = conn.execute("""UPDATE async_delegations SET state='unknown', completed_at=?,
                       updated_at=?, event_json=?, result_json=?, delivery_state='pending'
                       WHERE delegation_id=? AND state=? AND updated_at=?""",
                    (recovery.completed_at, now, json.dumps(recovery.event), json.dumps(recovery.result),
                     recovery.delegation_id, recovery.expected_state, recovery.expected_updated_at))
                recovered += cursor.rowcount
            return recovered
        with translated():
            return self._ledger_write(_do)

    # Every finished, undelivered completion to replay, oldest first; one past the replay age cap is dropped.
    def take_replayable_completions(self) -> tuple[PendingCompletion, ...]:
        now = time.time()
        def _do(conn):
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                   FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
                   ORDER BY completed_at, delegation_id""").fetchall()
            return _replay_pending(conn, rows, now)
        with translated():
            return self._ledger_write(_do)

    # The owner probe and the caller's lost copies arrive as ``dead_owners`` and ``lost_ids``; the offer runs in
    # the caller after the commit.
    def sweep_orphaned_completions(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(),
                                   dead_owners: frozenset[RecordedOwner],
                                   lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        def _do(conn):
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at,
                          owner_pid, owner_started_at, delivery_attempts
                   FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < ?
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)
                   ORDER BY completed_at, delegation_id""",
                (now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)).fetchall()
            orphans = []
            for delegation_id, payload, completed_at, dispatched_at, pid, started, attempts in rows:
                if delegation_id not in lost_ids and (
                        delegation_id in exclude_ids or RecordedOwner(pid=pid, started_at=started) not in dead_owners):
                    continue
                if (attempts or 0) >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
                    # Its last claimant died holding the final attempt; converge like release_delivery.
                    conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                                  delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
                           WHERE delegation_id=? AND delivery_state='pending'""", (now, delegation_id))
                    _exhausted_warning(delegation_id)
                    continue
                orphans.append((delegation_id, payload, completed_at, dispatched_at))
            return _replay_pending(conn, orphans, now)
        with translated():
            return self._ledger_write(_do)

    def _update_delivery(self, sql: str, params: tuple) -> bool:
        """Run one UPDATE on the ledger; True iff exactly one row changed."""
        def _do(conn):
            return conn.execute(sql, params).rowcount == 1
        with translated():
            return self._ledger_write(_do)

    def mark_delegation_delivered(self, *, delegation_id: DelegationId) -> bool:
        now = time.time()
        return self._update_delivery(
            """UPDATE async_delegations SET delivery_state='delivered', delivered_at=?, updated_at=?
               WHERE delegation_id=? AND delivery_state!='delivered'""", (now, now, delegation_id))

    # Claim a pending completion for delivery (a stale claim is taken over); each claim spends one attempt.
    def claim_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        now = time.time()
        def _do(conn):
            row = conn.execute(
                "SELECT delivery_state FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
            if row is None:
                return True  # legacy event created before durable dispatch
            cur = conn.execute("""UPDATE async_delegations SET delivery_claim=?, delivery_claimed_at=?,
                          delivery_attempts=delivery_attempts+1, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
                (claim_id, now, now, delegation_id, now - DELEGATION_CLAIM_STALE_SECONDS))
            return cur.rowcount == 1
        with translated():
            return self._ledger_write(_do)

    def release_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Attempts are counted at claim time; once the budget is spent the row converges to terminal ``dropped``."""
        now = time.time()
        def _do(conn):
            capped = conn.execute("""UPDATE async_delegations SET delivery_state='dropped',
                          delivery_claim=NULL, delivery_claimed_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=? AND delivery_attempts>=?""",
                (now, delegation_id, claim_id, DELEGATION_MAX_DELIVERY_ATTEMPTS))
            if capped.rowcount == 1:
                _exhausted_warning(delegation_id)
                return True
            cur = conn.execute("""UPDATE async_delegations SET delivery_claim=NULL,
                          delivery_claimed_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='pending'
                     AND delivery_claim=?""", (now, delegation_id, claim_id))
            return cur.rowcount == 1
        with translated():
            return self._ledger_write(_do)

    # Give the claim back and refund the attempt it spent.
    def defer_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        return self._update_delivery("""UPDATE async_delegations SET delivery_claim=NULL,
                      delivery_claimed_at=NULL, delivery_attempts=MAX(0, delivery_attempts-1),
                      updated_at=?
               WHERE delegation_id=? AND delivery_state='pending' AND delivery_claim=?""",
            (time.time(), delegation_id, claim_id))

    def drop_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        return self._update_delivery("""UPDATE async_delegations SET delivery_state='dropped',
                      updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""", (time.time(), delegation_id, claim_id))

    def complete_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        now = time.time()
        return self._update_delivery("""UPDATE async_delegations SET delivery_state='delivered',
                      delivered_at=?, updated_at=?, delivery_claim=NULL,
                      delivery_claimed_at=NULL
               WHERE delegation_id=? AND delivery_state='pending'
                 AND delivery_claim=?""", (now, now, delegation_id, claim_id))

    # A delivered completion goes back to pending (dropped once its attempts are spent); the caller offers the
    # returned completion after the commit.
    def reopen_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        now = time.time()
        def _do(conn):
            exhausted = conn.execute("""UPDATE async_delegations SET delivery_state='dropped', updated_at=?
                   WHERE delegation_id=? AND delivery_state='delivered' AND delivery_attempts>=?""",
                (now, delegation_id, DELEGATION_MAX_DELIVERY_ATTEMPTS))
            if exhausted.rowcount == 1:
                _exhausted_warning(delegation_id)
                return None
            reopened = conn.execute("""UPDATE async_delegations SET delivery_state='pending',
                          delivered_at=NULL, updated_at=?
                   WHERE delegation_id=? AND delivery_state='delivered' AND event_json IS NOT NULL""",
                (now, delegation_id))
            if reopened.rowcount != 1:
                return None
            rows = conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at
                   FROM async_delegations WHERE delegation_id=?""", (delegation_id,)).fetchall()
            offered = _replay_pending(conn, rows, now)
            return offered[0] if offered else None
        with translated():
            return self._ledger_write(_do)

    # The cleanup of a dispatch whose submit was never scheduled.
    def delete_unscheduled_delegation(self, *, delegation_id: DelegationId) -> None:
        def _do(conn):
            conn.execute("DELETE FROM async_delegations WHERE delegation_id=?", (delegation_id,))
        with translated():
            self._ledger_write(_do)

    # The caller redacts ``result``.
    def get_delegation(self, *, delegation_id: DelegationId) -> Optional[DelegationRecord]:
        def _read(conn):
            return conn.execute("""SELECT origin_session, state, dispatched_at, completed_at,
                          result_json, delivery_state, delivery_attempts,
                          origin_session_id
                   FROM async_delegations WHERE delegation_id=?""", (delegation_id,)).fetchone()
        with translated():
            row = self._ledger_read(_read)
        return None if row is None else DelegationRecord(
            delegation_id=delegation_id, origin_session=row[0], state=row[1], dispatched_at=row[2],
            completed_at=row[3], result=json.loads(row[4]) if row[4] else None,
            delivery_state=DeliveryState(row[5]), delivery_attempts=row[6], origin_session_id=row[7] or "")

    # A session's delegations finished inside the failure window, newest first; the caller expands failed
    # tasks.
    def list_finished_delegations(self, *, origin_ui_session_id: str = '', parent_session_id: str = '',
                                  limit: int = 20, now: Optional[float] = None) -> tuple[FinishedDelegationRow, ...]:
        selectors = [(col, val) for col, val in (
            ("origin_ui_session_id", origin_ui_session_id), ("parent_session_id", parent_session_id)) if val]
        if not selectors:
            return ()
        cutoff = (now if now is not None else time.time()) - DELEGATION_FAILURE_WINDOW_SECONDS
        owner_sql = " OR ".join(f"{col}=?" for col, _ in selectors)
        def _read(conn):
            return conn.execute(
                f"""SELECT delegation_id, state, dispatched_at, completed_at, task_json, result_json FROM async_delegations
                    WHERE ({owner_sql}) AND state NOT IN ('running','finalizing') AND completed_at >= ?
                    ORDER BY completed_at DESC LIMIT ?""",
                (*(val for _, val in selectors), cutoff, limit)).fetchall()
        with translated():
            rows = self._ledger_read(_read)
        return tuple(FinishedDelegationRow(
            delegation_id=DelegationId(delegation_id), state=state, dispatched_at=dispatched_at,
            completed_at=completed_at, task=_json_object(task_json), result=_json_object(result_json))
            for delegation_id, state, dispatched_at, completed_at, task_json, result_json in rows)

    # The candidates of ``recover_abandoned_delegations``, with the ``updated_at`` its conditional update
    # compares.
    def list_inflight_delegations(self) -> tuple[InflightDelegation, ...]:
        def _read(conn):
            return conn.execute("""SELECT delegation_id, origin_session, origin_ui_session_id,
                          parent_session_id, dispatched_at, owner_pid,
                          owner_started_at, task_json, origin_session_id, result_json, state, updated_at
                   FROM async_delegations WHERE state IN ('running','finalizing')""").fetchall()
        with translated():
            rows = self._ledger_read(_read)
        return tuple(InflightDelegation(
            delegation_id=DelegationId(delegation_id), origin_session=session_key, origin_ui_session_id=origin_ui,
            parent_session_id=parent_id, dispatched_at=dispatched_at, owner_pid=pid, owner_started_at=started,
            task=json.loads(task_json or "{}"), origin_session_id=origin_sid or "",
            result=json.loads(result_json) if result_json else None, state=last_state, updated_at=updated_at)
            for (delegation_id, session_key, origin_ui, parent_id, dispatched_at, pid, started, task_json, origin_sid,
                 result_json, last_state, updated_at) in rows)

    def has_replayable_completions(self) -> bool:
        def _read(conn):
            return conn.execute("""SELECT 1 FROM async_delegations
                   WHERE state != 'running' AND delivery_state='pending' AND event_json IS NOT NULL
                   LIMIT 1""").fetchone()
        with translated():
            return self._ledger_read(_read) is not None

    # The owner stamps of the completions ``sweep_orphaned_completions`` would consider.
    def list_orphan_completion_owners(self, *, now: float) -> frozenset[RecordedOwner]:
        def _read(conn):
            return conn.execute("""SELECT DISTINCT owner_pid, owner_started_at
                   FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < ?
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)""",
                (now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)).fetchall()
        with translated():
            rows = self._ledger_read(_read)
        return frozenset(RecordedOwner(pid=pid, started_at=started) for pid, started in rows)

    def list_orphan_completions(self, *, now: float) -> tuple[OrphanCompletion, ...]:
        def _read(conn):
            return conn.execute("""SELECT delegation_id, owner_pid, owner_started_at, updated_at
                   FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < ?
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)
                   ORDER BY completed_at, delegation_id""",
                (now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)).fetchall()
        with translated():
            rows = self._ledger_read(_read)
        return tuple(OrphanCompletion(delegation_id=DelegationId(delegation_id),
                                      owner=RecordedOwner(pid=pid, started_at=started), updated_at=updated_at)
                     for delegation_id, pid, started, updated_at in rows)


    # The completions ``sweep_orphaned_completions`` returns for the same arguments, read before its drops commit.
    def list_orphan_completion_offers(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(),
                                      dead_owners: frozenset[RecordedOwner],
                                      lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        def _read(conn):
            return conn.execute("""SELECT delegation_id, event_json, completed_at, dispatched_at,
                          owner_pid, owner_started_at, delivery_attempts
                   FROM async_delegations
                   WHERE state NOT IN ('running','finalizing') AND delivery_state='pending'
                     AND event_json IS NOT NULL AND updated_at < ?
                     AND (delivery_claim IS NULL OR delivery_claimed_at < ?)
                   ORDER BY completed_at, delegation_id""",
                (now - DELEGATION_ORPHAN_STALE_SECONDS, now - DELEGATION_CLAIM_STALE_SECONDS)).fetchall()
        with translated():
            rows = self._ledger_read(_read)
        return tuple(
            PendingCompletion(delegation_id=DelegationId(delegation_id), event=json.loads(payload),
                              completed_at=completed_at, dispatched_at=dispatched_at)
            for delegation_id, payload, completed_at, dispatched_at, pid, started, attempts in rows
            if (delegation_id in lost_ids or (delegation_id not in exclude_ids
                                              and RecordedOwner(pid=pid, started_at=started) in dead_owners))
            and (attempts or 0) < DELEGATION_MAX_DELIVERY_ATTEMPTS
            and not _past_replay_age(now, completed_at, dispatched_at))

    # The completion ``reopen_admitted_completion`` returns, read before the reopen commits.
    def get_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        def _read(conn):
            return conn.execute("""SELECT event_json, completed_at, dispatched_at, delivery_attempts
                   FROM async_delegations
                   WHERE delegation_id=? AND delivery_state='delivered' AND event_json IS NOT NULL""",
                (delegation_id,)).fetchone()
        with translated():
            row = self._ledger_read(_read)
        if (row is None or row[3] >= DELEGATION_MAX_DELIVERY_ATTEMPTS
                or _past_replay_age(time.time(), row[1], row[2])):
            return None
        return PendingCompletion(delegation_id=delegation_id, event=json.loads(row[0]), completed_at=row[1],
                                 dispatched_at=row[2])


register_repository("coordination", SQLiteCoordinationRepository, {"transactional_claims": True})
