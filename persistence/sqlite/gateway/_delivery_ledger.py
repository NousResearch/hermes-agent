"""The delivery-obligation verbs of the SQLite ``CoordinationRepository``: the ``delivery_obligations`` rows of the
bound profile's ``state.db``, each verb one transaction on the store's writer.

The ledger creates its own table: before a verb's transaction, the table step runs on the store's writer outside any
transaction, once per writer connection. The sweeps take the owner probe as data and claim each row guarded on its
previous owner stamp."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from typing import Any, Callable, Optional, TypeVar

from persistence.authority.coordination import (
    OBLIGATION_FLOOD_MARKER, OBLIGATION_MAX_ATTEMPTS, OBLIGATION_MAX_ROWS, OBLIGATION_RECONNECTED_MARKER,
    OBLIGATION_RETENTION_SECONDS, OBLIGATION_STALE_AFTER_SECONDS, ClaimedObligation, FailedObligationRow, OwnerStamp,
    RecordedOwner, RetryCandidate, RetryDue, flood_not_before, is_flood_error,
)
from sqlite_support import add_column_if_missing

from .._backend import translated

__all__ = ["LEDGER_PATIENCE_S", "DeliveryObligationVerbs"]

T = TypeVar("T")

# The lock wait, in seconds, of every write and read on the obligations table.
LEDGER_PATIENCE_S = 10.0


# Creates the obligations table and adds ``adapter_profile`` to a table that predates it.
def _ensure_obligations_table(conn) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT,
            adapter_profile TEXT
        )"""
    )
    if "adapter_profile" not in {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}:
        add_column_if_missing(conn, "delivery_obligations", "adapter_profile", "adapter_profile TEXT")


# Drops settled rows past retention, then the oldest rows over the row cap, settled ones first.
def _prune_unlocked(conn, now: float) -> None:
    """Retention DELETEs on the caller's open connection; must run inside the caller's transaction."""
    conn.execute(
        """DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""", (now - OBLIGATION_RETENTION_SECONDS,))
    total = conn.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[0]
    if total > OBLIGATION_MAX_ROWS:
        conn.execute(
            """DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations
                 ORDER BY CASE state
                            WHEN 'delivered' THEN 0
                            WHEN 'abandoned' THEN 1
                            ELSE 2
                          END, updated_at ASC
                 LIMIT ?)""", (total - OBLIGATION_MAX_ROWS,))


def _claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts, profile, *,
                 needs_marker: bool, runtime: bool = False, flood: bool = False,
                 last_error: Optional[str] = None) -> ClaimedObligation:
    """A claimed row for redelivery. A marked row names its own cause: ``flood`` gets the flood marker at boot or at
    runtime, a ``runtime`` reconnect replay the reconnect marker, and a boot-recovered crash the caller's restart
    marker default (``marker`` None). ``last_error`` is the row's pre-claim error."""
    marker = OBLIGATION_FLOOD_MARKER if flood else (OBLIGATION_RECONNECTED_MARKER if runtime else None)
    return ClaimedObligation(
        obligation_id=oid, session_key=session_key, platform=platform, chat_id=chat_id, thread_id=thread_id,
        content=content, needs_marker=needs_marker, marker=marker if needs_marker and marker else None,
        profile=profile, runtime_recovery=runtime, last_error=last_error or None, attempts=attempts + 1)


def _expected_profile(profile: Optional[str]) -> str:
    return "default" if not profile or profile == "default" else str(profile)


class DeliveryObligationVerbs:
    """The obligation verbs; the host class provides ``_ledger_write``, ``_ledger_read`` and ``_ledger_setup``."""

    _ledger_write: Callable[..., Any]
    _ledger_read: Callable[..., Any]
    _ledger_setup: Callable[..., Any]

    def _obligation_write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """``fn(conn)`` through ``_ledger_write``, after the table step."""
        self._ledger_setup(_ensure_obligations_table)
        return self._ledger_write(fn)

    def _obligation_read(self, read: Callable[[sqlite3.Connection], T]) -> T:
        """``read(conn)`` through ``_ledger_read``, after the table step."""
        self._ledger_setup(_ensure_obligations_table)
        return self._ledger_read(read)

    # Records a pending obligation owned by ``owner``, replacing a row with the same id, then prunes.
    def record_delivery_obligation(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str,
                                   thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None,
                                   owner: OwnerStamp) -> None:
        now, (pid, started) = time.time(), (owner.pid, owner.started_at)
        def _do(conn):
            conn.execute(
                """INSERT OR REPLACE INTO delivery_obligations
                   (obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, adapter_profile)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?)""",
                (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
                 content, now, now, pid, started, str(adapter_profile).strip() if adapter_profile else "default"))
            _prune_unlocked(conn, now)
        with translated():
            self._obligation_write(_do)

    def record_crash_left_reply(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str,
                                thread_id: Optional[str], content: str, since: float,
                                adapter_profile: Optional[str] = None) -> None:
        """Adopt a reply a killed process persisted but never ledgered: unowned and ``attempting``, a no-op when the
        same reply was ledgered since ``since``."""
        now = time.time()
        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO delivery_obligations
                   (obligation_id, session_key, platform, chat_id, thread_id,
                    content, state, attempts, created_at, updated_at,
                    owner_pid, owner_started_at, adapter_profile)
                   SELECT ?, ?, ?, ?, ?, ?, 'attempting', 0, ?, ?, NULL, NULL, ?
                   WHERE NOT EXISTS (SELECT 1 FROM delivery_obligations
                                     WHERE session_key = ? AND content = ? AND created_at >= ?)""",
                (obligation_id, session_key, platform, str(chat_id), str(thread_id) if thread_id else None,
                 content, now, now, str(adapter_profile).strip() if adapter_profile else "default",
                 session_key, content, since))
        with translated():
            self._obligation_write(_do)

    # Sets one row's state and error whatever its owner; the error keeps its first 500 characters.
    def _update_state(self, obligation_id: str, state: str, error: str = "") -> None:
        def _do(conn):
            conn.execute(
                """UPDATE delivery_obligations
                   SET state=?, updated_at=?, last_error=?
                   WHERE obligation_id=?""",
                (state, time.time(), error[:500] if error else None, obligation_id))
        with translated():
            self._obligation_write(_do)

    def mark_obligation_attempting(self, *, obligation_id: str) -> None:
        self._update_state(obligation_id, "attempting")

    def mark_obligation_delivered(self, *, obligation_id: str) -> None:
        self._update_state(obligation_id, "delivered")

    def mark_obligation_failed(self, *, obligation_id: str, error: str = '') -> None:
        self._update_state(obligation_id, "failed", error=error)

    def release_runtime_obligation_claim(self, *, obligation_id: str, error: str = '', owner: OwnerStamp) -> bool:
        """Return an unsent runtime claim to ``failed`` without spending an attempt, fail-closed to the exact
        process instance and ``attempting`` state."""
        pid, started = owner.pid, owner.started_at
        if started is None:
            return False
        def _do(conn):
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='failed', attempts=CASE
                           WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                       updated_at=?, last_error=?
                   WHERE obligation_id=? AND state='attempting'
                     AND owner_pid IS ? AND owner_started_at IS ?""",
                (time.time(), error[:500] if error else None, obligation_id, pid, started))
            return bool(cursor.rowcount)
        with translated():
            return self._obligation_write(_do)

    # The caller probes owner liveness and passes the dead owners as ``dead_owners``.
    def sweep_recoverable_obligations(self, *, now: Optional[float] = None,
                                      deliverable_platforms: Optional[frozenset[str]] = None,
                                      deliverable_targets: Optional[frozenset[tuple[str, str]]] = None,
                                      owner: OwnerStamp,
                                      dead_owners: frozenset[RecordedOwner]) -> tuple[ClaimedObligation, ...]:
        """Claim undelivered rows of dead owners: each claim re-stamps the owner, moves the row to ``attempting`` and
        spends an attempt, guarded on the previous owner stamp. Exhausted or stale rows become ``abandoned``; a
        flood row inside its wait is adopted without spending an attempt; rows of a platform or target the caller
        cannot send on are left alone."""
        now, (pid, started) = now if now is not None else time.time(), (owner.pid, owner.started_at)
        def _do(conn):
            claimed: list[ClaimedObligation] = []
            rows = conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, state, attempts, created_at,
                          owner_pid, owner_started_at, adapter_profile, last_error, updated_at
                   FROM delivery_obligations
                   WHERE state IN ('pending', 'attempting', 'failed')"""
            ).fetchall()
            for (oid, session_key, platform, chat_id, thread_id, content, state, attempts, created_at,
                 owner_pid, owner_started_at, adapter_profile, last_error, updated_at) in rows:
                if RecordedOwner(pid=owner_pid, started_at=owner_started_at) not in dead_owners:
                    continue  # a live gateway still owns this row
                if attempts >= OBLIGATION_MAX_ATTEMPTS or (now - created_at) > OBLIGATION_STALE_AFTER_SECONDS:
                    conn.execute(
                        """UPDATE delivery_obligations
                           SET state='abandoned', updated_at=? WHERE obligation_id=?""", (now, oid))
                    continue
                if ((deliverable_platforms is not None and platform not in deliverable_platforms)
                        or (deliverable_targets is not None
                            and (platform, adapter_profile) not in deliverable_targets)):
                    continue  # no adapter this boot: claiming would spend an attempt on a no-op
                flood_row = state == "failed" and is_flood_error(last_error)
                if flood_row and now < flood_not_before(updated_at, last_error):
                    # Still inside the platform's wait: adopt the dead owner's row without spending an attempt.
                    cursor = conn.execute(
                        """UPDATE delivery_obligations
                           SET owner_pid=?, owner_started_at=?,
                               adapter_profile=COALESCE(adapter_profile, 'default')
                           WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                        (pid, started, oid, owner_pid, owner_pid))
                    if cursor.rowcount:
                        claimed.append(ClaimedObligation(
                            obligation_id=oid, session_key=session_key, platform=platform, chat_id=chat_id,
                            thread_id=thread_id, content=content, profile=adapter_profile or "default",
                            attempts=attempts, adopted=True, not_before=flood_not_before(updated_at, last_error)))
                    continue
                # Every claim starts a send, so the row leaves 'pending'/'failed' for 'attempting' in the same CAS; a
                # claimed flood row also drops its stale refusal.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET owner_pid=?, owner_started_at=?, attempts=attempts+1, updated_at=?,
                           adapter_profile=COALESCE(adapter_profile, 'default'), state='attempting',
                           last_error=CASE WHEN ? THEN NULL ELSE last_error END
                       WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
                    (pid, started, now, 1 if flood_row else 0, oid, owner_pid, owner_pid))
                if cursor.rowcount:
                    # A never-claimed pending row was never sent: redeliver plainly; anything else carries a marker.
                    claimed.append(_claimed_row(oid, session_key, platform, chat_id, thread_id, content, attempts,
                                                adapter_profile or "default",
                                                needs_marker=state != "pending" or attempts > 0, flood=flood_row))
            return tuple(claimed)
        with translated():
            return self._obligation_write(_do)

    # The caller computes each row's deadline and passes it in ``due``; a row is claimed only while it is as the
    # candidate read saw it.
    def sweep_failed_obligations_for_runtime(self, platform: str, *, now: Optional[float] = None,
                                             profile: Optional[str] = None, owner: OwnerStamp,
                                             due: Sequence[RetryDue]) -> tuple[ClaimedObligation, ...]:
        """Claim this process's failed rows that are due for another send, for one adapter: only rows stamped to
        this exact process instance and profile, every update guarded by the prior owner stamp and ``failed``
        state. Claimed rows always carry a marker: the reconnect one, or the flood one."""
        now, (pid, started) = now if now is not None else time.time(), (owner.pid, owner.started_at)
        if started is None:  # PID alone cannot tell this process from a stale row left after PID reuse
            return ()
        expected_profile = _expected_profile(profile)
        def _do(conn):
            claimed: list[ClaimedObligation] = []
            rows = conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, attempts, created_at, owner_pid,
                          owner_started_at, last_error, adapter_profile, updated_at
                   FROM delivery_obligations
                   WHERE state='failed' AND platform=?""", (platform,)).fetchall()
            current = {row[0]: row for row in rows}
            for item in due:
                row = current.get(item.obligation_id)
                if row is None:
                    continue
                (oid, session_key, row_platform, chat_id, thread_id, content, attempts, created_at,
                 owner_pid, owner_started_at, last_error, adapter_profile, updated_at) = row
                # Exact process-start matching prevents PID reuse from stealing work.
                if adapter_profile != expected_profile or owner_pid != pid or owner_started_at != started:
                    continue
                if (updated_at, last_error, attempts) != (item.updated_at, item.last_error, item.attempts):
                    continue  # changed since the candidate read
                owner_guard = (now, oid, owner_pid, owner_started_at)
                if attempts >= OBLIGATION_MAX_ATTEMPTS or (now - created_at) > OBLIGATION_STALE_AFTER_SECONDS:
                    conn.execute(
                        """UPDATE delivery_obligations
                           SET state='abandoned', updated_at=?
                           WHERE obligation_id=? AND state='failed'
                             AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
                    continue
                if now < item.not_before:
                    continue  # the platform's wait or the backoff has not passed
                # The claim clears the stale error: this is a fresh attempt.
                cursor = conn.execute(
                    """UPDATE delivery_obligations
                       SET state='attempting', attempts=attempts+1, updated_at=?, last_error=NULL
                       WHERE obligation_id=? AND state='failed'
                         AND owner_pid IS ? AND owner_started_at IS ?""", owner_guard)
                if cursor.rowcount:
                    claimed.append(_claimed_row(oid, session_key, row_platform, chat_id, thread_id, content,
                                                attempts, adapter_profile, needs_marker=True, runtime=True,
                                                flood=is_flood_error(last_error), last_error=last_error))
            return tuple(claimed)
        with translated():
            return self._obligation_write(_do)

    # The read behind the retry schedule; the caller aggregates it per adapter.
    def list_owned_failed_obligations(self, *, owner: OwnerStamp) -> tuple[FailedObligationRow, ...]:
        pid, started = owner.pid, owner.started_at
        if started is None:
            return ()
        def _read(conn):
            return conn.execute(
                """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
                   FROM delivery_obligations
                   WHERE state='failed' AND owner_pid IS ? AND owner_started_at IS ?""", (pid, started)).fetchall()
        with translated():
            rows = self._obligation_read(_read)
        return tuple(FailedObligationRow(platform=platform, adapter_profile=adapter_profile, updated_at=updated_at,
                                         last_error=last_error, attempts=attempts, created_at=created_at)
                     for platform, adapter_profile, updated_at, last_error, attempts, created_at in rows)

    # The owner stamps of rows still awaiting delivery: pending, attempting or failed.
    def list_obligation_owners(self) -> frozenset[RecordedOwner]:
        def _read(conn):
            return conn.execute(
                """SELECT DISTINCT owner_pid, owner_started_at
                   FROM delivery_obligations
                   WHERE state IN ('pending', 'attempting', 'failed')""").fetchall()
        with translated():
            rows = self._obligation_read(_read)
        return frozenset(RecordedOwner(pid=pid, started_at=started) for pid, started in rows)

    # The candidate read of ``sweep_failed_obligations_for_runtime``, kept to this process instance and profile.
    def list_runtime_retry_candidates(self, platform: str, *, profile: Optional[str] = None,
                                      owner: OwnerStamp) -> tuple[RetryCandidate, ...]:
        pid, started = owner.pid, owner.started_at
        if started is None:
            return ()
        expected_profile = _expected_profile(profile)
        def _read(conn):
            return conn.execute(
                """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                          content, attempts, created_at, owner_pid,
                          owner_started_at, last_error, adapter_profile, updated_at
                   FROM delivery_obligations
                   WHERE state='failed' AND platform=?""", (platform,)).fetchall()
        with translated():
            rows = self._obligation_read(_read)
        return tuple(RetryCandidate(obligation_id=oid, updated_at=updated_at, last_error=last_error, attempts=attempts)
                     for (oid, _session_key, _platform, _chat_id, _thread_id, _content, attempts, _created_at,
                          owner_pid, owner_started_at, last_error, adapter_profile, updated_at) in rows
                     if adapter_profile == expected_profile and owner_pid == pid and owner_started_at == started)
