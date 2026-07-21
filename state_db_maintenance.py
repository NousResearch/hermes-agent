"""Bounded background maintenance for Hermes' shared SQLite state store.

Foreground state writes only record durable FTS debt in their existing
transaction and wake this coordinator after commit.  A process-level worker
then performs short FTS5 merge slices and passive WAL checkpoints on its own
connection.  A durable lease keeps multiple Hermes processes from merging the
same database concurrently.
"""

from __future__ import annotations

import logging
import os
import random
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from threading import Thread as _MaintenanceThread
from typing import Any, Optional

logger = logging.getLogger(__name__)


MAINTENANCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS state_maintenance_jobs (
    job_name TEXT PRIMARY KEY,
    requested_seq INTEGER NOT NULL DEFAULT 0,
    cycle_target_seq INTEGER NOT NULL DEFAULT 0,
    completed_seq INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'idle'
        CHECK (state IN ('idle', 'running')),
    base_phase TEXT NOT NULL DEFAULT 'start'
        CHECK (base_phase IN ('start', 'continue', 'done', 'missing')),
    trigram_phase TEXT NOT NULL DEFAULT 'start'
        CHECK (trigram_phase IN ('start', 'continue', 'done', 'missing')),
    next_table INTEGER NOT NULL DEFAULT 0
        CHECK (next_table IN (0, 1)),
    force_run INTEGER NOT NULL DEFAULT 1
        CHECK (force_run IN (0, 1)),
    lease_owner TEXT,
    lease_expires_at REAL NOT NULL DEFAULT 0,
    not_before REAL NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_error_code TEXT,
    updated_at REAL NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO state_maintenance_jobs
    (job_name, requested_seq)
VALUES ('fts_merge', 0);
"""

_FTS_JOB = "fts_merge"
_FTS_TABLES = ("messages_fts", "messages_fts_trigram")
_PHASE_COLUMNS = ("base_phase", "trigram_phase")
_DEBOUNCE_S = 0.250


def record_fts_maintenance_debt(
    conn: sqlite3.Connection, *, write_units: int
) -> None:
    """Record message-row maintenance debt in the caller's transaction."""
    now = time.time()
    units = max(1, int(write_units))
    conn.execute(
        """INSERT INTO state_maintenance_jobs (
               job_name, requested_seq, not_before, updated_at
           ) VALUES (?, ?, ?, ?)
           ON CONFLICT(job_name) DO UPDATE SET
               not_before = CASE
                   WHEN state = 'idle' AND not_before = 0
                   THEN excluded.not_before
                   ELSE not_before
               END,
               requested_seq = requested_seq + excluded.requested_seq,
               updated_at = excluded.updated_at""",
        (_FTS_JOB, units, now + _DEBOUNCE_S, now),
    )


class StateDbMaintenanceCoordinator:
    """One bounded background-maintenance worker per canonical database."""

    _LEASE_S = 15.0
    _MIN_MERGE_PAGES = 8
    _MAX_MERGE_PAGES = 256
    _INITIAL_MERGE_PAGES = 64
    _MAX_SLICES_PER_BURST = 4
    _MAX_BURST_S = 0.250
    _INTER_SLICE_S = 0.010
    _MAX_GATE_DEFER_S = 1.0
    _WORKER_ERROR_BACKOFF_S = 1.0

    def __init__(
        self,
        *,
        db_path: Path,
        db_key: str,
        write_gate: Any,
        fts_tables: tuple[str, ...],
        merge_interval: int,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_key = db_key
        self._write_gate = write_gate
        self._fts_tables = frozenset(fts_tables)
        self._merge_interval = max(1, int(merge_interval))
        self._owner_pid = os.getpid()
        self._lease_owner = f"{self._owner_pid}:{uuid.uuid4().hex}"
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        # os.fork() must never snapshot a process while this worker is inside
        # SQLite. The child would inherit SQLite's internal mutexes without the
        # thread that owns them and could deadlock even on sqlite3.connect().
        self._fork_guard = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._checkpoint_pending = False
        self._checkpoint_requested_seq = 0
        self._checkpoint_completed_seq = 0
        self._checkpoint_write_count = 0
        self._active = False
        self._merge_pages = self._INITIAL_MERGE_PAGES
        self._gate_deferred_since: Optional[float] = None
        # Bind the worker implementation to this module.  Several gateway
        # callers replace ``server.threading.Thread`` with a synchronous test
        # double; because ``threading`` is a shared module object, looking up
        # Thread dynamically here would also replace this long-lived worker
        # and run its wait loop inline on the request thread.
        self._thread = _MaintenanceThread(
            target=self._worker_main,
            name=f"state-db-maintenance-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        self._thread.start()
        # Probe durable work left by an unclean shutdown or an older version.
        self._wake.set()

    def notify_commit(
        self,
        *,
        write_units: int,
        checkpoint_interval: int,
        fts_dirty: bool,
        merge_interval: int,
    ) -> None:
        """O(1) post-commit notification; never touches SQLite or waits."""
        if self._owner_pid != os.getpid() or self._stop.is_set():
            return
        normalized_units = max(1, int(write_units))
        normalized_interval = max(1, int(checkpoint_interval))
        wake = fts_dirty
        with self._state_lock:
            self._merge_interval = max(1, int(merge_interval))
            previous = self._checkpoint_write_count
            self._checkpoint_write_count += normalized_units
            if (
                previous // normalized_interval
                < self._checkpoint_write_count // normalized_interval
            ):
                self._checkpoint_requested_seq += 1
                self._checkpoint_pending = True
                wake = True
        if wake:
            self._wake.set()

    def stop(self, timeout_s: float = 2.0) -> bool:
        """Stop at a slice boundary without holding any database write lock."""
        if self._owner_pid != os.getpid():
            return True
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            conn = self._connection
            if conn is not None:
                try:
                    conn.interrupt()
                except Exception:
                    pass
            self._thread.join(timeout=0.250)
        return not self._thread.is_alive()

    def is_alive(self) -> bool:
        """Return whether the single maintenance worker is still running."""
        return self._thread.is_alive()

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait until durable FTS debt and in-process checkpoint work drain."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self._state_lock:
                local_idle = not self._active and not self._checkpoint_pending
            if local_idle and self._durable_job_is_idle():
                return True
            time.sleep(0.010)
        return False

    def inherited_connection_after_fork(self) -> Optional[sqlite3.Connection]:
        """Detach, but do not close, the worker connection in a fork child."""
        conn = self._connection
        self._connection = None
        return conn

    def prepare_for_fork(self) -> None:
        """Interrupt long maintenance and quiesce SQLite before os.fork()."""
        conn = self._connection
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                pass
        self._fork_guard.acquire()

    def resume_after_fork_parent(self) -> None:
        self._fork_guard.release()

    def acquire_sqlite_activity_barrier(self, timeout_s: float = 2.0) -> bool:
        """Temporarily quiesce worker SQLite I/O for an explicit checkpoint."""
        acquired = self._fork_guard.acquire(timeout=max(0.0, timeout_s))
        if acquired:
            # A force-aged slice may have closed foreground admission and then
            # yielded the SQLite activity guard before it acquired the writer
            # gate. While this barrier is held the worker cannot retry, so the
            # explicit checkpoint must clear that abandoned reservation before
            # it waits for the foreground gate.
            self._write_gate.cancel_maintenance_reservation()
        return acquired

    def release_sqlite_activity_barrier(self) -> None:
        self._fork_guard.release()

    def _durable_job_is_idle(self) -> bool:
        try:
            with self._fork_guard:
                conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro",
                    uri=True,
                    timeout=0.050,
                    isolation_level=None,
                )
                try:
                    row = conn.execute(
                        "SELECT state, requested_seq, completed_seq, force_run "
                        "FROM state_maintenance_jobs WHERE job_name = ?",
                        (_FTS_JOB,),
                    ).fetchone()
                    return bool(
                        row is not None
                        and row[0] == "idle"
                        and int(row[3]) == 0
                        and int(row[1]) - int(row[2]) < self._merge_interval
                    )
                finally:
                    conn.close()
        except sqlite3.Error:
            return False

    def _worker_main(self) -> None:
        conn: Optional[sqlite3.Connection] = None
        next_run_at: Optional[float] = None
        try:
            while not self._stop.is_set():
                if conn is None:
                    try:
                        with self._fork_guard:
                            conn = sqlite3.connect(
                                str(self.db_path),
                                timeout=0.0,
                                isolation_level=None,
                            )
                            conn.execute("PRAGMA busy_timeout=0")
                            conn.execute("PRAGMA foreign_keys=ON")
                            self._connection = conn
                    except Exception:
                        logger.exception(
                            "state.db maintenance connection failed for %s; "
                            "retrying",
                            self.db_path,
                        )
                        conn = None
                        if self._stop.wait(self._WORKER_ERROR_BACKOFF_S):
                            break
                        continue

                timeout = None
                if next_run_at is not None:
                    timeout = max(0.0, next_run_at - time.monotonic())
                signaled = self._wake.wait(timeout)
                self._wake.clear()
                if self._stop.is_set():
                    break

                # A foreground commit may wake us while an FTS debounce or
                # retry delay is already scheduled. Coalesce that wake unless
                # a checkpoint generation also needs immediate service.
                if signaled and next_run_at is not None:
                    with self._state_lock:
                        checkpoint_due = self._checkpoint_pending
                    if not checkpoint_due and time.monotonic() < next_run_at:
                        continue

                try:
                    with self._fork_guard:
                        delay = self._run_burst(conn)
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.exception(
                        "state.db background maintenance burst failed for %s; "
                        "worker remains available",
                        self.db_path,
                    )
                    delay = self._WORKER_ERROR_BACKOFF_S
                next_run_at = (
                    None if delay is None else time.monotonic() + max(0.0, delay)
                )
        finally:
            self._write_gate.cancel_maintenance_reservation()
            if conn is not None:
                with self._fork_guard:
                    self._release_lease(conn)
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._connection = None

    def _run_burst(self, conn: sqlite3.Connection) -> Optional[float]:
        """Run one bounded burst and return its next delay, if any."""
        with self._state_lock:
            self._active = True
        try:
            # A force-aged FTS attempt may have reserved the foreground lane
            # before discovering that the current writer still owns it. A
            # PASSIVE checkpoint deliberately runs outside that lane, so it
            # must never carry the old reservation across its SQLite I/O.
            # Keep the aging timestamp: the following FTS slice may reserve
            # again once checkpointing has yielded.
            self._write_gate.cancel_maintenance_reservation()
            self._run_checkpoint_if_due(conn)
            burst_started = time.monotonic()
            slices = 0
            last_status = "idle"
            retry_after = 0.0
            while (
                not self._stop.is_set()
                and slices < self._MAX_SLICES_PER_BURST
                and time.monotonic() - burst_started < self._MAX_BURST_S
            ):
                last_status, retry_after = self._run_fts_slice(conn)
                if last_status != "progress":
                    break
                slices += 1
                if self._stop.wait(self._INTER_SLICE_S):
                    break
            if self._stop.is_set():
                return None
            if last_status == "progress":
                # Yield to foreground writers between bounded bursts.
                return self._INTER_SLICE_S
            if last_status == "wait":
                return min(max(retry_after, 0.025), 1.0)
            return None
        finally:
            with self._state_lock:
                self._active = False

    def _run_checkpoint_if_due(self, conn: sqlite3.Connection) -> None:
        with self._state_lock:
            if not self._checkpoint_pending:
                return
            target_seq = self._checkpoint_requested_seq
        try:
            started = time.monotonic()
            # PASSIVE checkpointing has its own SQLite checkpoint lock and
            # never waits for readers or writers. Do not occupy Hermes' Python
            # writer gate while frames are copied: foreground commits may run
            # concurrently and SQLite will stop the checkpoint at their WAL
            # boundary.
            result = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            with self._state_lock:
                self._checkpoint_completed_seq = max(
                    self._checkpoint_completed_seq, target_seq
                )
                self._checkpoint_pending = (
                    self._checkpoint_requested_seq > self._checkpoint_completed_seq
                )
                checkpoint_still_due = self._checkpoint_pending
            if checkpoint_still_due:
                self._wake.set()
            if result and result[1] > 0:
                logger.debug(
                    "state.db maintenance checkpoint pages=%d/%d duration_ms=%.1f",
                    result[2],
                    result[1],
                    (time.monotonic() - started) * 1000,
                )
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                self._wake.set()
                return
            if not self._is_busy(exc):
                logger.warning(
                    "state.db background PASSIVE checkpoint failed: %s", exc
                )
            self._wake.set()

    def _run_fts_slice(self, conn: sqlite3.Connection) -> tuple[str, float]:
        monotonic_now = time.monotonic()
        if self._gate_deferred_since is None:
            self._gate_deferred_since = monotonic_now
        reserve_slice = (
            monotonic_now - self._gate_deferred_since >= self._MAX_GATE_DEFER_S
        )
        if not self._write_gate.try_acquire_maintenance(force=reserve_slice):
            return "wait", random.uniform(0.025, 0.100)
        self._gate_deferred_since = None
        transaction_open = False
        try:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            transaction_open = True
            row = conn.execute(
                """SELECT requested_seq, cycle_target_seq, completed_seq,
                          state, base_phase, trigram_phase, next_table,
                          lease_owner, lease_expires_at, not_before, force_run
                   FROM state_maintenance_jobs WHERE job_name = ?""",
                (_FTS_JOB,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO state_maintenance_jobs "
                    "(job_name, requested_seq, force_run, updated_at) "
                    "VALUES (?, 0, 1, ?)",
                    (_FTS_JOB, now),
                )
                row = (
                    0,
                    0,
                    0,
                    "idle",
                    "start",
                    "start",
                    0,
                    None,
                    0.0,
                    0.0,
                    1,
                )

            (
                requested_seq,
                cycle_target_seq,
                completed_seq,
                state,
                base_phase,
                trigram_phase,
                next_table,
                lease_owner,
                lease_expires_at,
                not_before,
                force_run,
            ) = row

            if float(not_before) > now:
                conn.rollback()
                transaction_open = False
                return "wait", float(not_before) - now
            if (
                state == "running"
                and lease_owner not in (None, self._lease_owner)
                and float(lease_expires_at) > now
            ):
                conn.rollback()
                transaction_open = False
                return "wait", float(lease_expires_at) - now
            if state == "idle":
                debt = int(requested_seq) - int(completed_seq)
                if not int(force_run) and debt < self._merge_interval:
                    conn.execute(
                        """UPDATE state_maintenance_jobs
                           SET not_before = 0, updated_at = ?
                           WHERE job_name = ?""",
                        (now, _FTS_JOB),
                    )
                    conn.commit()
                    transaction_open = False
                    return "idle", 0.0
                cycle_target_seq = requested_seq
                base_phase = "start"
                trigram_phase = "start"
                next_table = 0

            conn.execute(
                """UPDATE state_maintenance_jobs
                   SET state = 'running', cycle_target_seq = ?,
                       base_phase = ?, trigram_phase = ?, next_table = ?,
                       force_run = 0, not_before = 0,
                       lease_owner = ?, lease_expires_at = ?, updated_at = ?
                   WHERE job_name = ?""",
                (
                    cycle_target_seq,
                    base_phase,
                    trigram_phase,
                    next_table,
                    self._lease_owner,
                    now + self._LEASE_S,
                    now,
                    _FTS_JOB,
                ),
            )

            phases = [base_phase, trigram_phase]
            table_index = self._next_pending_table(phases, int(next_table))
            if table_index is None:
                self._complete_cycle(conn, int(cycle_target_seq), now)
                conn.commit()
                transaction_open = False
                return "progress", 0.0

            table_name = _FTS_TABLES[table_index]
            phase_column = _PHASE_COLUMNS[table_index]
            if not self._fts_table_exists(conn, table_name):
                phases[table_index] = "missing"
                self._update_phase(
                    conn,
                    phase_column=phase_column,
                    phase="missing",
                    next_table=1 - table_index,
                    now=now,
                )
                self._finish_cycle_if_complete(
                    conn,
                    phases=phases,
                    cycle_target_seq=int(cycle_target_seq),
                    now=now,
                )
                conn.commit()
                transaction_open = False
                return "progress", 0.0

            signed_pages = (
                -self._merge_pages
                if phases[table_index] == "start"
                else self._merge_pages
            )
            started = time.monotonic()
            before = conn.total_changes
            conn.execute(
                f"INSERT INTO {table_name}({table_name}, rank) VALUES('merge', ?)",
                (signed_pages,),
            )
            merge_changes = conn.total_changes - before
            duration_s = time.monotonic() - started
            next_phase = "continue" if merge_changes >= 2 else "done"
            phases[table_index] = next_phase
            self._update_phase(
                conn,
                phase_column=phase_column,
                phase=next_phase,
                next_table=1 - table_index,
                now=now,
            )
            self._finish_cycle_if_complete(
                conn,
                phases=phases,
                cycle_target_seq=int(cycle_target_seq),
                now=now,
            )
            conn.commit()
            transaction_open = False
            self._adapt_merge_pages(duration_s, merge_changes)
            logger.debug(
                "state.db FTS merge table=%s pages=%d changes=%d "
                "duration_ms=%.1f phase=%s",
                table_name,
                signed_pages,
                merge_changes,
                duration_s * 1000,
                next_phase,
            )
            return "progress", 0.0
        except sqlite3.OperationalError as exc:
            if transaction_open:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if self._is_busy(exc):
                logger.debug("state.db FTS merge skipped because SQLite is busy")
                return "wait", random.uniform(0.050, 0.200)
            if "interrupted" in str(exc).lower():
                return "wait", 0.050
            self._defer_after_error(conn, type(exc).__name__)
            logger.warning("state.db background FTS merge failed: %s", exc)
            return "wait", 1.0
        except Exception as exc:
            if transaction_open:
                try:
                    conn.rollback()
                except Exception:
                    pass
            self._defer_after_error(conn, type(exc).__name__)
            logger.warning("state.db background FTS maintenance failed: %s", exc)
            return "wait", 1.0
        finally:
            self._write_gate.release_maintenance()

    @staticmethod
    def _next_pending_table(phases: list[str], preferred: int) -> Optional[int]:
        for table_index in (preferred, 1 - preferred):
            if phases[table_index] in ("start", "continue"):
                return table_index
        return None

    def _fts_table_exists(
        self, conn: sqlite3.Connection, table_name: str
    ) -> bool:
        if table_name not in self._fts_tables:
            return False
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _update_phase(
        conn: sqlite3.Connection,
        *,
        phase_column: str,
        phase: str,
        next_table: int,
        now: float,
    ) -> None:
        if phase_column not in _PHASE_COLUMNS:
            raise ValueError(f"invalid FTS phase column: {phase_column}")
        conn.execute(
            f"""UPDATE state_maintenance_jobs
                SET {phase_column} = ?, next_table = ?,
                    lease_expires_at = ?, failure_count = 0,
                    last_error_code = NULL, updated_at = ?
                WHERE job_name = ?""",
            (phase, next_table, now + StateDbMaintenanceCoordinator._LEASE_S, now, _FTS_JOB),
        )

    @staticmethod
    def _finish_cycle_if_complete(
        conn: sqlite3.Connection,
        *,
        phases: list[str],
        cycle_target_seq: int,
        now: float,
    ) -> None:
        if all(phase in ("done", "missing") for phase in phases):
            StateDbMaintenanceCoordinator._complete_cycle(
                conn, cycle_target_seq, now
            )

    @staticmethod
    def _complete_cycle(
        conn: sqlite3.Connection,
        cycle_target_seq: int,
        now: float,
    ) -> None:
        # Only acknowledge the generation frozen at cycle start. Writes that
        # commit between the base and trigram slices remain durable debt.
        conn.execute(
            """UPDATE state_maintenance_jobs
               SET completed_seq = MAX(completed_seq, ?), state = 'idle',
                   lease_owner = NULL, lease_expires_at = 0,
                   not_before = 0,
                   failure_count = 0, last_error_code = NULL, updated_at = ?
               WHERE job_name = ?""",
            (cycle_target_seq, now, _FTS_JOB),
        )

    def _adapt_merge_pages(self, duration_s: float, merge_changes: int) -> None:
        if duration_s > 0.100:
            self._merge_pages = max(
                self._MIN_MERGE_PAGES, self._merge_pages // 2
            )
        elif duration_s < 0.020 and merge_changes >= 2:
            self._merge_pages = min(
                self._MAX_MERGE_PAGES, self._merge_pages * 2
            )

    def _defer_after_error(self, conn: sqlite3.Connection, code: str) -> None:
        try:
            now = time.time()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT failure_count FROM state_maintenance_jobs "
                "WHERE job_name = ?",
                (_FTS_JOB,),
            ).fetchone()
            failures = int(row[0]) + 1 if row else 1
            backoff_s = min(60.0, float(2 ** min(failures, 6)))
            conn.execute(
                """UPDATE state_maintenance_jobs
                   SET lease_owner = NULL, lease_expires_at = 0,
                       not_before = ?, failure_count = ?,
                       last_error_code = ?, updated_at = ?
                   WHERE job_name = ?""",
                (now + backoff_s, failures, code[:80], now, _FTS_JOB),
            )
            conn.commit()
        except sqlite3.Error:
            try:
                conn.rollback()
            except Exception:
                pass

    def _release_lease(self, conn: sqlite3.Connection) -> None:
        if self._owner_pid != os.getpid():
            return
        if not self._write_gate.try_acquire_maintenance():
            return
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE state_maintenance_jobs
                   SET lease_owner = NULL, lease_expires_at = 0,
                       updated_at = ?
                   WHERE job_name = ? AND lease_owner = ?""",
                (time.time(), _FTS_JOB, self._lease_owner),
            )
            conn.commit()
        except sqlite3.Error:
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            self._write_gate.release_maintenance()

    @staticmethod
    def _is_busy(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message


_COORDINATORS: dict[str, tuple[StateDbMaintenanceCoordinator, int]] = {}
_COORDINATORS_GUARD = threading.Lock()
_FORK_COORDINATORS_HELD: list[StateDbMaintenanceCoordinator] = []
_FORK_REGISTRY_HELD = False


def prepare_state_db_maintenance_for_fork() -> None:
    """Quiesce every maintenance worker before the process is forked."""
    global _FORK_COORDINATORS_HELD, _FORK_REGISTRY_HELD
    _COORDINATORS_GUARD.acquire()
    _FORK_REGISTRY_HELD = True
    coordinators = [entry[0] for entry in _COORDINATORS.values()]
    held: list[StateDbMaintenanceCoordinator] = []
    try:
        for coordinator in coordinators:
            coordinator.prepare_for_fork()
            held.append(coordinator)
        _FORK_COORDINATORS_HELD = held
    except BaseException:
        for coordinator in reversed(held):
            coordinator.resume_after_fork_parent()
        _COORDINATORS_GUARD.release()
        _FORK_REGISTRY_HELD = False
        _FORK_COORDINATORS_HELD = []


def resume_state_db_maintenance_after_fork_parent() -> None:
    """Resume workers and registry mutation in the parent after fork."""
    global _FORK_COORDINATORS_HELD, _FORK_REGISTRY_HELD
    held = _FORK_COORDINATORS_HELD
    _FORK_COORDINATORS_HELD = []
    for coordinator in reversed(held):
        coordinator.resume_after_fork_parent()
    if _FORK_REGISTRY_HELD:
        _COORDINATORS_GUARD.release()
        _FORK_REGISTRY_HELD = False


def acquire_state_db_maintenance(
    *,
    db_path: Path,
    db_key: str,
    write_gate: Any,
    fts_tables: tuple[str, ...],
    merge_interval: int,
) -> StateDbMaintenanceCoordinator:
    """Acquire a shared process-level coordinator reference."""
    with _COORDINATORS_GUARD:
        entry = _COORDINATORS.get(db_key)
        if entry is not None:
            coordinator, references = entry
            if references == 0:
                # A previous final close timed out. Never create a second
                # worker for the same database while the old one may still be
                # inside SQLite; retry its bounded stop first.
                if not coordinator.stop():
                    raise RuntimeError(
                        "state.db maintenance worker is still stopping"
                    )
                _COORDINATORS.pop(db_key, None)
            else:
                _COORDINATORS[db_key] = (coordinator, references + 1)
                return coordinator
        coordinator = StateDbMaintenanceCoordinator(
            db_path=db_path,
            db_key=db_key,
            write_gate=write_gate,
            fts_tables=fts_tables,
            merge_interval=merge_interval,
        )
        _COORDINATORS[db_key] = (coordinator, 1)
        return coordinator


def release_state_db_maintenance(
    *,
    db_key: str,
    coordinator: StateDbMaintenanceCoordinator,
) -> str:
    """Release a reference and stop the worker when the final handle closes."""
    with _COORDINATORS_GUARD:
        entry = _COORDINATORS.get(db_key)
        if entry is None or entry[0] is not coordinator:
            return "shared"
        references = entry[1] - 1
        if references > 0:
            _COORDINATORS[db_key] = (coordinator, references)
            return "shared"

        # Keep the registry guard and entry until join completes. A concurrent
        # reopen must not install a second coordinator during the stop window.
        stopped = coordinator.stop()
        if stopped:
            _COORDINATORS.pop(db_key, None)
            return "stopped"
        _COORDINATORS[db_key] = (coordinator, 0)
        return "timeout"


def get_state_db_maintenance(
    db_key: str,
) -> Optional[StateDbMaintenanceCoordinator]:
    """Return the current coordinator without changing its reference count."""
    with _COORDINATORS_GUARD:
        entry = _COORDINATORS.get(db_key)
        return entry[0] if entry is not None else None


def reset_state_db_maintenance_after_fork(
    retained_connections: list[sqlite3.Connection],
) -> None:
    """Reset inherited worker state without closing SQLite in the child."""
    global _COORDINATORS, _COORDINATORS_GUARD
    global _FORK_COORDINATORS_HELD, _FORK_REGISTRY_HELD
    inherited = _COORDINATORS
    _COORDINATORS = {}
    _COORDINATORS_GUARD = threading.Lock()
    # The old registry/fork guards were deliberately locked by the forking
    # thread. Never release inherited Python locks in the child; discard them.
    _FORK_COORDINATORS_HELD = []
    _FORK_REGISTRY_HELD = False
    for coordinator, _references in inherited.values():
        try:
            conn = coordinator.inherited_connection_after_fork()
            if isinstance(conn, sqlite3.Connection):
                retained_connections.append(conn)
        except Exception:
            pass
