"""Availability contracts for state.db background maintenance."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import sqlite3
import threading
import time

import pytest

import hermes_state as state_module
from hermes_state import SessionDB
import state_db_maintenance as maintenance_module
from state_db_maintenance import StateDbMaintenanceCoordinator


def _fts_job(db):
    return db._conn.execute(
        "SELECT requested_seq, completed_seq, state "
        "FROM state_maintenance_jobs WHERE job_name = 'fts_merge'"
    ).fetchone()


def test_write_path_never_runs_full_fts_optimize_inline(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    optimize_entered = threading.Event()
    release_optimize = threading.Event()

    def slow_full_optimize():
        optimize_entered.set()
        release_optimize.wait(timeout=10)
        return 2

    monkeypatch.setattr(db, "optimize_fts", slow_full_optimize)
    db._OPTIMIZE_EVERY_N_WRITES = 1

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            write = pool.submit(db.create_session, "foreground", "test")
            try:
                write.result(timeout=2)
            finally:
                release_optimize.set()
        assert not optimize_entered.is_set()
        assert db.get_session("foreground") is not None
    finally:
        release_optimize.set()
        db.close()


def test_periodic_checkpoint_never_delays_committed_write(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    checkpoint_entered = threading.Event()
    release_checkpoint = threading.Event()

    def slow_checkpoint():
        checkpoint_entered.set()
        release_checkpoint.wait(timeout=10)

    monkeypatch.setattr(db, "_try_wal_checkpoint", slow_checkpoint)
    db._CHECKPOINT_EVERY_N_WRITES = 1
    db._OPTIMIZE_EVERY_N_WRITES = 10_000

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            write = pool.submit(db.create_session, "foreground", "test")
            try:
                write.result(timeout=2)
            finally:
                release_checkpoint.set()
        assert db.get_session("foreground") is not None
    finally:
        release_checkpoint.set()
        db.close()


def test_background_checkpoint_does_not_hold_foreground_writer_gate(
    tmp_path, monkeypatch
):
    checkpoint_entered = threading.Event()
    release_checkpoint = threading.Event()
    armed = threading.Event()
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(statement):
            normalized = "".join(statement.lower().split())
            if armed.is_set() and "wal_checkpoint(passive)" in normalized:
                checkpoint_entered.set()
                release_checkpoint.wait(timeout=10)

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(maintenance_module.sqlite3, "connect", traced_connect)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db._CHECKPOINT_EVERY_N_WRITES = 1
        armed.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(db.create_session, "first", "test")
            first.result(timeout=2)
            assert checkpoint_entered.wait(timeout=10)
            # The checkpoint is still blocked in its worker connection. A
            # second foreground write must not queue behind a Python lock.
            second = pool.submit(db.create_session, "second", "test")
            second.result(timeout=2)
        assert db.get_session("second") is not None
    finally:
        release_checkpoint.set()
        db.close()


def test_checkpoint_commit_during_checkpoint_is_not_lost(tmp_path, monkeypatch):
    checkpoint_count = 0
    checkpoint_count_lock = threading.Lock()
    first_checkpoint_started = threading.Event()
    second_checkpoint_started = threading.Event()
    release_first_checkpoint = threading.Event()
    armed = threading.Event()
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(statement):
            nonlocal checkpoint_count
            normalized = "".join(statement.lower().split())
            if not armed.is_set() or "wal_checkpoint(passive)" not in normalized:
                return
            with checkpoint_count_lock:
                checkpoint_count += 1
                generation = checkpoint_count
            if generation == 1:
                first_checkpoint_started.set()
                release_first_checkpoint.wait(timeout=10)
            elif generation == 2:
                second_checkpoint_started.set()

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(maintenance_module.sqlite3, "connect", traced_connect)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db._CHECKPOINT_EVERY_N_WRITES = 1
        armed.set()

        db.create_session("checkpoint-one", "test")
        assert first_checkpoint_started.wait(timeout=10)

        # This commit crosses the next checkpoint generation while the first
        # checkpoint is still in flight. Completing generation one must not
        # clear generation two's pending work.
        db.create_session("checkpoint-two", "test")
        release_first_checkpoint.set()

        assert second_checkpoint_started.wait(timeout=10)
        assert db._maintenance.wait_idle(timeout_s=10)
    finally:
        release_first_checkpoint.set()
        db.close()


def test_subthreshold_fts_debt_is_idle_without_forcing_merge(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db.create_session("below-threshold", "test")
        baseline = tuple(_fts_job(db))
        db._OPTIMIZE_EVERY_N_WRITES = 5

        db.append_messages(
            "below-threshold",
            [
                {"role": "user", "content": f"subthreshold {index}"}
                for index in range(3)
            ],
        )

        # Durable debt remains visible, but it is deliberately below the
        # configured merge interval and therefore counts as quiescent work.
        assert db._maintenance.wait_idle(timeout_s=2)
        requested, completed, state = _fts_job(db)
        assert requested == baseline[0] + 3
        assert completed == baseline[1]
        assert state == "idle"
    finally:
        db.close()


def test_legacy_maintenance_table_adds_force_run_and_completes_catch_up(tmp_path):
    db_path = tmp_path / "state.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(
        """
        CREATE TABLE state_maintenance_jobs (
            job_name TEXT PRIMARY KEY,
            requested_seq INTEGER NOT NULL DEFAULT 0,
            cycle_target_seq INTEGER NOT NULL DEFAULT 0,
            completed_seq INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'idle',
            base_phase TEXT NOT NULL DEFAULT 'start',
            trigram_phase TEXT NOT NULL DEFAULT 'start',
            next_table INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at REAL NOT NULL DEFAULT 0,
            not_before REAL NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            last_error_code TEXT,
            updated_at REAL NOT NULL DEFAULT 0
        );
        INSERT INTO state_maintenance_jobs (job_name, requested_seq)
        VALUES ('fts_merge', 0);
        """
    )
    legacy.commit()
    legacy.close()

    db = SessionDB(db_path=db_path)
    try:
        columns = {
            row[1]
            for row in db._conn.execute(
                "PRAGMA table_info(state_maintenance_jobs)"
            ).fetchall()
        }
        assert "force_run" in columns
        assert db._maintenance.wait_idle(timeout_s=10)
        row = db._conn.execute(
            "SELECT requested_seq, completed_seq, state, force_run "
            "FROM state_maintenance_jobs WHERE job_name = 'fts_merge'"
        ).fetchone()
        assert tuple(row) == (0, 0, "idle", 0)
    finally:
        db.close()


def test_message_write_runs_bounded_merge_and_never_full_optimize(
    tmp_path, monkeypatch
):
    merge_seen = threading.Event()
    armed = threading.Event()
    statements = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(statement):
            if not armed.is_set():
                return
            normalized = "".join(statement.lower().split())
            statements.append(normalized)
            if "values('merge'," in normalized:
                merge_seen.set()

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(maintenance_module.sqlite3, "connect", traced_connect)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db._OPTIMIZE_EVERY_N_WRITES = 1
        before = db._conn.execute(
            "SELECT requested_seq FROM state_maintenance_jobs "
            "WHERE job_name = 'fts_merge'"
        ).fetchone()[0]
        armed.set()
        db.create_session("searchable", "test")
        db.append_message("searchable", "user", "bounded merge needle")
        assert merge_seen.wait(timeout=10)
        assert db._maintenance.wait_idle(timeout_s=10)
        job = db._conn.execute(
            "SELECT requested_seq, completed_seq, state "
            "FROM state_maintenance_jobs WHERE job_name = 'fts_merge'"
        ).fetchone()
        assert tuple(job) == (before + 1, before + 1, "idle")
        for table_name in ("messages_fts", "messages_fts_trigram"):
            assert any(
                f"insertinto{table_name}({table_name},rank)values('merge',-"
                in statement
                for statement in statements
            )
        assert not any("optimize" in statement for statement in statements)
        assert len(db.search_messages("needle")) == 1
    finally:
        db.close()


def test_cycle_completion_preserves_new_debt_and_reopen_resumes(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(SessionDB, "_OPTIMIZE_EVERY_N_WRITES", 1)
    original_worker_main = StateDbMaintenanceCoordinator._worker_main

    def dormant_worker(coordinator):
        coordinator._stop.wait()

    monkeypatch.setattr(
        StateDbMaintenanceCoordinator,
        "_worker_main",
        dormant_worker,
    )
    db = SessionDB(db_path=db_path)
    manual = sqlite3.connect(str(db_path), timeout=0.0, isolation_level=None)
    try:
        manual.execute("PRAGMA busy_timeout=0")
        db.create_session("cycle", "test")
        db.append_message("cycle", "user", "first generation")
        manual.execute(
            "UPDATE state_maintenance_jobs SET not_before = 0 "
            "WHERE job_name = 'fts_merge'"
        )
        status, _retry_after = db._maintenance._run_fts_slice(manual)
        assert status == "progress"
        frozen_target = manual.execute(
            "SELECT cycle_target_seq FROM state_maintenance_jobs "
            "WHERE job_name = 'fts_merge'"
        ).fetchone()[0]

        db.append_message("cycle", "assistant", "second generation")
        first_cycle = None
        for _ in range(20):
            status, _retry_after = db._maintenance._run_fts_slice(manual)
            assert status in {"progress", "idle"}
            row = manual.execute(
                "SELECT requested_seq, completed_seq, state "
                "FROM state_maintenance_jobs WHERE job_name = 'fts_merge'"
            ).fetchone()
            if row[2] == "idle":
                first_cycle = row
                break
        assert first_cycle is not None
        assert tuple(first_cycle) == (frozen_target + 1, frozen_target, "idle")
    finally:
        manual.close()
        db.close()

    monkeypatch.setattr(
        StateDbMaintenanceCoordinator,
        "_worker_main",
        original_worker_main,
    )
    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened._maintenance.wait_idle(timeout_s=10)
        row = reopened._conn.execute(
            "SELECT requested_seq, completed_seq, state "
            "FROM state_maintenance_jobs WHERE job_name = 'fts_merge'"
        ).fetchone()
        assert row[0] == row[1]
        assert row[2] == "idle"
    finally:
        reopened.close()


def test_worker_survives_one_burst_failure_and_drains_debt(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    first_failure = threading.Event()
    calls = 0
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db._OPTIMIZE_EVERY_N_WRITES = 1
        db.create_session("burst-retry", "test")
        original_run_burst = db._maintenance._run_burst

        def fail_once(conn):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_failure.set()
                raise RuntimeError("simulated maintenance burst failure")
            return original_run_burst(conn)

        monkeypatch.setattr(db._maintenance, "_run_burst", fail_once)
        before = _fts_job(db)[0]
        db.append_message("burst-retry", "user", "durable retry debt")

        assert first_failure.wait(timeout=10)
        assert db._maintenance.wait_idle(timeout_s=10)
        assert db._maintenance._thread.is_alive()
        assert calls >= 2
        requested, completed, state = _fts_job(db)
        assert (requested, completed, state) == (before + 1, before + 1, "idle")
    finally:
        db.close()


def test_maintenance_aging_forces_one_slice_after_sustained_contention(
    tmp_path, monkeypatch
):
    class AlwaysContendedGate:
        def __init__(self):
            self.attempts = []
            self.acquired = False

        def try_acquire_maintenance(self, *, force=False):
            self.attempts.append(force)
            self.acquired = bool(force)
            return self.acquired

        def release_maintenance(self):
            assert self.acquired
            self.acquired = False

        def cancel_maintenance_reservation(self):
            self.acquired = False

    def dormant_worker(coordinator):
        coordinator._stop.wait()

    monkeypatch.setattr(
        StateDbMaintenanceCoordinator,
        "_worker_main",
        dormant_worker,
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    manual = sqlite3.connect(str(db.db_path), timeout=0.0, isolation_level=None)
    gate = AlwaysContendedGate()
    real_gate = db._maintenance._write_gate
    try:
        db._OPTIMIZE_EVERY_N_WRITES = 1
        db.create_session("aging", "test")
        db.append_message("aging", "user", "maintenance aging debt")
        manual.execute(
            "UPDATE state_maintenance_jobs SET not_before = 0 "
            "WHERE job_name = 'fts_merge'"
        )
        db._maintenance._write_gate = gate

        status, _retry_after = db._maintenance._run_fts_slice(manual)
        assert status == "wait"
        assert gate.attempts == [False]
        assert db._maintenance._gate_deferred_since is not None

        # Model a continuously queued foreground lane without sleeping. Once
        # the bounded aging window has elapsed, maintenance gets one forced
        # acquisition instead of starving forever.
        db._maintenance._gate_deferred_since = time.monotonic() - 86_400
        status, _retry_after = db._maintenance._run_fts_slice(manual)
        assert status == "progress"
        assert gate.attempts[-1] is True
    finally:
        db._maintenance._write_gate = real_gate
        manual.close()
        db.close()


def test_forced_checkpoint_clears_abandoned_maintenance_reservation(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    second = SessionDB(db_path=db_path)
    coordinator = first._maintenance
    gate = first._write_lock
    barrier_entered = threading.Event()
    barrier_results = []
    real_acquire_barrier = coordinator.acquire_sqlite_activity_barrier

    def traced_acquire_barrier(timeout_s=2.0):
        acquired = real_acquire_barrier(timeout_s)
        barrier_results.append(acquired)
        barrier_entered.set()
        return acquired

    monkeypatch.setattr(
        coordinator,
        "acquire_sqlite_activity_barrier",
        traced_acquire_barrier,
    )
    gate_held = False
    try:
        assert coordinator is second._maintenance
        assert coordinator.wait_idle(timeout_s=10)

        with ThreadPoolExecutor(max_workers=2) as pool:
            gate.acquire()
            gate_held = True
            closing = None
            try:
                # A force-aged maintenance attempt closes foreground admission,
                # but cannot take the writer gate while this holder is active.
                reserved = pool.submit(
                    gate.try_acquire_maintenance,
                    force=True,
                ).result(timeout=2)
                assert reserved is False
                assert gate._maintenance_reserved is True

                closing = pool.submit(first.close, force_checkpoint=True)
                assert barrier_entered.wait(timeout=5)
                assert barrier_results == [True]
                assert gate._maintenance_reserved is False
                assert not closing.done()
            finally:
                # This also makes the failure mode self-cleaning: an old/broken
                # barrier that leaves the reservation behind cannot hang pytest.
                gate.cancel_maintenance_reservation()
                if gate_held:
                    gate.release()
                    gate_held = False

            assert closing is not None
            closing.result(timeout=5)

        assert coordinator.is_alive()
        second.create_session("after-forced-checkpoint", "test")
    finally:
        gate.cancel_maintenance_reservation()
        if gate_held:
            gate.release()
        first.close()
        second.close()


def test_checkpoint_burst_clears_failed_forced_reservation_before_sqlite_io(
    tmp_path, monkeypatch
):
    checkpoint_entered = threading.Event()
    release_checkpoint = threading.Event()
    armed = threading.Event()
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(statement):
            normalized = "".join(statement.lower().split())
            if armed.is_set() and "wal_checkpoint(passive)" in normalized:
                checkpoint_entered.set()
                release_checkpoint.wait(timeout=10)

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(maintenance_module.sqlite3, "connect", traced_connect)
    db = SessionDB(db_path=tmp_path / "state.db")
    gate = db._write_lock
    gate_held = False
    writer_pool = None
    try:
        assert db._maintenance.wait_idle(timeout_s=10)

        gate.acquire()
        gate_held = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            acquired = pool.submit(
                gate.try_acquire_maintenance,
                force=True,
            ).result(timeout=2)
        assert acquired is False
        assert gate._maintenance_reserved is True
        gate.release()
        gate_held = False

        armed.set()
        db._maintenance.notify_commit(
            write_units=1,
            checkpoint_interval=1,
            fts_dirty=False,
            merge_interval=db._OPTIMIZE_EVERY_N_WRITES,
        )
        assert checkpoint_entered.wait(timeout=10)

        # The checkpoint is still inside worker SQLite I/O. Foreground admission
        # must already be open, otherwise this write waits forever on the stale
        # forced reservation even though PASSIVE itself owns no writer gate.
        assert gate._maintenance_reserved is False
        writer_pool = ThreadPoolExecutor(max_workers=1)
        write = writer_pool.submit(db.create_session, "during-checkpoint", "test")
        write.result(timeout=2)
        assert db.get_session("during-checkpoint") is not None
    finally:
        gate.cancel_maintenance_reservation()
        if gate_held:
            gate.release()
        release_checkpoint.set()
        if writer_pool is not None:
            writer_pool.shutdown(wait=True)
        db.close()


def test_canonical_database_uses_one_maintenance_coordinator(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    first = SessionDB(db_path=real_dir / "state.db")
    second = SessionDB(db_path=link_dir / "state.db")
    coordinator = first._maintenance
    db_key = first._db_key
    reopened = None
    try:
        assert coordinator is second._maintenance
        assert first._write_lock is second._write_lock
        first.create_session("first", "test")

        first.close()
        assert coordinator.is_alive()
        second.create_session("second", "test")

        second.close()
        assert not coordinator.is_alive()
        assert maintenance_module.get_state_db_maintenance(db_key) is None

        reopened = SessionDB(db_path=link_dir / "state.db")
        assert reopened._maintenance is not coordinator
        assert reopened._maintenance.is_alive()
        assert maintenance_module.get_state_db_maintenance(db_key) is reopened._maintenance
    finally:
        first.close()
        second.close()
        if reopened is not None:
            reopened.close()


def test_reopen_waits_until_final_coordinator_stop_completes(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    coordinator = first._maintenance
    db_key = first._db_key
    stop_entered = threading.Event()
    release_stop = threading.Event()
    reopen_started = threading.Event()
    opened = []
    real_stop = coordinator.stop

    def delayed_stop(timeout_s=2.0):
        stop_entered.set()
        assert release_stop.wait(timeout=10)
        return real_stop(timeout_s)

    def reopen():
        reopen_started.set()
        handle = SessionDB(db_path=db_path)
        opened.append(handle)
        return handle

    monkeypatch.setattr(coordinator, "stop", delayed_stop)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            closing = pool.submit(first.close)
            assert stop_entered.wait(timeout=10)
            opening = pool.submit(reopen)
            assert reopen_started.wait(timeout=10)
            try:
                with pytest.raises(FutureTimeoutError):
                    opening.result(timeout=0.2)

                # release_state_db_maintenance keeps the old registry entry and
                # guard through join. The concurrent constructor therefore
                # cannot publish a second live coordinator in this window.
                assert maintenance_module._COORDINATORS[db_key][0] is coordinator
                assert coordinator.is_alive()
            finally:
                release_stop.set()

            closing.result(timeout=10)
            reopened = opening.result(timeout=10)

        assert not coordinator.is_alive()
        assert reopened._maintenance is not coordinator
        assert reopened._maintenance.is_alive()
        assert maintenance_module.get_state_db_maintenance(db_key) is reopened._maintenance
    finally:
        release_stop.set()
        first.close()
        for handle in opened:
            handle.close()


def test_reopen_stop_timeout_cleans_failed_connection_and_writer_registration(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    coordinator = first._maintenance
    db_key = first._db_key
    real_stop = coordinator.stop
    real_connect = sqlite3.connect
    failed_connections = []

    def never_stops(timeout_s=2.0):
        return False

    def tracked_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        if kwargs.get("check_same_thread") is False:
            failed_connections.append(conn)
        return conn

    monkeypatch.setattr(coordinator, "stop", never_stops)
    try:
        first.close(checkpoint=False)
        assert maintenance_module._COORDINATORS[db_key] == (coordinator, 0)
        assert state_module._process_db_writer_count(db_path) == 0

        monkeypatch.setattr(state_module.sqlite3, "connect", tracked_connect)
        with pytest.raises(
            RuntimeError,
            match="state.db maintenance worker is still stopping",
        ):
            SessionDB(db_path=db_path)

        assert state_module._process_db_writer_count(db_path) == 0
        assert maintenance_module._COORDINATORS[db_key] == (coordinator, 0)
        assert len(failed_connections) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            failed_connections[0].execute("SELECT 1")
    finally:
        monkeypatch.setattr(state_module.sqlite3, "connect", real_connect)
        monkeypatch.setattr(coordinator, "stop", real_stop)
        coordinator.stop()
        with maintenance_module._COORDINATORS_GUARD:
            entry = maintenance_module._COORDINATORS.get(db_key)
            if entry is not None and entry[0] is coordinator:
                maintenance_module._COORDINATORS.pop(db_key, None)
        first.close(checkpoint=False)


def test_failed_message_transaction_does_not_publish_fts_debt(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._maintenance.wait_idle(timeout_s=10)
        db._OPTIMIZE_EVERY_N_WRITES = 1
        before = tuple(_fts_job(db))

        def fail_at_commit(conn):
            conn.execute("PRAGMA defer_foreign_keys=ON")
            conn.execute(
                "INSERT INTO messages (session_id, role, timestamp) VALUES (?, ?, ?)",
                ("missing-session", "user", time.time()),
            )

        # The deferred FK violation happens at COMMIT, after durable FTS debt
        # was recorded in the same transaction. Rollback must remove both.
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(fail_at_commit, fts_dirty=True)

        assert tuple(_fts_job(db)) == before
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'missing-session'"
        ).fetchone()[0] == 0
        assert db._maintenance.wait_idle(timeout_s=2)
    finally:
        db.close()


def test_64_parallel_message_writes_have_no_locked_failures(tmp_path):
    db_path = tmp_path / "state.db"
    handles = [SessionDB(db_path=db_path) for _ in range(64)]
    primary = handles[0]
    try:
        for index in range(64):
            primary.create_session(f"session-{index}", "test")
        assert primary._maintenance.wait_idle(timeout_s=10)
        baseline_requested, _baseline_completed, _baseline_state = _fts_job(primary)
        for handle in handles:
            handle._conn.execute("PRAGMA busy_timeout=0")
            handle._WRITE_MAX_RETRIES = 1
            handle._CHECKPOINT_EVERY_N_WRITES = 100_000

        barrier = threading.Barrier(64, timeout=30)

        def append(index):
            barrier.wait()
            return handles[index].append_message(
                f"session-{index}",
                "user",
                f"parallel message {index}",
            )

        with ThreadPoolExecutor(max_workers=64) as pool:
            writes = [pool.submit(append, index) for index in range(64)]
            message_ids = [write.result(timeout=60) for write in writes]

        assert len(set(message_ids)) == 64
        assert primary._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0] == 64
        # Sub-threshold debt is intentionally quiescent at the default merge
        # interval; callers must not wait for completed_seq to catch up.
        assert primary._maintenance.wait_idle(timeout_s=20)
        requested, completed, state = _fts_job(primary)
        assert requested == baseline_requested + 64
        assert completed <= requested
        assert state == "idle"
    finally:
        for handle in handles:
            handle.close()
