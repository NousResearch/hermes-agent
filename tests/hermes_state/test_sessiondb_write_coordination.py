"""Concurrency contracts for process-local state.db writers."""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

import pytest

from hermes_state import SessionDB


class _NotifyingLock:
    def __init__(self, lock, attempted):
        self._lock = lock
        self._attempted = attempted

    def __enter__(self):
        self.acquire()
        return self

    def acquire(self):
        self._attempted.set()
        return self._lock.acquire()

    def release(self):
        return self._lock.release()

    def __exit__(self, *_args):
        self.release()


def test_waiting_on_one_connection_does_not_occupy_global_writer_lane(tmp_path):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    second = SessionDB(db_path=db_path)
    first_started = threading.Event()
    first_lock_attempted = threading.Event()
    real_first_lock = first._lock
    first._lock = _NotifyingLock(real_first_lock, first_lock_attempted)

    def write_first():
        first_started.set()
        first._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("first", "test", time.time()),
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            real_first_lock.acquire()
            try:
                blocked = pool.submit(write_first)
                assert first_started.wait(timeout=10)
                assert first_lock_attempted.wait(timeout=10)
                independent = pool.submit(
                    second._execute_write,
                    lambda conn: conn.execute(
                        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                        ("second", "test", time.time()),
                    ),
                )
                independent.result(timeout=2)
            finally:
                real_first_lock.release()
            blocked.result(timeout=10)
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_inherited_sessiondb_fails_fast_and_child_can_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    lock_held = threading.Event()
    release = threading.Event()

    def hold_writer_lock():
        with db._write_lock:
            lock_held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_writer_lock)
    holder.start()
    assert lock_held.wait(timeout=10)

    child_pid = os.fork()  # windows-footgun: ok - test is skipif-gated above
    if child_pid == 0:  # pragma: no cover - assertions run in child process
        try:
            try:
                db._execute_write(lambda conn: conn.execute("SELECT 1"))
            except RuntimeError as exc:
                if "cannot be reused after fork" not in str(exc):
                    os._exit(2)
            else:
                os._exit(3)

            child_db = SessionDB(db_path=db_path)
            grandchild_pid = os.fork()  # windows-footgun: ok - inherited skipif gate
            if grandchild_pid == 0:
                try:
                    for inherited_db in (db, child_db):
                        try:
                            inherited_db._execute_write(
                                lambda conn: conn.execute("SELECT 1")
                            )
                        except RuntimeError:
                            pass
                        else:
                            os._exit(5)
                    grandchild_db = SessionDB(db_path=db_path)
                    grandchild_db.create_session("grandchild", "test")
                    grandchild_db.close()
                    os._exit(0)
                except BaseException:
                    os._exit(6)
            _, grandchild_status = os.waitpid(grandchild_pid, 0)
            if os.waitstatus_to_exitcode(grandchild_status) != 0:
                os._exit(7)

            child_db._execute_write(
                lambda conn: conn.execute(
                    "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                    ("child", "test", time.time()),
                )
            )
            child_db.close()
            import gc
            import hermes_state as state_module
            import weakref

            retained_count = len(state_module._FORK_RETAINED_CONNECTIONS)
            if retained_count < 1:
                os._exit(10)
            inherited_db_ref = weakref.ref(db)
            del holder
            del hold_writer_lock
            del db
            gc.collect()
            if inherited_db_ref() is not None:
                os._exit(8)
            if len(state_module._FORK_RETAINED_CONNECTIONS) != retained_count:
                os._exit(9)
            os._exit(0)
        except BaseException:
            os._exit(4)

    try:
        _, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        holder.join(timeout=10)
        db.close()


def test_raw_state_writers_join_sessiondb_writer_lane(tmp_path, monkeypatch):
    import hermes_state
    from gateway import delivery_ledger
    from tools import async_delegation

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: db_path)
    monkeypatch.setattr(async_delegation, "_db_path", lambda: db_path)
    ledger_entered = threading.Event()
    delegation_entered = threading.Event()
    ledger_lock_attempted = threading.Event()
    delegation_lock_attempted = threading.Event()
    writer_lock = db._write_lock
    monkeypatch.setattr(
        delivery_ledger,
        "_db_write_lock",
        lambda: _NotifyingLock(writer_lock, ledger_lock_attempted),
    )
    monkeypatch.setattr(
        hermes_state,
        "get_process_db_write_lock",
        lambda _path: _NotifyingLock(writer_lock, delegation_lock_attempted),
    )

    def write_ledger():
        delivery_ledger.record_obligation(
            obligation_id="obligation",
            session_key="agent:main:test",
            platform="test",
            chat_id="chat",
            thread_id=None,
            content="answer",
        )
        ledger_entered.set()

    def take_delegation_lock():
        with async_delegation._DB_LOCK:
            delegation_entered.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            with db._write_lock:
                ledger = pool.submit(write_ledger)
                delegation = pool.submit(take_delegation_lock)
                assert ledger_lock_attempted.wait(timeout=10)
                assert delegation_lock_attempted.wait(timeout=10)
                assert not ledger_entered.is_set()
                assert not delegation_entered.is_set()
            ledger.result(timeout=10)
            delegation.result(timeout=10)
        assert ledger_entered.is_set()
        assert delegation_entered.is_set()
    finally:
        db.close()


def test_same_database_instances_serialize_write_transactions(tmp_path):
    """Two SessionDB handles must not race as independent SQLite writers."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    db_path = real_dir / "state.db"
    first = SessionDB(db_path=db_path)
    # A symlinked spelling of the same file must join the same coordinator.
    second = SessionDB(db_path=link_dir / "state.db")
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    # Make an uncoordinated second BEGIN fail immediately and deterministically.
    first._conn.execute("PRAGMA busy_timeout=0")
    second._conn.execute("PRAGMA busy_timeout=0")
    first._WRITE_MAX_RETRIES = 1
    second._WRITE_MAX_RETRIES = 1

    def hold_first_transaction(conn):
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("first", "test", time.time()),
        )
        entered.set()
        assert release.wait(timeout=10)

    def write_second():
        second_started.set()
        second._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("second", "test", time.time()),
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_write = pool.submit(first._execute_write, hold_first_transaction)
            assert entered.wait(timeout=10)
            second_write = pool.submit(write_second)
            assert second_started.wait(timeout=10)
            # Give an uncoordinated writer enough time to hit BEGIN IMMEDIATE.
            time.sleep(0.05)
            release.set()
            first_write.result(timeout=10)
            second_write.result(timeout=10)

        rows = first._conn.execute(
            "SELECT id FROM sessions WHERE id IN ('first', 'second') ORDER BY id"
        ).fetchall()
        assert [row[0] for row in rows] == ["first", "second"]
    finally:
        release.set()
        first.close()
        second.close()
