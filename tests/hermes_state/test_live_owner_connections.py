"""Live owner connection lifetime and transaction fences for dependent stores."""
import sqlite3
import threading

import pytest

from hermes_state import SessionDB, StateDbCorruptError, StateDbReplacedError


def test_live_read_borrows_owner_and_fences_close_without_reopening(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    close_started = threading.Event()
    closed = threading.Event()
    errors = []

    def close_in_thread():
        close_started.set()
        try:
            db.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            closed.set()

    try:
        with db.live_read_connection() as conn:
            assert conn is db._conn
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            closer = threading.Thread(target=close_in_thread)
            closer.start()
            assert close_started.wait(2)
            assert not closed.wait(0.05)
        assert closed.wait(5)
        closer.join(timeout=2)
        assert not errors
        with db.live_read_connection() as conn:
            assert conn is None
        assert db._conn is None
    finally:
        if not closed.is_set():
            db.close()


@pytest.mark.parametrize("failure", [None, "caller", "foreign", "interrupt"])
def test_live_write_commits_once_or_rolls_back_without_callback_replay(tmp_path, failure):
    db = SessionDB(db_path=tmp_path / "state.db")
    foreign_path = tmp_path / "foreign.db"
    foreign_path.write_bytes(b"not an SQLite database")
    foreign = sqlite3.connect(foreign_path)
    attempts = []
    try:
        def dependent_write():
            with db.live_write_connection() as conn:
                attempts.append(conn)
                assert conn is db._conn and conn.in_transaction
                conn.execute("INSERT INTO state_meta(key,value) VALUES('live.owner','yes')")
                if failure == "caller":
                    raise ValueError("caller refused")
                if failure == "foreign":
                    # A real error on the dependent store cannot quarantine state.db.
                    foreign.execute("SELECT name FROM sqlite_master").fetchall()
                if failure == "interrupt":
                    raise KeyboardInterrupt("caller stopped")

        if failure is None:
            dependent_write()
        else:
            with pytest.raises((ValueError, sqlite3.DatabaseError, KeyboardInterrupt)):
                dependent_write()
        assert len(attempts) == 1
        if failure is None:
            queued = threading.Event()
            writer_finished = threading.Event()
            writer_errors = []

            def competing_writer():
                queued.set()
                try:
                    db._execute_write(lambda conn: conn.execute(
                        "INSERT INTO state_meta(key,value) VALUES('live.competing','yes')"))
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    writer_finished.set()

            with db.live_write_connection() as held:
                held.execute("INSERT INTO state_meta(key,value) VALUES('live.held','yes')")
                writer = threading.Thread(target=competing_writer)
                writer.start()
                assert queued.wait(2)
                assert not writer_finished.wait(0.05)
            assert writer_finished.wait(5)
            writer.join(timeout=2)
            assert not writer_errors
            assert db.get_meta("live.held") == db.get_meta("live.competing") == "yes"
        with sqlite3.connect(db.db_path) as peer:
            row = peer.execute("SELECT value FROM state_meta WHERE key='live.owner'").fetchone()
            assert row == (("yes",) if failure is None else None)
        assert not db._db_corrupt
        with db.live_write_connection() as conn:
            assert conn is db._conn
            conn.execute("INSERT INTO state_meta(key,value) VALUES('live.next','yes')")
        assert db.get_meta("live.next") == "yes"
        db._db_replaced = True
        try:
            with pytest.raises(StateDbReplacedError):
                with db.live_write_connection():
                    pytest.fail("replaced generation admitted")
        finally:
            db._db_replaced = False
        db._db_corrupt = True
        try:
            with pytest.raises(StateDbCorruptError):
                with db.live_write_connection():
                    pytest.fail("corrupt owner admitted a write")
        finally:
            db._db_corrupt = False
        db.close()
        with pytest.raises(sqlite3.ProgrammingError):
            with db.live_write_connection():
                pytest.fail("closed owner reopened")
        assert db._conn is None
        readonly = SessionDB(db_path=db.db_path, read_only=True)
        try:
            with pytest.raises(sqlite3.ProgrammingError):
                with readonly.live_write_connection():
                    pytest.fail("read-only owner admitted a write")
        finally:
            readonly.close()
    finally:
        foreign.close()
        db.close()
