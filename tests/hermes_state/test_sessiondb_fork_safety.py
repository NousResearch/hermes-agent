"""Process-boundary contracts for ``SessionDB`` SQLite handles."""

import os
import signal
import threading

import pytest

from hermes_cli import sqlite_safe_read
from hermes_state import SessionDB


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_inherited_sessiondb_fails_fast_and_child_can_reopen(tmp_path):
    """A fork child must reject inherited handles but may open its own."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    tracking_lock_held = threading.Event()
    release_tracking_lock = threading.Event()

    def hold_tracking_lock():
        # Fork at the hostile boundary: both guards are owned by a thread that
        # will not exist in the child. Neither inherited instance rejection
        # nor the child's fresh tracked open may wait on these orphaned locks.
        with sqlite_safe_read._live_lock, db._lock:
            tracking_lock_held.set()
            release_tracking_lock.wait(timeout=10)

    holder = threading.Thread(target=hold_tracking_lock)
    holder.start()
    assert tracking_lock_held.wait(timeout=10)

    child_pid = os.fork()  # windows-footgun: ok — test is skip-gated above
    if child_pid == 0:  # pragma: no cover - assertions execute in child
        try:
            signal.alarm(10)
            try:
                db.create_session("unsafe-child-write", source="test")
            except RuntimeError as exc:
                if "cannot be reused after fork" not in str(exc):
                    os._exit(2)
            else:
                os._exit(3)

            try:
                db.get_session("parent")
            except RuntimeError as exc:
                if "cannot be reused after fork" not in str(exc):
                    os._exit(4)
            else:
                os._exit(5)

            # Closing an inherited instance must not sqlite3_close() the
            # parent's handle. The child owns only handles it opens itself.
            db.close()
            child_db = SessionDB(db_path=db_path)
            child_db.create_session("child", source="test")
            child_db.close()
            os._exit(0)
        except BaseException:
            os._exit(6)

    try:
        _, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release_tracking_lock.set()
        holder.join(timeout=10)

    try:
        db.create_session("parent", source="test")
        assert db.get_session("parent") is not None
        assert db.get_session("child") is not None
    finally:
        db.close()
