"""Best-effort activity and lease renewal must not wait forever on the local writer mutex."""
import sqlite3
import threading

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("operation", ["activity", "lease"])
def test_write_patience_bounds_local_mutex_wait(tmp_path, operation):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("owned", source="test")
    assert db.acquire_session_turn_lease("owned", "holder", ttl_seconds=300)
    entered = threading.Event()
    done = threading.Event()
    errors = []

    def write():
        entered.set()
        try:
            if operation == "activity":
                db.touch_session_activity("owned", 123.0, description="receiving stream response")
            else:
                db.refresh_session_turn_lease("owned", "holder", patience_s=0.05)
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    db._lock.acquire()
    worker = threading.Thread(target=write, daemon=True)
    try:
        worker.start()
        assert entered.wait(5), "writer thread never started"
        completed_while_mutex_held = done.wait(2)
    finally:
        db._lock.release()
        worker.join(5)
        db.close()
    assert not worker.is_alive(), "test writer did not settle after releasing the mutex"
    assert completed_while_mutex_held, "observation/lease write ignored its patience while the Python mutex stayed held"
    assert len(errors) == 1 and isinstance(errors[0], sqlite3.OperationalError)
    assert "locked" in str(errors[0]).lower()


def test_uncontended_writes_still_commit_and_return_values(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("owned", source="test")
        db.touch_session_activity("owned", 123.0, description="receiving stream response")
        row = db.get_session("owned")
        assert row is not None
        assert row["last_activity_description"] == "receiving stream response"
        assert db.acquire_session_turn_lease("owned", "holder", ttl_seconds=300)
        assert db.refresh_session_turn_lease("owned", "holder", patience_s=0.05)
        assert not db.refresh_session_turn_lease("owned", "foreign", patience_s=0.05)
