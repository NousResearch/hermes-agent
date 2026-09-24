"""#100871: a transient SQLITE_IOERR on a warm pooled read connection is retried on the same
connection, then surfaced -- never quarantined, never a close+reopen."""
import sqlite3

import pytest

import hermes_state
from hermes_state import SessionDB


_STATE = {"failures_left": 0, "attempts": 0}  # module-level: the tracking factory subclasses _FlakyReads


class _FlakyReads(sqlite3.Connection):
    """Real SQLite connection whose first N SELECTs fail the way a mid-checkpoint mode=ro reader does."""

    def execute(self, sql, *args, **kwargs):  # type: ignore[override]
        if str(sql).lstrip().upper().startswith("SELECT"):
            _STATE["attempts"] += 1
            if _STATE["failures_left"] > 0:
                _STATE["failures_left"] -= 1
                raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, *args, **kwargs)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", "cli")
    if not db._wal_active:
        db.close()
        pytest.skip("read pool needs WAL")
    real_connect = hermes_state._connect_tracked_db

    def flaky_connect(path, *args, **kwargs):
        if str(path).startswith("file:"):
            kwargs["factory"] = _FlakyReads
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", flaky_connect)
    while db._evict_one_idle_read_conn():  # the next read opens through the flaky factory
        pass
    _STATE.update(failures_left=0, attempts=0)
    yield db
    db.close()


def test_transient_ioerr_on_pooled_read_is_retried_on_the_same_connection(db):
    _STATE["failures_left"] = 1
    row = db.get_session("s")
    assert row is not None and row["id"] == "s"
    assert _STATE["attempts"] == 2  # one failure, one replay
    assert db._db_corrupt is False and db._db_wal_generation_lost is False


def test_persistent_ioerr_propagates_after_the_budget(db):
    _STATE["failures_left"] = 10 ** 6
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        db.get_session("s")
    assert _STATE["attempts"] == hermes_state._READ_ONLY_IOERR_RETRY_ATTEMPTS + 1
    assert db._db_corrupt is False  # busy/EIO is not corruption: no quarantine
