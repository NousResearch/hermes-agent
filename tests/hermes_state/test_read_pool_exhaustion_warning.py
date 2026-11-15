"""Read-pool exhaustion is visible at WARNING, rate-limited (field report: NFS stalls).

When every pooled read permit is out, ``_get_read_conn`` serves the read from the writer connection under
``SessionDB._lock``: that read then queues behind every writer, including one parked on a slow ``-shm`` lock.
The message used to be DEBUG, invisible in a production log. It is now a WARNING, at most one per database
per window, carrying a count of the reads that degraded since the last one. No SQL or parameters are logged.
"""

import logging

import pytest

import hermes_state_readpool
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session("s", source="test")
    yield handle
    handle.close()


def _exhaust(monkeypatch, db):
    monkeypatch.setattr(db, "_wal_active", True)
    monkeypatch.setattr(db._read_budget, "acquire", lambda requester: False)


def test_pool_exhaustion_warns_once_per_window_with_a_count(db, monkeypatch, caplog):
    _exhaust(monkeypatch, db)
    clock = [1000.0]
    monkeypatch.setattr(hermes_state_readpool.time, "monotonic", lambda: clock[0], raising=False)
    getattr(hermes_state_readpool, "_pool_exhaustion_warned_at", {}).clear()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        for _ in range(5):
            assert db._get_read_conn() is None
            assert db.get_session("s") is not None  # still served, from the locked writer
        clock[0] += 61.0
        assert db._get_read_conn() is None
    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and "read pool at capacity" in r.getMessage()]
    assert len(warnings) == 2, [r.getMessage() for r in warnings]
    assert "state.db" in warnings[0].getMessage()
    assert "locked writer connection" in warnings[0].getMessage()
    # The second window reports how many reads degraded since the first warning.
    assert "reads since the last warning" in warnings[1].getMessage()


def test_pool_exhaustion_count_is_per_database(tmp_path, monkeypatch, caplog):
    getattr(hermes_state_readpool, "_pool_exhaustion_warned_at", {}).clear()
    a, b = SessionDB(tmp_path / "a" / "state.db"), SessionDB(tmp_path / "b" / "state.db")
    try:
        for handle in (a, b):
            _exhaust(monkeypatch, handle)
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            a._get_read_conn()
            b._get_read_conn()
            a._get_read_conn()
    finally:
        a.close()
        b.close()
    warnings = [r for r in caplog.records if "read pool at capacity" in r.getMessage()]
    assert len(warnings) == 2  # one per database, the repeat on `a` suppressed
