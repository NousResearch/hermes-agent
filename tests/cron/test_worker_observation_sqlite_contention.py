"""Actual SQLite read contention is unavailable observation, never claim loss."""
import sqlite3


def test_busy_database_does_not_become_terminal_observation(tmp_path,monkeypatch):
    from cron import scheduler
    from cron.scheduler_worker_failure import observe_worker_execution
    db=tmp_path/'contended.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE fixture (status TEXT)')
        conn.execute("INSERT INTO fixture VALUES ('running')")
    def read(_eid):
        with sqlite3.connect(db,timeout=0) as reader:
            return {'status':reader.execute('SELECT status FROM fixture').fetchone()[0]}
    monkeypatch.setattr(scheduler,'get_execution',read)
    lock=sqlite3.connect(db,timeout=0)
    try:
        lock.execute('BEGIN EXCLUSIVE')
        assert observe_worker_execution('fixture') is None
    finally:
        lock.rollback()
        lock.close()
    assert observe_worker_execution('fixture')=={'status':'running'}
