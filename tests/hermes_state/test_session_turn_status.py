"""Public read-only turn status is conservative across lease lineage and failures."""
import os
import sqlite3
import time
from contextlib import contextmanager

import pytest

from hermes_state import SessionDB


def test_status_uses_writer_lineage_and_fork_boundary(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('root', source='cli')
    db.end_session('root', 'compression')
    db.create_session('child', source='cli', parent_session_id='root')
    db.create_session('fork', source='cli', parent_session_id='root',
                      model_config={'_branched_from': 'root'})
    holder = f'pid={os.getpid()}:turn=first'
    assert db.try_acquire_session_turn_lease('child', holder)
    assert db.get_session_turn_statuses(['root', 'child', 'fork', 'absent']) == {
        'root': True, 'child': True, 'fork': False, 'absent': None}
    db.release_session_turn_lease('root', holder)
    assert db.get_session_turn_statuses(['root', 'child']) == {'root': False, 'child': False}


def test_status_expired_and_dead_local_holder_are_inactive_without_reclaim(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    for sid in ('expired', 'dead'):
        db.create_session(sid, source='cli')
    with db._lock:
        db._conn.execute("INSERT INTO session_turn_leases VALUES (?, ?, ?, ?)",
                         ('expired', f'pid={os.getpid()}', time.time() - 10, time.time() - 1))
        db._conn.execute("INSERT INTO session_turn_leases VALUES (?, ?, ?, ?)",
                         ('dead', 'pid=999999999:turn=test', time.time() - 10, time.time() + 60))
        db._conn.commit()
    assert db.get_session_turn_statuses(['expired', 'dead']) == {'expired': False, 'dead': False}
    with db._read_ctx() as conn:
        assert conn.execute('SELECT count(*) FROM session_turn_leases').fetchone()[0] == 2


def test_status_read_failure_and_bounded_lineage_are_unknown(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('root', source='cli')
    db.end_session('root', 'compression')
    db.create_session('child', source='cli', parent_session_id='root')
    original = db._session_turn_lease_key_on_conn
    def broken(conn, sid, **kwargs):
        if sid == 'child':
            raise sqlite3.OperationalError('database is locked')
        return original(conn, sid, **kwargs)
    monkeypatch.setattr(db, '_session_turn_lease_key_on_conn', broken)
    assert db.get_session_turn_statuses(['root', 'child']) == {'root': False, 'child': None}
    monkeypatch.setattr(db, '_session_turn_lease_key_on_conn', original)
    with db._lock:
        db._conn.execute("UPDATE sessions SET parent_session_id='child' WHERE id='root'")
        db._conn.execute("UPDATE sessions SET end_reason='compression' WHERE id='child'")
        db._conn.commit()
    assert db.get_session_turn_statuses(['root', 'child']) == {'root': None, 'child': None}


def test_status_long_lineage_is_unknown(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('root', source='cli')
    previous = 'root'
    for n in range(101):
        db.end_session(previous, 'compression')
        child = f'child-{n}'
        db.create_session(child, source='cli', parent_session_id=previous)
        previous = child
    assert db.get_session_turn_statuses([previous]) == {previous: None}


def test_status_rejects_oversized_batch_and_is_profile_scoped(tmp_path):
    a = SessionDB(tmp_path / 'A' / 'state.db')
    b = SessionDB(tmp_path / 'B' / 'state.db')
    for db in (a, b):
        db.create_session('same', source='cli')
    assert a.try_acquire_session_turn_lease('same', f'pid={os.getpid()}:turn=A')
    assert a.get_session_turn_statuses(['same']) == {'same': True}
    assert b.get_session_turn_statuses(['same']) == {'same': False}
    assert a.get_session_turn_statuses(['same']) == {'same': True}
    assert SessionDB(tmp_path / 'A' / 'state.db', read_only=True).get_session_turn_statuses(['same']) == {'same': True}
    with pytest.raises(ValueError):
        a.get_session_turn_statuses([str(i) for i in range(101)])

def test_status_cooperative_deadline_does_not_bound_connection_checkout(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('waiting', source='cli')
    original = db._read_ctx

    @contextmanager
    def delayed_checkout():
        time.sleep(0.6)
        with original() as conn:
            yield conn

    monkeypatch.setattr(db, '_read_ctx', delayed_checkout)
    started = time.monotonic()
    assert db.get_session_turn_statuses(['waiting']) == {'waiting': None}
    assert time.monotonic() - started >= 0.5
