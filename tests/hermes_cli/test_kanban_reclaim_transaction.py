"""Worker stops follow the durable outer transaction, including savepoint failure."""
from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db_connect as db
from hermes_cli import kanban_worker_scope as scope
from hermes_cli import kanban_worker_stop as stop


@pytest.mark.parametrize('outcome', ['commit', 'inner_rollback', 'outer_rollback', 'commit_failure', 'unowned_outer'])
def test_stop_intents_follow_durable_row_changes(tmp_path, monkeypatch, outcome):
    path = tmp_path / 'transaction.db'
    conn = db.connect(path)
    peer = db.connect(path)
    conn.execute('CREATE TABLE receipt_probe (id TEXT PRIMARY KEY, value INTEGER)')
    conn.executemany('INSERT INTO receipt_probe VALUES (?, 0)', [('outer',), ('inner',)])
    effects = []
    stop.reset_scope_stop_service_for_tests()
    monkeypatch.setattr(scope, '_kanban_scope_state', lambda unit: 'active')
    monkeypatch.setattr(stop, '_ensure_scope_stop_thread', lambda: None)
    request_stop = stop.request_worker_scope_stop

    def visible():
        return dict(peer.execute('SELECT id, value FROM receipt_probe').fetchall())

    def observed_request(unit, **kwargs):
        if kwargs.get('conn') is None:
            effects.append((unit, visible()))
        return request_stop(unit, **kwargs)

    monkeypatch.setattr(stop, 'request_worker_scope_stop', observed_request)
    boundary = db._execute_boundary_with_retry

    def fail_commit(connection, statement):
        if outcome == 'commit_failure' and statement == 'COMMIT':
            raise sqlite3.OperationalError('synthetic commit refusal')
        return boundary(connection, statement)

    monkeypatch.setattr(db, '_execute_boundary_with_retry', fail_commit)
    try:
        if outcome == 'unowned_outer':
            # A raw outer transaction has no commit notification to flush intents.
            # Nested composition must refuse it instead of silently losing stops.
            conn.execute('BEGIN IMMEDIATE')
            with pytest.raises(RuntimeError, match='transaction|intent|owned'):
                with db.write_txn(conn, allow_nested=True):
                    conn.execute("UPDATE receipt_probe SET value=1 WHERE id='inner'")
                    stop.request_worker_scope_stop('inner.scope', conn=conn)
            conn.execute('ROLLBACK')
            expected = {'outer': 0, 'inner': 0}
            expected_units = []
        else:
            try:
                with db.write_txn(conn):
                    conn.execute("UPDATE receipt_probe SET value=1 WHERE id='outer'")
                    stop.request_worker_scope_stop('outer.scope', conn=conn)
                    try:
                        with db.write_txn(conn, allow_nested=True):
                            conn.execute("UPDATE receipt_probe SET value=1 WHERE id='inner'")
                            stop.request_worker_scope_stop('inner.scope', conn=conn)
                            assert effects == []
                            assert visible() == {'outer': 0, 'inner': 0}
                            if outcome == 'inner_rollback':
                                raise ValueError('inner rollback')
                    except ValueError:
                        assert outcome == 'inner_rollback'
                    assert effects == []
                    assert visible() == {'outer': 0, 'inner': 0}
                    if outcome == 'outer_rollback':
                        raise ValueError('outer rollback')
            except (ValueError, sqlite3.OperationalError):
                assert outcome in ('outer_rollback', 'commit_failure')
            successful = outcome in ('commit', 'inner_rollback')
            expected = {'outer': int(successful), 'inner': int(outcome == 'commit')}
            expected_units = (['outer.scope', 'inner.scope'] if outcome == 'commit'
                              else ['outer.scope'] if outcome == 'inner_rollback' else [])
        assert visible() == expected
        assert effects == [(unit, expected) for unit in expected_units]
        assert not conn.in_transaction
        with stop._scope_stop_lock:
            assert set(stop._scope_stop_pending) == set(expected_units)
    finally:
        if conn.in_transaction:
            conn.rollback()
        stop.reset_scope_stop_service_for_tests()
        peer.close()
        conn.close()
