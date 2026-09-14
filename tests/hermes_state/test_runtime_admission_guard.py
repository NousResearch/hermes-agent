"""Private NEW-write authorization prerequisite for #106742; no executor."""
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch


def test_private_guard_owns_insert_transaction_and_denial_rolls_back(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='api')
        args = dict(epoch=begin_runtime_epoch(db, instance_id='owner'),
                    principal_id='api', session_id='s', request_id='request',
                    payload={'text': 'hello'})
        calls = []

        def guard(conn):
            assert conn.in_transaction
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            with sqlite3.connect(db.db_path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute('BEGIN IMMEDIATE')
            conn.execute("INSERT INTO state_meta(key,value) VALUES('guard-test','checked')")
            calls.append(True)
            raise PermissionError('denied at serialization')

        with pytest.raises(PermissionError, match='denied at serialization'):
            admit_session_input(db, **args, _authorize_write=guard)
        assert calls == [True]
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert db._conn.execute("SELECT value FROM state_meta WHERE key='guard-test'").fetchone() is None
        row = admit_session_input(db, **args)
        assert row['payload'] == args['payload']
    finally:
        db.close()


@pytest.mark.parametrize('terminal', [False, True])
def test_private_guard_never_reauthorizes_exact_existing_work(tmp_path, terminal):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='cli')
        args = dict(epoch=begin_runtime_epoch(db, instance_id='owner'),
                    principal_id='native', session_id='s', request_id='request',
                    payload={'text': 'hello'})
        calls = []
        row = admit_session_input(db, **args, _authorize_write=lambda conn: calls.append(conn.in_transaction))
        assert calls == [True]
        if terminal:
            db._execute_write(lambda conn: conn.execute(
                "UPDATE session_admissions SET status='terminal',outcome='completed' WHERE admission_id=?",
                (row['admission_id'],)))

        def deny(conn):
            raise PermissionError('must not run for a replay')

        replay = admit_session_input(db, **args, _authorize_write=deny)
        assert replay['admission_id'] == row['admission_id']
        assert replay['payload'] == row['payload']
        assert replay['status'] == ('terminal' if terminal else 'queued')
        with pytest.raises(ValueError, match='admission_conflict'):
            admit_session_input(db, **(args | {'payload': {'text': 'changed'}}), _authorize_write=deny)
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1
    finally:
        db.close()
