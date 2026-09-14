"""G custody / runtime NEW-grant composition, without execution or services."""
import hashlib
import sqlite3

import pytest

from gateway.session_admission import admission_fingerprint
from hermes_state import SessionDB
from hermes_state_input_custody import (
    accept_prepared_input, add_copy, begin_preparation, create_schema,
    finish_preparation,
)
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch


def ready(db, args, *, name='one.txt', payload=None):
    data = b'independent fixture bytes'
    def write(conn):
        create_schema(conn)
        handle = begin_preparation(conn, epoch=args['epoch'], principal_id=args['principal_id'],
            session_id=args['session_id'], request_id=args['request_id'])
        copy = add_copy(conn, handle=handle, ordinal=0, name=name,
            digest=hashlib.sha256(data).hexdigest(), size=len(data))
        conn.execute("UPDATE input_custody_copies SET state='ready' WHERE copy_id=?", (copy['copy_id'],))
        digest = admission_fingerprint(canonical_target=args['session_id'],
            payload={'input': payload or args['payload'], 'intent': 'queue'})
        finish_preparation(conn, epoch=args['epoch'], handle=handle, payload_digest=digest)
        return handle
    return db._execute_write(write)


@pytest.fixture
def store(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='api')
        args = dict(epoch=begin_runtime_epoch(db, instance_id='owner'), principal_id='api',
            session_id='s', request_id='request', payload={'text': 'read', 'api_turn_v1': {
                'history': [], 'settings': {}, 'run_owner_scope': 'a' * 64}})
        yield db, args


def test_new_guard_and_custody_share_admission_transaction(store):
    db, args = store
    handle = ready(db, args)
    calls = []
    def deny(conn):
        assert conn.in_transaction
        assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        with sqlite3.connect(db.db_path, timeout=0) as other:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                other.execute('BEGIN IMMEDIATE')
        calls.append(True)
        raise PermissionError('grant revoked')
    with pytest.raises(PermissionError, match='grant revoked'):
        admit_session_input(db, **args, input_custody=handle, _authorize_write=deny)
    assert calls == [True]
    assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    # Local-Submission digest cannot authorize the final API-shaped payload.
    wrong = ready(db, args, payload={'text': 'read'})
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_session_input(db, **args, input_custody=wrong, _authorize_write=lambda conn: None)
    assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    row = admit_session_input(db, **args, input_custody=handle,
        _authorize_write=lambda conn: calls.append(conn.in_transaction))
    assert calls == [True, True]
    ref = dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone())
    saved = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
    for field in ('admission_id', 'principal_id', 'target_session_id', 'request_id', 'payload_digest', 'intent'):
        assert ref[field] == saved[field]
    assert ref['generation'] == 1
    # Neither an expired lease nor a new unrelated handle changes accepted refs.
    extra = ready(db, args, name='unrelated.txt')
    db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
    for replay_handle in (None, handle, extra):
        replay = admit_session_input(db, **args, input_custody=replay_handle, _authorize_write=deny)
        assert replay['admission_id'] == row['admission_id']
        assert dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone()) == ref
    db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='terminal',outcome='completed'"))
    assert admit_session_input(db, **args, _authorize_write=deny)['status'] == 'terminal'


def test_preparation_cannot_retrofit_custody_onto_old_admission(store):
    db, args = store
    handle = ready(db, args)
    admit_session_input(db, **args)
    old = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        db._execute_write(lambda conn: accept_prepared_input(conn, epoch=args['epoch'],
            admission=old, handle=handle, new_admission=False))
    assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
