"""Input custody binds inside the admission transaction, without execution or services."""
import hashlib

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


def test_custody_refs_bind_with_the_admission_and_survive_replay(store):
    db, args = store
    handle = ready(db, args)
    # Local-Submission digest cannot authorize the final API-shaped payload.
    wrong = ready(db, args, payload={'text': 'read'})
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_session_input(db, **args, input_custody=wrong)
    assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    row = admit_session_input(db, **args, input_custody=handle)
    ref = dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone())
    saved = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
    for field in ('admission_id', 'principal_id', 'target_session_id', 'request_id', 'payload_digest', 'intent'):
        assert ref[field] == saved[field]
    assert ref['generation'] == 1
    # Neither an expired lease nor a new unrelated handle changes accepted refs.
    extra = ready(db, args, name='unrelated.txt')
    db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
    for replay_handle in (None, handle, extra):
        replay = admit_session_input(db, **args, input_custody=replay_handle)
        assert replay['admission_id'] == row['admission_id']
        assert dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone()) == ref
    db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='terminal',outcome='completed'"))
    assert admit_session_input(db, **args)['status'] == 'terminal'


def test_preparation_cannot_retrofit_custody_onto_old_admission(store):
    db, args = store
    handle = ready(db, args)
    admit_session_input(db, **args)
    old = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        db._execute_write(lambda conn: accept_prepared_input(conn, epoch=args['epoch'],
            admission=old, handle=handle, new_admission=False))
    assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0

@pytest.mark.parametrize('failure', ['authorization', 'projection'])
def test_composed_writer_commits_authorization_custody_and_logical_evidence_together(store, monkeypatch, failure):
    db, args = store
    handle = ready(db, args)
    guarded = []
    def authorize(conn):
        assert conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 0
        guarded.append(True)
        if failure == 'authorization':
            raise RuntimeStoreError('permission_denied')
    if failure == 'projection':
        db._execute_write(lambda conn: conn.execute("CREATE TRIGGER refuse_projection BEFORE INSERT ON logical_attempts WHEN NEW.admission_id!='' BEGIN SELECT RAISE(ABORT,'projection refused'); END"))
    import sqlite3
    with pytest.raises((RuntimeStoreError, sqlite3.IntegrityError)):
        admit_session_input(db, **args, input_custody=handle, _authorize_write=authorize)
    assert guarded
    for table in ('session_admissions', 'input_custody_refs'):
        assert db._conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0
    assert db._conn.execute("SELECT COUNT(*) FROM logical_attempts WHERE admission_id!=''").fetchone()[0] == 0
    if failure == 'projection':
        db._execute_write(lambda conn: conn.execute('DROP TRIGGER refuse_projection'))
    row = admit_session_input(db, **args, input_custody=handle)
    ref = dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone())
    logical = dict(db._conn.execute("SELECT * FROM logical_attempts WHERE admission_id!=''").fetchone())
    raw = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
    for key in ('admission_id', 'principal_id', 'request_id', 'payload_digest', 'intent'):
        assert ref[key] == logical[key] == raw[key]
    assert logical['session_id'] == ref['target_session_id'] == args['session_id']
    def forbidden(*_args, **_kwargs):
        pytest.fail('exact replay must not reauthorize or reproject an accepted input')
    monkeypatch.setattr('hermes_state_logical_attempts.project_admission', forbidden)
    replay = admit_session_input(db, **args, _authorize_write=forbidden)
    assert replay['admission_id'] == row['admission_id']
    assert dict(db._conn.execute('SELECT * FROM input_custody_refs').fetchone()) == ref
