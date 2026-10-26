"""Transactional contracts against exclusively owned temporary profile databases."""
import sqlite3

import pytest

from hermes_state import SessionDB


def store(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    # This is the baseline behavior assertion, before importing the new API.
    assert db._conn.execute("SELECT name FROM sqlite_master WHERE name='runtime_epoch'").fetchone(), 'canonical runtime schema missing'
    return db


def test_admission_retry_claim_and_restart_are_fenced(tmp_path):
    db = store(tmp_path)
    import hermes_state_runtime as rt
    from gateway.session_admission import admission_fingerprint
    assert admission_fingerprint(canonical_target='s', payload={'a': 1, 'b': 2}) == admission_fingerprint(canonical_target='s', payload={'b': 2, 'a': 1})
    with pytest.raises(ValueError):
        admission_fingerprint(canonical_target='s', payload={'a': float('nan')})
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='one', payload={'text': 'hello'})
        first = rt.admit_session_input(db, **args)
        assert rt.admit_session_input(db, **args) == first
        for change in ({'payload': {'text': 'changed'}}, {'intent': 'steer'}):
            with pytest.raises(rt.RuntimeStoreError) as exc:
                rt.admit_session_input(db, **(args | change))
            assert exc.value.reason == 'admission_conflict'
        second = rt.admit_session_input(db, **(args | {'request_id': 'two'}))
        third = rt.admit_session_input(db, **(args | {'principal_id': 'other'}))
        assert first['seq'] < second['seq'] < third['seq']
        claimed = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assert claimed['admission_id'] == first['admission_id'] and claimed['status'] == 'started'
        assert rt.claim_session_input(db, epoch=epoch, session_id='s') is None
        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.settle_session_input(db, epoch=epoch, admission_id=claimed['admission_id'], generation=claimed['generation'] + 1, outcome='completed')
        assert exc.value.reason == 'stale_generation'
        rt.settle_session_input(db, epoch=epoch, admission_id=claimed['admission_id'], generation=claimed['generation'], outcome='completed')
        running = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assert running['admission_id'] == second['admission_id'] and running['generation'] > claimed['generation']
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        assert new_epoch > epoch
        assert rt.recover_session_inputs(db, epoch=new_epoch) == 1
        assert rt.recover_session_inputs(db, epoch=new_epoch) == 0
        assert rt.get_session_admission(db, admission_id=second['admission_id'])['status'] == 'unknown'
        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.claim_session_input(db, epoch=new_epoch, session_id='s')
        assert exc.value.reason == 'unknown_execution'
        with pytest.raises(rt.RuntimeStoreError) as exc:
            rt.settle_session_input(db, epoch=epoch, admission_id=running['admission_id'], generation=running['generation'], outcome='completed')
        assert exc.value.reason == 'stale_epoch'
        assert rt.cancel_session_input(db, epoch=new_epoch, admission_id=third['admission_id'])['outcome'] == 'cancelled'
        assert [r['admission_id'] for r in rt.list_session_admissions(db, session_id='s')] == [second['admission_id']]
        assert db._conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(lambda conn: conn.execute("DELETE FROM sessions WHERE id='s'"))
    finally:
        db.close()


def test_sqlite_failure_never_acknowledges_or_partially_claims(tmp_path):
    db = store(tmp_path)
    import hermes_state_runtime as rt
    try:
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='one', payload={'text': 'hello'})
        db._conn.execute("CREATE TRIGGER refuse_admission BEFORE INSERT ON session_admissions BEGIN SELECT RAISE(ABORT, 'full fixture'); END")
        with pytest.raises(sqlite3.IntegrityError, match='full fixture'):
            rt.admit_session_input(db, **args)
        assert rt.list_session_admissions(db, session_id='s') == []
        db._conn.execute('DROP TRIGGER refuse_admission')
        row = rt.admit_session_input(db, **args)
        db._conn.execute("CREATE TRIGGER refuse_claim BEFORE UPDATE ON session_admissions BEGIN SELECT RAISE(ABORT, 'claim fixture'); END")
        with pytest.raises(sqlite3.IntegrityError, match='claim fixture'):
            rt.claim_session_input(db, epoch=epoch, session_id='s')
        assert db.get_session('s')['runtime_generation'] == 0
        assert rt.get_session_admission(db, admission_id=row['admission_id'])['status'] == 'queued'
        db._conn.execute('DROP TRIGGER refuse_claim')
        with sqlite3.connect(tmp_path / 'state.db', timeout=0) as peer:
            assert peer.execute('SELECT admission_id FROM session_admissions').fetchone()[0] == row['admission_id']
            peer.execute('BEGIN IMMEDIATE')
            db._WRITE_PATIENCE_S = 0
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                rt.admit_session_input(db, **(args | {'request_id': 'busy'}))
            peer.rollback()
        assert len(rt.list_session_admissions(db, session_id='s')) == 1
    finally:
        db.close()
