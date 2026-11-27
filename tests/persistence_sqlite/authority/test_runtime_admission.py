"""Transactional contracts against exclusively owned temporary profile databases."""
import sqlite3

import pytest

from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import (
    admit_session_input, cancel_session_input, claim_session_input, get_session_admission, list_session_admissions,
    recover_session_inputs, settle_session_input,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import get_session


def _holder(store):
    store.sessions.create(NewSession(session_id='s', source='test'))
    holder = store.holder()
    # This is the baseline behavior assertion, before importing the new API.
    assert holder._conn.execute("SELECT name FROM sqlite_master WHERE name='runtime_epoch'").fetchone(), 'canonical runtime schema missing'
    return holder


def test_admission_retry_claim_and_restart_are_fenced(backend, store):
    holder = _holder(store)
    from persistence.authority.vocabulary import admission_fingerprint
    assert admission_fingerprint(canonical_target='s', payload={'a': 1, 'b': 2}) == admission_fingerprint(canonical_target='s', payload={'b': 2, 'a': 1})
    with pytest.raises(ValueError):
        admission_fingerprint(canonical_target='s', payload={'a': float('nan')})
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='one', payload={'text': 'hello'})
    first, _ = admit_session_input(holder, **args)
    assert admit_session_input(holder, **args)[0] == first
    for change in ({'payload': {'text': 'changed'}}, {'intent': 'steer'}):
        with pytest.raises(AuthorityRefusal) as exc:
            admit_session_input(holder, **(args | change))
        assert exc.value.reason == 'admission_conflict'
    second, _ = admit_session_input(holder, **(args | {'request_id': 'two'}))
    third, _ = admit_session_input(holder, **(args | {'principal_id': 'other'}))
    assert first['seq'] < second['seq'] < third['seq']
    claimed = claim_session_input(holder, epoch=epoch, session_id='s')
    assert claimed['admission_id'] == first['admission_id'] and claimed['status'] == 'started'
    assert claim_session_input(holder, epoch=epoch, session_id='s') is None
    with pytest.raises(AuthorityRefusal) as exc:
        settle_session_input(holder, epoch=epoch, admission_id=claimed['admission_id'], generation=claimed['generation'] + 1, outcome='completed')
    assert exc.value.reason == 'stale_generation'
    settle_session_input(holder, epoch=epoch, admission_id=claimed['admission_id'], generation=claimed['generation'], outcome='completed')
    running = claim_session_input(holder, epoch=epoch, session_id='s')
    assert running['admission_id'] == second['admission_id'] and running['generation'] > claimed['generation']
    backend.reopen()
    holder = store.holder()
    new_epoch = begin_runtime_epoch(holder, instance_id='replacement')
    assert new_epoch > epoch
    assert recover_session_inputs(holder, epoch=new_epoch) == 1
    assert recover_session_inputs(holder, epoch=new_epoch) == 0
    assert get_session_admission(holder, admission_id=second['admission_id'])['status'] == 'unknown'
    with pytest.raises(AuthorityRefusal) as exc:
        claim_session_input(holder, epoch=new_epoch, session_id='s')
    assert exc.value.reason == 'unknown_execution'
    with pytest.raises(AuthorityRefusal) as exc:
        settle_session_input(holder, epoch=epoch, admission_id=running['admission_id'], generation=running['generation'], outcome='completed')
    assert exc.value.reason == 'stale_epoch'
    assert cancel_session_input(holder, epoch=new_epoch, admission_id=third['admission_id'])['outcome'] == 'cancelled'
    assert [r['admission_id'] for r in list_session_admissions(holder, session_id='s')] == [second['admission_id']]
    assert holder._conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        holder._execute_write(lambda conn: conn.execute("DELETE FROM sessions WHERE id='s'"))


def test_sqlite_failure_never_acknowledges_or_partially_claims(store):
    holder = _holder(store)
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='one', payload={'text': 'hello'})
    holder._conn.execute("CREATE TRIGGER refuse_admission BEFORE INSERT ON session_admissions BEGIN SELECT RAISE(ABORT, 'full fixture'); END")
    with pytest.raises(sqlite3.IntegrityError, match='full fixture'):
        admit_session_input(holder, **args)
    assert list_session_admissions(holder, session_id='s') == []
    holder._conn.execute('DROP TRIGGER refuse_admission')
    row, _ = admit_session_input(holder, **args)
    holder._conn.execute("CREATE TRIGGER refuse_claim BEFORE UPDATE ON session_admissions BEGIN SELECT RAISE(ABORT, 'claim fixture'); END")
    with pytest.raises(sqlite3.IntegrityError, match='claim fixture'):
        claim_session_input(holder, epoch=epoch, session_id='s')
    assert get_session(holder, 's')['runtime_generation'] == 0
    assert get_session_admission(holder, admission_id=row['admission_id'])['status'] == 'queued'
    holder._conn.execute('DROP TRIGGER refuse_claim')
    with sqlite3.connect(store.store_path, timeout=0) as peer:
        assert peer.execute('SELECT admission_id FROM session_admissions').fetchone()[0] == row['admission_id']
        peer.execute('BEGIN IMMEDIATE')
        holder._WRITE_PATIENCE_S = 0
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            admit_session_input(holder, **(args | {'request_id': 'busy'}))
        peer.rollback()
    assert len(list_session_admissions(holder, session_id='s')) == 1
