import sqlite3

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def append(db, *, epoch, sequence, content, **assignment):
    return rt.mutate_worker_execution(db, epoch=epoch, **assignment, sequence=sequence,
        operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': content}]})


@pytest.mark.parametrize('recover_first', [False, True])
def test_explicit_worker_adoption_preserves_claim_without_reexecution(tmp_path, recover_first):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        accepted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'work'})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assignment = dict(execution_id='compute', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
        new_epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        if recover_first:
            rt.recover_session_inputs(db, epoch=new_epoch)
        rt.adopt_worker_execution(db, epoch=new_epoch, **assignment, adoption_secret='private')
        rt.recover_session_inputs(db, epoch=new_epoch)
        restored = rt.get_session_admission(db, admission_id=accepted['admission_id'])
        assert restored['status'] == 'started' and restored['owner_epoch'] == new_epoch
        assert restored['generation'] == claim['generation']
        assert rt.claim_session_input(db, epoch=new_epoch, session_id='s') is None
        message_args = dict(epoch=new_epoch, **assignment, sequence=1, content='committed')
        receipt = append(db, **message_args)
        result = rt.settle_session_input(db, epoch=new_epoch, admission_id=accepted['admission_id'], generation=claim['generation'], outcome='completed')
        assert result['status'] == 'terminal'
        before = db._conn.total_changes
        assert append(db, **message_args) == receipt
        assert db._conn.total_changes == before
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            append(db, epoch=new_epoch, **assignment, sequence=2, content='late')
    finally:
        db.close()


@pytest.mark.parametrize('refuse_revoke', [False, True])
def test_unknown_resolution_atomically_revokes_worker_before_unblocking(tmp_path, refuse_revoke):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        first = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='first', payload={})
        claim = rt.claim_session_input(db, epoch=epoch, session_id='s')
        assignment = dict(execution_id='worker', session_id='s', generation=claim['generation'])
        rt.register_worker_execution(db, epoch=epoch, **assignment, kind='compute', adoption_secret='private')
        follower = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='next', payload={})
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        rt.recover_session_inputs(db, epoch=epoch)
        args = dict(epoch=epoch, admission_id=first['admission_id'], generation=claim['generation'])
        before = db.get_session('s')
        if refuse_revoke:
            db._execute_write(lambda c: c.execute("CREATE TRIGGER refuse_revoke BEFORE UPDATE OF status ON worker_executions BEGIN SELECT RAISE(ABORT, 'revoke refused'); END"))
            with pytest.raises(sqlite3.IntegrityError, match='revoke refused'):
                rt.resolve_unknown_session_input(db, **args)
            assert rt.get_session_admission(db, admission_id=first['admission_id'])['status'] == 'unknown'
            assert db.get_session('s') == before
            db._execute_write(lambda c: c.execute('DROP TRIGGER refuse_revoke'))
        result = rt.resolve_unknown_session_input(db, **args)
        assert result['status'] == 'terminal' and result['outcome'] == 'interrupted'
        db.close()
        db = SessionDB(db_path=tmp_path / 'state.db')
        with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
            rt.adopt_worker_execution(db, epoch=epoch, **assignment, adoption_secret='private')
        with pytest.raises(rt.RuntimeStoreError, match='stale_epoch'):
            append(db, epoch=epoch, **assignment, sequence=1, content='late')
        assert db.get_messages('s') == []
        assert rt.claim_session_input(db, epoch=epoch, session_id='s')['admission_id'] == follower['admission_id']
    finally:
        db.close()
