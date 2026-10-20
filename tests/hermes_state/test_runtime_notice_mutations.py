"""Externally supervised operation notices use the canonical mutation receipt."""
import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_notice_and_receipt_survive_owner_restart_without_duplicate(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='first')
        args = dict(principal_id='operator', session_id='s', request_id='rollout:final',
                    expected_revision=0, expected_generation=0, operation='notice',
                    payload={'text': '[Delivery test] operation passed'})
        receipt = rt.mutate_runtime_session(db, epoch=epoch, **args)
        assert receipt['notice_row_id'] > 0
    with SessionDB(db_path=path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
        assert rt.mutate_runtime_session(db, epoch=epoch, **args) == receipt
        rows = db.get_messages('s')
        assert len(rows) == 1
        assert rows[0]['content'] == args['payload']['text']
        assert rows[0]['display_kind'] == 'internal_notification'
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.mutate_runtime_session(db, epoch=epoch, **(args | {'payload': {'text': 'changed'}}))


@pytest.mark.parametrize('outstanding', ['queued', 'started', 'unknown'])
def test_notice_never_writes_through_an_outstanding_turn(tmp_path, outstanding):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='first')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                               request_id='input', payload={'text': 'busy'})
        if outstanding != 'queued':
            rt.claim_session_input(db, epoch=epoch, session_id='s')
        if outstanding == 'unknown':
            epoch = rt.begin_runtime_epoch(db, instance_id='replacement')
            rt.recover_session_inputs(db, epoch=epoch)
        row = db.get_session('s')
        with pytest.raises(rt.RuntimeStoreError, match='session_busy|unknown_execution'):
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='operator', session_id='s',
                request_id='notice', expected_revision=row['runtime_revision'],
                expected_generation=row['runtime_generation'], operation='notice', payload={'text': 'notice'})
        assert db.get_messages('s') == []


def test_notice_requires_generation_and_rolls_back_with_receipt(tmp_path):
    import sqlite3
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='first')
        args = dict(epoch=epoch, principal_id='operator', session_id='s', request_id='notice',
                    expected_revision=0, operation='notice', payload={'text': 'notice'})
        with pytest.raises(rt.RuntimeStoreError, match='invalid_params'):
            rt.mutate_runtime_session(db, **args)
        db._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta "
            "WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
            rt.mutate_runtime_session(db, expected_generation=0, **args)
        assert db.get_messages('s') == []
        assert db.get_session('s')['runtime_revision'] == 0
