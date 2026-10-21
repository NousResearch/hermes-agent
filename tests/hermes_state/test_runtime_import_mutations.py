"""Import rows and retry receipt either commit together or not at all."""
import sqlite3
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_import_rolls_back_rows_with_receipt_failure(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(epoch=epoch, principal_id='human', session_id='first', request_id='import',
                    expected_revision=0, operation='import', payload={'sessions': [
                        {'id': 'first', 'source': 'cli', 'messages': [{'role': 'user', 'content': 'keep'}]},
                        {'id': 'second', 'source': 'cli', 'parent_session_id': 'first', 'messages': []}]})
        db._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
            rt.mutate_runtime_session(db, **args)
        assert db.get_session('first') is None and db.get_session('second') is None
        db._execute_write(lambda c: c.execute('DROP TRIGGER fail_receipt'))
        result = rt.mutate_runtime_session(db, **args)
        assert result['imported_ids'] == ['first', 'second']
        assert db.get_session('second')['parent_session_id'] == 'first'
        assert db.get_messages('first')[0]['content'] == 'keep'
        assert rt.mutate_runtime_session(db, **args) == result
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.mutate_runtime_session(db, **(args | {'payload': {'sessions': []}}))


def test_import_never_overwrites_existing_live_rows(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('live', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='live', request_id='queued', payload={'text': 'keep'})
        before = db.get_session('live')
        result = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='new',
            request_id='import', expected_revision=0, operation='import', payload={'sessions': [
                {'id': 'new', 'source': 'cli', 'messages': []},
                {'id': 'live', 'source': 'cli', 'title': 'Overwrite', 'messages': []}]})
        assert result['skipped_ids'] == ['live']
        assert db.get_session('live') == before
        assert rt.list_session_admissions(db, session_id='live')[0]['status'] == 'queued'
