"""Action-specific mutations share the canonical receipt transaction."""
import sqlite3
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt


def test_sidebar_edit_is_atomic_and_revision_fenced(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit',
                    expected_revision=0, operation='sidebar',
                    payload={'title': 'Title', 'pinned': True, 'hidden': True, 'unread': True})
        db._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
            rt.mutate_runtime_session(db, **args)
        assert db.get_session('s')['title'] is None
        assert not db.get_session('s')['pinned']
        db._execute_write(lambda c: c.execute('DROP TRIGGER fail_receipt'))
        receipt = rt.mutate_runtime_session(db, **args)
        assert receipt['revision'] == 1
        assert receipt['pinned'] and receipt['hidden'] and receipt['unread']
        with SessionDB(db_path=tmp_path / 'state.db') as peer:
            with pytest.raises(rt.RuntimeStoreError, match='revision_conflict'):
                rt.mutate_runtime_session(peer, **(args | {'principal_id': 'peer'}))
            assert rt.mutate_runtime_session(peer, **args) == receipt
        assert db.get_session('s')['runtime_revision'] == 1
