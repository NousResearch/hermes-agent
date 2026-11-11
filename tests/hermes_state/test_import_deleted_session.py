"""An import never restores a user-deleted session id: it refuses it by name at import time.

The deletion fence is permanent for a user delete, so a row an import put back beside it would list
fine and then refuse every turn. ``import_sessions`` (behind ``adopt_session_lineage_from``) and
the runtime ``import`` mutation both refuse such an id when they read the payload, and write
nothing. An id a housekeeping sweep retired restores once its fence ages out, as before.
"""
import pytest

import hermes_state_runtime as rt
from hermes_state import SessionDB
from hermes_state_mutation_retirement import prune_retired_tombstones


def _exported_then_deleted(db, epoch, sid):
    db.create_session(sid, source='cli')
    db.append_message(sid, role='user', content='keep me')
    exported = db.export_session(sid)
    snap = db.get_session(sid)
    rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid, request_id='del-' + sid,
                              operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                              expected_generation=snap['runtime_generation'])
    db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    return exported


def test_import_sessions_refuses_a_deleted_id_by_name(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        exported = _exported_then_deleted(db, epoch, 's')
        result = db.import_sessions([exported])
        assert result['ok'] is False and result['imported'] == 0, result
        assert any('s' in error and 'deleted' in error for error in result['errors']), result
        assert db.get_session('s') is None, 'the import put the deleted row back beside its fence'


def test_the_runtime_import_refuses_a_deleted_id_by_name(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        exported = _exported_then_deleted(db, epoch, 's')
        with pytest.raises(rt.RuntimeStoreError) as refused:
            rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='s', request_id='restore',
                                      expected_revision=0, operation='import', payload={'sessions': [exported]})
        assert refused.value.reason == 'admission_conflict'
        assert refused.value.field == 'sessions' and 'deleted' in (refused.value.detail or ''), vars(refused.value)
        assert db.get_session('s') is None


def test_a_swept_session_restores_once_its_fence_ages_out(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('swept', source='cron')
        db.append_message('swept', role='user', content='keep me')
        exported = db.export_session('swept')
        db.end_session('swept', 'test')
        assert db.prune_sessions(older_than_days=None) == 1
        db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        assert db.import_sessions([exported])['imported'] == 1
        assert db.get_session('swept') is not None
