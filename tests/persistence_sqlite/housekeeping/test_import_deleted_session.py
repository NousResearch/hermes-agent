"""An import into a SQLite backend store never restores a user-deleted session id: it refuses it by name at import time.

The deletion fence is permanent for a user delete, so a row an import put back beside it would list
fine and then refuse every turn. The maintenance repository's ``import_sessions`` and the runtime
``import`` mutation (``authority._mutation.mutate_runtime_session``) both refuse such an id when they
read the payload, and write nothing. An id a housekeeping sweep retired restores once its fence ages
out, as before.
"""
import pytest

from hermes_cli.session_export import to_legacy_export_dict
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from persistence.sqlite.housekeeping._retirement import prune_retired_tombstones
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def _exported_then_deleted(db, store, epoch, sid):
    create_session(db, sid, source='cli')
    append_message(db, sid, role='user', content='keep me')
    exported = to_legacy_export_dict(store.maintenance.export_session(sid))
    snap = get_session(db, sid)
    mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid, request_id='del-' + sid,
                           operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                           expected_generation=snap['runtime_generation'], media_root=MEDIA_ROOT)
    db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    return exported


def test_import_sessions_refuses_a_deleted_id_by_name(db, store):
    epoch = begin_runtime_epoch(db, instance_id='owner')
    exported = _exported_then_deleted(db, store, epoch, 's')
    result = store.maintenance.import_sessions([exported])
    assert result.ok is False and result.imported == 0, result
    assert any('s' in error and 'deleted' in error for error in result.errors), result
    assert get_session(db, 's') is None, 'the import put the deleted row back beside its fence'


def test_the_runtime_import_refuses_a_deleted_id_by_name(db, store):
    epoch = begin_runtime_epoch(db, instance_id='owner')
    exported = _exported_then_deleted(db, store, epoch, 's')
    with pytest.raises(AuthorityRefusal) as refused:
        mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id='s', request_id='restore',
                               expected_revision=0, operation='import', payload={'sessions': [exported]})
    assert refused.value.reason == 'admission_conflict'
    assert refused.value.field == 'sessions' and 'deleted' in (refused.value.detail or ''), vars(refused.value)
    assert get_session(db, 's') is None


def test_a_swept_session_restores_once_its_fence_ages_out(db, store):
    create_session(db, 'swept', source='cron')
    append_message(db, 'swept', role='user', content='keep me')
    exported = to_legacy_export_dict(store.maintenance.export_session('swept'))
    end_session(db, 'swept', 'test')
    assert store.maintenance.prune_sessions(older_than_days=None, media_root=MEDIA_ROOT).count == 1
    db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    assert store.maintenance.import_sessions([exported]).imported == 1
    assert get_session(db, 'swept') is not None
