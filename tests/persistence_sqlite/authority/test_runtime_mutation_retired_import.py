"""A retired physical ID cannot be assigned to a different imported transcript."""
import pytest
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def test_import_cannot_reuse_a_deleted_receipt_identity(store):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    create_session(holder, 's', 'cli')
    deleted, _, _ = mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s',
        request_id='delete', expected_revision=0, expected_generation=0, operation='delete', payload={})
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='new',
            request_id='import', expected_revision=0, operation='import', payload={'sessions': [
                {'id': 'new', 'source': 'cli'}, {'id': 's', 'source': 'cli'}]})
    assert get_session(holder, 's') is None and get_session(holder, 'new') is None
    assert mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s',
        request_id='delete', expected_revision=0, expected_generation=0, operation='delete', payload={})[0] == deleted
