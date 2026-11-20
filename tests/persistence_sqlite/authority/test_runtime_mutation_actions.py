"""Action-specific mutations share the canonical receipt transaction."""
import sqlite3
import pytest
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def test_sidebar_edit_is_atomic_and_revision_fenced(store):
    holder = store.holder()
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='edit',
                expected_revision=0, operation='sidebar',
                payload={'title': 'Title', 'pinned': True, 'hidden': True, 'unread': True})
    holder._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
        mutate_runtime_session(holder, **args)
    assert get_session(holder, 's')['title'] is None
    assert not get_session(holder, 's')['pinned']
    holder._execute_write(lambda c: c.execute('DROP TRIGGER fail_receipt'))
    receipt, _, _ = mutate_runtime_session(holder, **args)
    assert receipt['revision'] == 1
    assert receipt['pinned'] and receipt['hidden'] and receipt['unread']
    peer = StoreHolder(store.store_path)
    try:
        with pytest.raises(AuthorityRefusal, match='revision_conflict'):
            mutate_runtime_session(peer, **(args | {'principal_id': 'peer'}))
        assert mutate_runtime_session(peer, **args)[0] == receipt
    finally:
        peer.close()
    assert get_session(holder, 's')['runtime_revision'] == 1
