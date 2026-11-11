"""Housekeeping on a SQLite backend store keeps the logical owner of a retained local reset child.

A local reset keeps policy/FIFO/generation on the creation id and advances the receipt to a
physical child. Retention and empty-session sweeps age the ended root on its own; deleting it
would leave the child's history behind with no owner to admit into or restore from.
"""
import time
import uuid

import pytest

from persistence.sqlite.authority._admission_store import admit_session_input
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, delete_empty_sessions, get_session
from tests.contracts._scopes import MEDIA_ROOT
from tests.persistence_sqlite.housekeeping._local_session import _local_session, _mutate


def _store_reset(db, epoch, sid):
    from gateway.session import SessionEntry
    from persistence.sqlite.authority._local import local_receipt, reset_local_target
    old = SessionEntry.from_dict(local_receipt(db, sid)['entry'])
    child = uuid.uuid4().hex
    entry = SessionEntry(old.session_key, child, old.created_at, old.updated_at, origin=old.origin,
                         platform=old.platform, chat_type=old.chat_type, is_fresh_reset=True)
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=entry.to_dict())
    return child


@pytest.mark.parametrize('sweep', ['retention', 'empty'])
@pytest.mark.parametrize('producer', ['canonical', 'session_store'])
def test_sweeps_keep_the_owner_a_retained_reset_child_needs(backend, scope_a, tmp_path, producer, sweep):
    from persistence.sqlite.authority._local import local_physical_target
    store = backend.open(scope_a)
    db = store.holder()
    epoch = begin_runtime_epoch(db, instance_id='owner')
    sid = _local_session(db, epoch, tmp_path)
    child = (_mutate(db, epoch, sid, 'reset')['target_session_id'] if producer == 'canonical'
             else _store_reset(db, epoch, sid))
    append_message(db, child, 'user', 'hi')
    append_message(db, child, 'assistant', 'hello')
    aged = time.time() - 200 * 86400
    db._execute_write(lambda c: c.execute(
        'UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?', (aged, aged, sid)))
    # An unrelated aged, empty, ended session is still swept: the filter is ownership, not a blanket skip.
    create_session(db, 'stray', source='cli')
    db._execute_write(lambda c: c.execute(
        'UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=? WHERE id=?', (aged, aged, aged, 'stray')))
    swept = (store.maintenance.prune_sessions(older_than_days=90, media_root=MEDIA_ROOT).count if sweep == 'retention'
             else len(delete_empty_sessions(db, media_root=MEDIA_ROOT)))
    assert swept == 1 and get_session(db, 'stray') is None
    assert get_session(db, sid) is not None
    assert [m['content'] for m in get_messages(db, child)] == ['hi', 'hello']
    with db._read_ctx() as conn:
        assert local_physical_target(conn, sid) == child
    admitted, _ = admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                      request_id='after', payload={'text': 'after'})
    assert admitted['status'] == 'queued'
