"""Housekeeping keeps the logical owner of a retained local reset child.

A local reset keeps policy/FIFO/generation on the creation id and advances the receipt to a
physical child. Retention and empty-session sweeps age the ended root on its own; deleting it
would leave the child's history behind with no owner to admit into or restore from.
"""
import time
import uuid
from contextlib import closing

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.hermes_state.test_target_advance_fence import _local_session, _mutate


def _store_reset(db, epoch, sid):
    from gateway.session import SessionEntry
    from hermes_state_local import local_receipt
    from hermes_state_local_lineage import reset_local_target
    old = SessionEntry.from_dict(local_receipt(db, sid)['entry'])
    child = uuid.uuid4().hex
    entry = SessionEntry(old.session_key, child, old.created_at, old.updated_at, origin=old.origin,
                         platform=old.platform, chat_type=old.chat_type, is_fresh_reset=True)
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=entry.to_dict())
    return child


@pytest.mark.parametrize('sweep', ['retention', 'empty'])
@pytest.mark.parametrize('producer', ['canonical', 'session_store'])
def test_sweeps_keep_the_owner_a_retained_reset_child_needs(tmp_path, producer, sweep):
    from hermes_state_local_lineage import local_physical_target
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid = _local_session(db, epoch, tmp_path)
        child = (_mutate(db, epoch, sid, 'reset')['target_session_id'] if producer == 'canonical'
                 else _store_reset(db, epoch, sid))
        db.append_message(child, 'user', 'hi')
        db.append_message(child, 'assistant', 'hello')
        aged = time.time() - 200 * 86400
        db._execute_write(lambda c: c.execute(
            'UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?', (aged, aged, sid)))
        # An unrelated aged, empty, ended session is still swept: the filter is ownership, not a blanket skip.
        db.create_session('stray', source='cli')
        db._execute_write(lambda c: c.execute(
            'UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=? WHERE id=?', (aged, aged, aged, 'stray')))
        swept = db.prune_sessions(older_than_days=90) if sweep == 'retention' else db.delete_empty_sessions()
        assert swept == 1 and db.get_session('stray') is None
        assert db.get_session(sid) is not None
        assert [m['content'] for m in db.get_messages(child)] == ['hi', 'hello']
        with db._read_ctx() as conn:
            assert local_physical_target(conn, sid) == child
        admitted = rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                                          request_id='after', payload={'text': 'after'})
        assert admitted['status'] == 'queued'
