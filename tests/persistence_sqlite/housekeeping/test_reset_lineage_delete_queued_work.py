"""On a SQLite backend store a local reset conversation with accepted queued work survives every delete and sweep
of its current physical segment (JoaoMarcos44 R1, andrexibiza 8, dokterdok corr-reset-delete).

Canonical reset keeps the creation id S0 as the owner (policy, FIFO, generation) and moves the
transcript to child S1; a follower queued on S0 runs against S1. Deleting or sweeping S1 alone
used to retire S0's policy and leave the queued admission with nothing to run on."""
import time

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.authority._admission_store import admit_session_input, get_session_admission
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    delete_empty_sessions, delete_session, delete_sessions, get_session,
)
from tests.contracts._scopes import MEDIA_ROOT
from tests.persistence_sqlite.housekeeping._local_session import _local_session, _mutate

OPERATIONS = {
    'single': lambda db, store, s0, s1: delete_session(db, s1, exclude_active_write_guards=True, media_root=MEDIA_ROOT),
    'bulk': lambda db, store, s0, s1: delete_sessions(db, [s1], exclude_active_write_guards=True, media_root=MEDIA_ROOT),
    'never-active': lambda db, store, s0, s1: store.routing.prune_never_active(older_than_days=30, media_root=MEDIA_ROOT),
    'prune-child': lambda db, store, s0, s1: store.maintenance.prune_sessions(
        older_than_days=30, criteria=PruneCriteria(started_before=time.time() - 30 * 86400), media_root=MEDIA_ROOT),
    'prune-all': lambda db, store, s0, s1: store.maintenance.prune_sessions(
        older_than_days=None, criteria=PruneCriteria(last_active_before=time.time() + 60), media_root=MEDIA_ROOT),
    # The empty-session sweep shares retire_prunable: an ended, still-empty reset tip is not an empty chat.
    'prune-empty': lambda db, store, s0, s1: delete_empty_sessions(db, media_root=MEDIA_ROOT),
}


def _conversation(store, tmp_path, operation, queued):
    db = store.holder()
    epoch = begin_runtime_epoch(db, instance_id='owner')
    s0 = _local_session(db, epoch, tmp_path)
    append_message(db, s0, 'user', 'root history')
    s1 = _mutate(db, epoch, s0, 'reset')['target_session_id']
    old = time.time() - 200 * 86400
    # Aged past every cutoff; the current segment ended for the ended-only prune, open for never-active.
    db._execute_write(lambda c: c.execute('UPDATE sessions SET started_at=? WHERE id IN (?,?)', (old, s0, s1)))
    if operation.startswith('prune'):
        db._execute_write(lambda c: c.execute('UPDATE sessions SET ended_at=? WHERE id=?', (old, s1)))
    admission = queued and admit_session_input(db, epoch=epoch, principal_id='human', session_id=s0,
                                               request_id='queued', payload={'text': 'continue'})[0]
    return db, s0, s1, admission


def _state(db, s0, s1):
    with db._read_ctx() as conn:
        policy = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (POLICY_PREFIX + s0,)).fetchone() is not None
    return {'s0': get_session(db, s0) is not None, 's1': get_session(db, s1) is not None, 'policy': policy}


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.mark.parametrize('operation', sorted(OPERATIONS))
def test_segment_delete_or_sweep_keeps_a_conversation_with_queued_work(store, tmp_path, operation):
    db, s0, s1, admission = _conversation(store, tmp_path, operation, queued=True)
    if operation == 'never-active':
        assert s1 in [r.session_id for r in store.routing.never_active_keyed_sessions(older_than_days=30)]
    try:
        OPERATIONS[operation](db, store, s0, s1)
    except AuthorityRefusal as exc:
        assert operation in {'single', 'bulk'} and exc.reason == 'session_busy', (operation, exc.reason)
    assert _state(db, s0, s1) == {'s0': True, 's1': True, 'policy': True}, operation
    assert get_session_admission(db, admission_id=admission['admission_id'])['status'] == 'queued'


@pytest.mark.parametrize('operation', ['single', 'bulk', 'prune-all'])
def test_idle_conversation_deletes_whole_from_its_current_segment(store, tmp_path, operation):
    db, s0, s1, _ = _conversation(store, tmp_path, operation, queued=False)
    if operation == 'never-active':
        assert s1 in [r.session_id for r in store.routing.never_active_keyed_sessions(older_than_days=30)]
    OPERATIONS[operation](db, store, s0, s1)
    assert _state(db, s0, s1) == {'s0': False, 's1': False, 'policy': False}, operation
