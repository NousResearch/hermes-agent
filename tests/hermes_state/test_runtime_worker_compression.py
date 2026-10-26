"""Worker compression retains local guard and transcript contracts."""
import time

import pytest

from hermes_state_runtime import mutate_worker_execution


def test_guard_rollback_and_streaks_are_durable_receipts(worker):
    db, store = worker
    deadline = time.time() + 3600
    snapshot = store.get_compression_failure_cooldown_row('owned')
    store.record_compression_failure_cooldown('owned', deadline, 'first')
    store.record_compression_failure_cooldown('owned', deadline - 100, 'latest')
    assert db.get_compression_failure_cooldown_row('owned') == {
        'session_exists': True, 'cooldown_until': deadline, 'error': 'latest'}
    store.restore_compression_failure_cooldown_row('owned', snapshot)
    assert db.get_compression_failure_cooldown_row('owned') == snapshot
    store.record_compression_failure_cooldown('owned', deadline)
    store.clear_compression_failure_cooldown('owned')
    assert store.get_compression_failure_cooldown('owned') is None
    store.set_compression_fallback_streak('owned', 4)
    store.set_compression_ineffective_count('owned', 3)
    store.set_compression_recovery_deadline('owned', deadline)
    assert (db.get_compression_fallback_streak('owned'), db.get_compression_ineffective_count('owned'),
            db.get_compression_recovery_deadline('owned')) == (4, 3, deadline)
    seq = store.journal['next_sequence']
    with pytest.raises(Exception, match='cannot restore absent'):
        mutate_worker_execution(db, **store.scope, sequence=seq, operation='compression.cooldown.restore',
                                payload={'snapshot': {'session_exists': False, 'cooldown_until': None, 'error': None}})
    assert db._read_one('SELECT last_sequence FROM worker_executions')[0] == seq - 1


def test_compression_lease_cannot_be_revived_or_released_by_old_holder(worker):
    db, store = worker
    assert store.try_acquire_compression_lock('owned', 'old')
    assert not store.try_acquire_compression_lock('owned', 'new')
    db._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.refresh_compression_lock('owned', 'old')
    db._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.try_acquire_compression_lock('owned', 'new')
    assert not store.refresh_compression_lock('owned', 'old')
    store.release_compression_lock('owned', 'old')
    assert store.get_compression_lock_holder('owned') == 'new'
    seq = store.journal['next_sequence']
    for scope, error in [(dict(store.scope, epoch=store.scope['epoch'] - 1), 'stale_epoch'),
                         (dict(store.scope, session_id='foreign'), 'permission_denied')]:
        with pytest.raises(Exception, match=error):
            mutate_worker_execution(db, **scope, sequence=seq, operation='compression.lock.release', payload={'holder': 'new'})
    assert db.get_compression_lock_holder('owned') == 'new'
    store.release_compression_lock('owned', 'new')
    assert store.get_compression_lock_holder('owned') is None
