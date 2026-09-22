"""Worker compression retains local guard and transcript contracts."""
import time

import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution


@pytest.fixture
def worker(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli', system_prompt='prefix')
    db.create_session('foreign', 'cli', system_prompt='secret')
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
    store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
    yield db, store
    store.failure = None
    store.journal['pending'] = []
    store.close()
    db.close()


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
