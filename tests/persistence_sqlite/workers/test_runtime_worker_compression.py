"""Worker compression retains local guard and transcript contracts."""
import time

import pytest

from persistence.sqlite.conversation._store.compression import (
    get_compression_failure_cooldown_row, get_compression_fallback_streak, get_compression_ineffective_count,
    get_compression_recovery_deadline,
)
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def test_guard_rollback_and_streaks_are_durable_receipts(worker):
    persistence, store = worker
    holder = persistence.holder()
    deadline = time.time() + 3600
    snapshot = store.get_compression_failure_cooldown_row('owned')
    store.record_compression_failure_cooldown('owned', deadline, 'first')
    store.record_compression_failure_cooldown('owned', deadline - 100, 'latest')
    assert get_compression_failure_cooldown_row(holder, 'owned') == {
        'session_exists': True, 'cooldown_until': deadline, 'error': 'latest'}
    store.restore_compression_failure_cooldown_row('owned', snapshot)
    assert get_compression_failure_cooldown_row(holder, 'owned') == snapshot
    store.record_compression_failure_cooldown('owned', deadline)
    store.clear_compression_failure_cooldown('owned')
    assert store.get_compression_failure_cooldown('owned') is None
    store.set_compression_fallback_streak('owned', 4)
    store.set_compression_ineffective_count('owned', 3)
    store.set_compression_recovery_deadline('owned', deadline)
    assert (get_compression_fallback_streak(holder, 'owned'), get_compression_ineffective_count(holder, 'owned'),
            get_compression_recovery_deadline(holder, 'owned')) == (4, 3, deadline)
    seq = store.journal['next_sequence']
    with pytest.raises(Exception, match='cannot restore absent'):
        mutate_worker_execution(holder, **store.scope, sequence=seq, operation='compression.cooldown.restore',
                                payload={'snapshot': {'session_exists': False, 'cooldown_until': None, 'error': None}})
    assert holder._read_one('SELECT last_sequence FROM worker_executions')[0] == seq - 1


def test_compression_lease_cannot_be_revived_or_released_by_old_holder(worker):
    persistence, store = worker
    holder = persistence.holder()
    assert store.try_acquire_compression_lock('owned', 'old')
    assert not store.try_acquire_compression_lock('owned', 'new')
    holder._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.refresh_compression_lock('owned', 'old')
    holder._write_sql('UPDATE compression_locks SET expires_at=0 WHERE session_id=?', ('owned',))
    assert store.try_acquire_compression_lock('owned', 'new')
    assert not store.refresh_compression_lock('owned', 'old')
    store.release_compression_lock('owned', 'old')
    assert store.get_compression_lock_holder('owned') == 'new'
    assert persistence.coordination.get_compression_lock_holder(session_id='owned') == 'new'
    store.release_compression_lock('owned', 'new')
    assert store.get_compression_lock_holder('owned') is None
