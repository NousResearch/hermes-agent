"""Logical turn cleanup remains holder-fenced after physical rotation."""
from types import SimpleNamespace

import pytest

from agent.turn_facade_lease import DurableTurnLease
from hermes_state_runtime import mutate_worker_execution
from tests.state.test_runtime_worker_compression import worker  # noqa: F401


def test_admitted_parent_lease_releases_after_rotation_without_stealing_successor(worker):
    db, store = worker
    agent = SimpleNamespace(session_id='owned', _active_session_turn_lease_holder='original')
    assert store.try_acquire_session_turn_lease('owned', 'original')
    lease = DurableTurnLease(agent, store, 'owned', 'original')
    assert store.try_acquire_compression_lock('owned', 'compressor')
    store.publish_compression_child(parent_session_id='owned', child_session_id='child', source='cli',
        messages=[{'role': 'assistant', 'content': 'summary'}], compression_lock_holder='compressor')
    agent.session_id = 'child'
    lease.release()
    assert db._read_one('SELECT COUNT(*) FROM session_turn_leases')[0] == 0
    assert store.try_acquire_session_turn_lease('child', 'successor')
    lease.release()
    assert db._read_one('SELECT holder FROM session_turn_leases')[0] == 'successor'
    with pytest.raises(Exception, match='permission_denied'):
        mutate_worker_execution(db, **store.scope, sequence=store.journal['next_sequence'],
            operation='turn.cleanup', payload={'target': 'foreign', 'holder': 'successor'})
    assert db._read_one('SELECT holder FROM session_turn_leases')[0] == 'successor'
    store.release_session_turn_lease('child', 'successor')
