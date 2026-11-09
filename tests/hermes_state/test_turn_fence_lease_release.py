"""A turn that lost its claim still releases its own durable turn lease.

The release is a holder-qualified delete of the turn's own ``session_turn_leases`` row: it writes no
content, so the claim's fence must not refuse it. Refused, the row outlives the turn until its TTL and
the session's next turn waits behind a holder that already finished.
"""
from contextlib import closing

import pytest

from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence
from hermes_state import SessionDB
from tests.hermes_state.test_turn_fence_writes import _claimed, _elsewhere, _move_generation, _take_over


@pytest.fixture
def store(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        yield db


@pytest.mark.parametrize('supersede', [_move_generation, _take_over], ids=['generation', 'owner_epoch'])
def test_a_superseded_turn_still_releases_its_own_turn_lease(store, tmp_path, supersede):
    sid, row = _claimed(store, tmp_path)
    with bind_profile_scope(tmp_path), bind_turn(sid, row['admission_id'], claim_fence(row), store=store.db_path):
        assert store.try_acquire_session_turn_lease(sid, 'finished-turn', ttl_seconds=300)
        supersede(store, sid)
        store.release_session_turn_lease(sid, 'finished-turn')
    assert _elsewhere(lambda: store.try_acquire_session_turn_lease(sid, 'next-turn', ttl_seconds=300))
