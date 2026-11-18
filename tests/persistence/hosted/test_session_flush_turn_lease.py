"""A claimed turn's transcript flush keeps the durable turn lease guard.

The turn's rows go through the agent's store handle (``agent.session_persistence._db_flush_write``), which carries the
agent's durable turn lease holder and TTL into the store's own lease-checked write, unchanged by the claimed-turn
fence: a lease replaced while the claimed turn still runs refuses the stale transcript (``TurnLeaseLost``, not the
fence's refusal), and an expired lease the holder still owns is renewed by the write itself, for the agent's TTL.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from persistence import TurnFenceSuperseded, TurnLeaseLost
from tests.contracts._waiting import wait_past
from tests.persistence.hosted._turn_rows import flush
from tests.persistence.hosted.test_turn_fence_lifetime import S, _contents, _turn


def _agent(store, holder, ttl=None):
    return SimpleNamespace(persistence=store, _session_db=store, session_id=S, _active_session_turn_lease_holder=holder,
                           _active_session_turn_lease_ttl_seconds=ttl)


def test_a_replaced_durable_lease_refuses_the_claimed_turns_flush(claimed):
    store, _epoch, row = claimed
    lease = store.coordination
    assert lease.try_acquire_turn_lease(session_id=S, holder="old", ttl_seconds=300.0)
    lease.release_turn_lease(session_id=S, holder="old")
    assert lease.try_acquire_turn_lease(session_id=S, holder="replacement", ttl_seconds=300.0)
    before = lease.turn_lease_expires_at(session_id=S, holder="replacement")
    with _turn(store, row), pytest.raises(TurnLeaseLost) as lost:
        flush(_agent(store, "old"), "stale durable holder")
    assert not isinstance(lost.value, TurnFenceSuperseded)
    assert _contents(store) == []
    assert lease.turn_lease_expires_at(session_id=S, holder="replacement") == before


def test_an_expired_durable_lease_is_renewed_by_the_claimed_turns_flush(claimed):
    store, _epoch, row = claimed
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="mine", ttl_seconds=0.1)
    wait_past(time.time() + 0.1)  # a starved refresher: the lease expired, the holder still owns it
    with _turn(store, row):
        flush(_agent(store, "mine", ttl=299.0), "renewed holder")
    assert _contents(store) == ["renewed holder"]
    after = store.coordination.turn_lease_expires_at(session_id=S, holder="mine")
    assert after is not None and after - time.time() > 200, "the flush must renew the expired lease its holder still owns"
