"""The durable turn lease an append names, inside a claimed turn whose fence still holds: the claimed-turn fence adds
to the lease check and replaces none of it. An append naming a holder whose lease was replaced is refused with
``TurnLeaseLost`` (not the fence's ``TurnFenceSuperseded``), lands nothing and leaves the replacement's lease as it was;
an append by a holder whose lease expired while it still owns it lands and renews the lease by the TTL it named; an
append naming no holder ignores another holder's live lease and neither renews nor releases it. Requires
authority_fences (``require_capability`` in the claim helper).
"""

from __future__ import annotations

import time

import pytest

from persistence import TurnFenceSuperseded, TurnLeaseLost
from tests.contracts._waiting import wait_past
from tests.contracts.hosted.test_turn_fence_contract import SESSION, _claimed, _contents, _msg, _turn


def _expires(store, holder: str):
    return store.coordination.turn_lease_expires_at(session_id=SESSION, holder=holder)


def test_an_append_naming_a_replaced_holder_is_refused_and_writes_nothing(backend_name: str, backend, scope_a) -> None:
    """The holder released its lease and another acquired it while the turn still ran: the stale holder's append is
    TurnLeaseLost, lands nothing and renews, moves or drops nothing of the replacement's lease."""
    store, _epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="stale-holder", ttl_seconds=300.0)
        store.coordination.release_turn_lease(session_id=SESSION, holder="stale-holder")
        assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="replacement", ttl_seconds=300.0)
        before = _expires(store, "replacement")
        with pytest.raises(TurnLeaseLost) as lost:
            store.messages.append(SESSION, _msg("stale"), turn_lease_holder="stale-holder", turn_lease_ttl_seconds=30.0)
    assert not isinstance(lost.value, TurnFenceSuperseded)
    assert _contents(store) == []
    assert _expires(store, "replacement") == before
    assert _expires(store, "stale-holder") is None


def test_an_append_by_the_holder_renews_its_expired_lease(backend_name: str, backend, scope_a) -> None:
    """A starved refresher: the holder's lease expired but still names it. Its append under the current fence lands
    and moves the lease's expiry forward by the TTL the append named."""
    store, _epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="holder", ttl_seconds=0.1)
        wait_past(time.time() + 0.1)
        assert _expires(store, "holder") <= time.time()
        before = time.time()
        store.messages.append(SESSION, _msg("renewed"), turn_lease_holder="holder", turn_lease_ttl_seconds=600.0)
    assert _contents(store) == ["renewed"]
    renewed = _expires(store, "holder")
    assert renewed is not None and before + 590.0 <= renewed <= time.time() + 600.0


def test_an_append_naming_no_holder_ignores_a_live_lease(backend_name: str, backend, scope_a) -> None:
    """An append naming no holder lands under the current fence while another holder's lease is live, and leaves that
    lease exactly as it was."""
    store, _epoch, claimed = _claimed(backend, scope_a)
    with _turn(scope_a, claimed):
        assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="another", ttl_seconds=300.0)
        before = _expires(store, "another")
        store.messages.append(SESSION, _msg("no holder"))
    assert _contents(store) == ["no holder"]
    assert _expires(store, "another") == before
