"""Contract cases for the turn-lease fence of ``MessageRepository.rewrite_pruned_rows``: a writer inside a turn rewrites only while its turn lease still names it, as its appends do.

Every case seeds through public protocol operations (``SessionRepository.create``, ``append``, the coordination turn-lease verbs) and observes through public reads; refusals are proven non-mutating against re-read rows and the stored lease deadline.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from persistence import PruneRowUnresolved, TurnLeaseLost
from persistence.conversation.messages import HeldMessage, NewMessage, PrunedRowChange
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId
from tests.contracts._waiting import wait_past
from tests.contracts.conftest import require_capability

S = SessionId("prune-lease-s1")
CALL = {"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}


def msg(content, i: int, role: str = "user", **kwargs) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp(), **kwargs)


def _seed(backend, scope):
    """A tool turn; returns the store and the change that prunes its result to a stub."""
    require_capability(backend, "transactional_claims")
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    store.messages.append(S, msg("run it", 0))
    store.messages.append(S, msg("", 1, "assistant", tool_calls=(CALL,)))
    result = store.messages.append(S, msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))
    original = HeldMessage(role="tool", content="long output", tool_call_id="call-1", row_id=result, persisted=True)
    return store, PrunedRowChange(original, msg("[pruned]", 2, "tool", tool_call_id="call-1", tool_name="terminal"),
                                  msg("long output", 2, "tool", tool_call_id="call-1", tool_name="terminal"))


def _contents(store) -> list:
    return [r.content for r in store.messages.read_transcript(S)]


def test_the_holder_of_the_live_turn_lease_rewrites(backend, scope_a):
    """CONTRACT: a rewrite naming the holder of the live turn lease commits and leaves the lease with it."""
    store, change = _seed(backend, scope_a)
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="turn-1", ttl_seconds=300.0) is True
    done = store.messages.rewrite_pruned_rows(S, [change], turn_lease_holder="turn-1")
    assert done.rewritten == 1
    assert _contents(store) == ["run it", "", "[pruned]"]
    assert store.coordination.refresh_turn_lease(session_id=S, holder="turn-1", ttl_seconds=300.0) is True


@pytest.mark.parametrize("other_holder", [None, "turn-2"])
def test_a_rewrite_whose_turn_lease_is_lost_writes_nothing(backend, scope_a, other_holder):
    """CONTRACT: with no turn lease, or one naming another holder, the rewrite is TurnLeaseLost with the store's text and nothing changes."""
    store, change = _seed(backend, scope_a)
    if other_holder is not None:
        assert store.coordination.try_acquire_turn_lease(session_id=S, holder=other_holder, ttl_seconds=300.0) is True
    with pytest.raises(TurnLeaseLost) as lost:
        store.messages.rewrite_pruned_rows(S, [change], model_config_patch={"_prune_anchor": 3}, turn_lease_holder="turn-1")
    assert str(lost.value) == f"Session turn lease lost; refusing transcript write for {S!r}"
    assert _contents(store) == ["run it", "", "long output"]
    assert "_prune_anchor" not in (store.sessions.get(S).model_config or {})


def test_the_turn_lease_is_checked_before_the_row_refusals(backend, scope_a):
    """CONTRACT: a lost lease refuses before a change that cannot be named in place is looked at."""
    store, change = _seed(backend, scope_a)
    user = PrunedRowChange(HeldMessage(role="user", content="run it", persisted=True), msg("[pruned]", 0), msg("run it", 0))
    with pytest.raises(TurnLeaseLost):
        store.messages.rewrite_pruned_rows(S, [change, user], turn_lease_holder="turn-1")
    with pytest.raises(PruneRowUnresolved):
        store.messages.rewrite_pruned_rows(S, [change, user])


def test_an_expired_lease_naming_the_holder_is_renewed(backend, scope_a):
    """CONTRACT: an expired turn lease that still names the holder lets the rewrite commit and is renewed for the given time limit."""
    store, change = _seed(backend, scope_a)
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="turn-1", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    before = time.time()
    assert store.messages.rewrite_pruned_rows(S, [change], turn_lease_holder="turn-1", turn_lease_ttl_seconds=600.0).rewritten == 1
    expires_at = store.coordination.turn_lease_expires_at(session_id=S, holder="turn-1")
    assert expires_at is not None and before + 590.0 <= expires_at <= time.time() + 600.0


def test_a_refused_rewrite_renews_no_lease(backend, scope_a):
    """CONTRACT: an expired lease naming the holder stays expired when a later refusal stops the rewrite."""
    store, change = _seed(backend, scope_a)
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="turn-1", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    user = PrunedRowChange(HeldMessage(role="user", content="run it", persisted=True), msg("[pruned]", 0), msg("run it", 0))
    with pytest.raises(PruneRowUnresolved):
        store.messages.rewrite_pruned_rows(S, [change, user], turn_lease_holder="turn-1")
    expires_at = store.coordination.turn_lease_expires_at(session_id=S, holder="turn-1")
    assert expires_at is not None and expires_at <= time.time()
    assert _contents(store) == ["run it", "", "long output"]


@pytest.mark.parametrize("holder", [None, ""])
def test_no_holder_rewrites_unfenced(backend, scope_a, holder):
    """CONTRACT: with no holder, or an empty one, the rewrite ignores another holder's live turn lease and leaves it in place."""
    store, change = _seed(backend, scope_a)
    assert store.coordination.try_acquire_turn_lease(session_id=S, holder="turn-2", ttl_seconds=300.0) is True
    assert store.messages.rewrite_pruned_rows(S, [change], turn_lease_holder=holder).rewritten == 1
    assert _contents(store) == ["run it", "", "[pruned]"]
    assert store.coordination.refresh_turn_lease(session_id=S, holder="turn-2", ttl_seconds=300.0) is True
