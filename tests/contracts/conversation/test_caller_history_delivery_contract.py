"""Contract cases for the caller-history delivery verbs: claim, reserve, commit and release of delegation rows."""

from __future__ import annotations

from persistence.conversation.messages import DelegationDelivery, NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.identity import SessionId

S = SessionId("caller-history-s1")


def _store(backend, scope):
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="api_server"))
    store.messages.append(S, NewMessage(role="user", content="go"))
    return store


def _deliver(store, delegation_id):
    return store.messages.append_delegation_delivery(S, DelegationDelivery(f"RESULT-{delegation_id}", {"delegation_id": delegation_id}))


def test_claim_consumes_each_delivery_once_and_release_returns_it(backend, scope_a):
    store = _store(backend, scope_a)
    row_id = _deliver(store, "a")
    claimed = store.messages.claim_caller_history_deliveries(S)
    # CONTRACT: only delegation rows are claimed, once; a release of the claim makes them pending again.
    assert [(r["id"], r["content"], r["display_metadata"]["delegation_id"]) for r in claimed] == [(row_id, "RESULT-a", "a")]
    assert store.messages.claim_caller_history_deliveries(S) == []
    assert store.messages.release_caller_history_deliveries(S, [row_id]) == 1
    assert [r["id"] for r in store.messages.claim_caller_history_deliveries(S)] == [row_id]


def test_a_reservation_holds_rows_until_its_owner_settles_it(backend, scope_a):
    store = _store(backend, scope_a)
    _deliver(store, "a")
    lease = store.messages.reserve_caller_history_deliveries(S, "webui", ttl_seconds=60)
    token = lease[0]["reservation_token"]
    # CONTRACT: a live lease hides the row from claims and other reservations; only its owner settles it.
    assert store.messages.claim_caller_history_deliveries(S) == []
    assert store.messages.reserve_caller_history_deliveries(S, "other", ttl_seconds=60) == []
    assert store.messages.commit_caller_history_deliveries(token, "intruder") == 0
    assert store.messages.release_caller_history_deliveries(reservation_token=token, owner="webui") == 1
    again = store.messages.reserve_caller_history_deliveries(S, "webui", ttl_seconds=60, limit=1)
    assert store.messages.commit_caller_history_deliveries([again[0]["id"]], "webui") == 1
    assert store.messages.reserve_caller_history_deliveries(S, "webui", ttl_seconds=60) == []


def test_an_expired_reservation_is_taken_over(backend, scope_a):
    store = _store(backend, scope_a)
    _deliver(store, "a")
    short = store.messages.reserve_caller_history_deliveries(S, "crashy", ttl_seconds=-1)
    retry = store.messages.reserve_caller_history_deliveries(S, "webui", ttl_seconds=60)
    # CONTRACT: a lapsed lease is pending again; its old token no longer settles the row.
    assert [r["id"] for r in retry] == [short[0]["id"]]
    assert store.messages.commit_caller_history_deliveries(short[0]["reservation_token"], "crashy") == 0
    assert store.messages.commit_caller_history_deliveries(retry[0]["reservation_token"], "webui") == 1
