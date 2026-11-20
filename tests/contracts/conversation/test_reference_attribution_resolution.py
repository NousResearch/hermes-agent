"""Main-loop attribution resolves missing fields from the updated session."""

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting


@pytest.mark.parametrize("provider", [None, ""])
@pytest.mark.parametrize("missing_fields", [False, True])
def test_attribution_uses_stored_provider_and_coalesces_keys(backend, scope_a, provider, missing_fields):
    store_a = backend.open(scope_a)
    sid = SessionId("route-resolution")
    store_a.sessions.create(NewSession(session_id=sid, source="cli", model="seed"))
    store_a.usage.set_billing_route(sid, provider="new", base_url="u", billing_mode="api")
    route = BillingRoute(model="z", billing_provider=provider, billing_base_url=provider if missing_fields else "u", billing_mode=provider if missing_fields else "api")
    expected = BillingRoute(model="z", billing_provider="new", billing_base_url="u", billing_mode="api")
    for incoming, calls in [(route, 1), (expected, 2), (route, 3)]:
        store_a.usage.queue_usage(sid, UsageDelta(route=incoming, api_call_count=1))
        recent = store_a.usage.recent_route(sid)
        assert recent.route == expected
        assert recent.api_call_count == calls
    assert (store_a.sessions.get(sid)).model == "seed"
    before = accounting(store_a, sid)
    store_a.usage.record_auxiliary_usage(sid, "vision", model="aux", billing_provider=provider, api_call_count=2, tokens=TokenCounts(input_tokens=3))
    assert accounting(store_a, sid) == before
    assert (store_a.usage.recent_route(sid)).route == expected
    auxiliary = store_a.usage.auxiliary_usage_by_task(sid)
    assert len(auxiliary) == 1
    assert auxiliary["vision"].api_call_count == 2
    assert auxiliary["vision"].tokens.input_tokens == 3
    if hasattr(backend, "session_db"):
        row = backend.session_db(scope_a)._read_one("SELECT model,billing_provider,billing_base_url,billing_mode FROM session_model_usage WHERE session_id=? AND task='vision'", (sid,))
        assert tuple(row) == ("aux", "", "", "")
    else:
        routes = backend.open(scope_a).usage._tables(scope_a)["routes"]
        auxiliary_routes = [value.route for (session, key), value in routes.items() if session == sid and key[0] == "vision"]
        assert len(auxiliary_routes) == 1
        assert auxiliary_routes[0].model == "aux"
        assert not auxiliary_routes[0].billing_provider
        assert not auxiliary_routes[0].billing_base_url
        assert not auxiliary_routes[0].billing_mode


def test_routeless_delta_inherits_only_for_main_loop(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = SessionId("routeless-resolution")
    store_a.sessions.create(NewSession(session_id=sid, source="cli", model="stored"))
    store_a.usage.set_billing_route(sid, provider="p", base_url="u", billing_mode="api")
    store_a.usage.queue_usage(sid, UsageDelta(api_call_count=1))
    recent = store_a.usage.recent_route(sid)
    assert recent.route == BillingRoute(model="stored", billing_provider="p", billing_base_url="u", billing_mode="api")
    assert recent.api_call_count == 1
    store_a.usage.record_auxiliary_usage(sid, "vision", api_call_count=2)
    assert store_a.usage.recent_route(sid) == recent
    assert (accounting(store_a, sid)).api_call_count == 1
