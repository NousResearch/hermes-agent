"""Recent routes exclude unusable identities without dropping accounting."""

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting


@pytest.mark.parametrize("model,provider,eligible", [("neighbor", None, False), ("neighbor", "", False), ("unknown", "p", False), ("neighbor", "p", True)])
@pytest.mark.parametrize("earlier", [False, True])
def test_cost_only_recent_route_eligibility(backend, scope_a, model, provider, eligible, earlier):
    store_a = backend.open(scope_a)
    sid = SessionId("route-eligibility")
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    previous = BillingRoute(model="earlier", billing_provider="old", billing_base_url="old-url", billing_mode="api")
    if earlier:
        store_a.usage.queue_usage(sid, UsageDelta(route=previous, estimated_cost_usd=0.25))
        store_a.usage.set_billing_route(sid, provider=None, base_url=None, billing_mode="")
    route = BillingRoute(model=model, billing_provider=provider, billing_base_url="url", billing_mode="api")
    store_a.usage.queue_usage(sid, UsageDelta(route=route, estimated_cost_usd=0.75))
    recent = store_a.usage.recent_route(sid)
    if eligible or earlier:
        assert recent.route == (route if eligible else previous)
        assert recent.api_call_count == 0
    else:
        assert recent is None
    usage = accounting(store_a, sid)
    assert usage.tokens == TokenCounts()
    assert usage.api_call_count == 0
    assert usage.estimated_cost_usd == (1.0 if earlier else 0.75)
    assert usage.actual_cost_usd is None
    if hasattr(backend, "session_db"):
        rows = backend.session_db(scope_a)._read_all("SELECT estimated_cost_usd FROM session_model_usage WHERE session_id=? AND task=''", (sid,))
        assert len(rows) == (2 if earlier else 1)
        assert sum(row[0] for row in rows) == usage.estimated_cost_usd
    else:
        tables = backend.open(scope_a).usage._tables(scope_a)
        rows = [value for (session, key), value in tables["route_usage"].items() if session == sid and key[0] is None]
        assert len(rows) == (2 if earlier else 1)
        assert sum(cost for _, cost in rows) == usage.estimated_cost_usd
