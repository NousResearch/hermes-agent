"""Worker usage attribution follows the main-loop and auxiliary gates."""

import pytest

from persistence.conversation.usage import BillingRoute, RouteUsage, TokenCounts, UsageDelta
from tests.contracts.conversation.test_usage import accounting
from tests.contracts.workers._support import SESSION, ident, worker


@pytest.mark.parametrize("usage,expected_model", [
    ({"model": "m", "billing_provider": "p", "estimated_cost_usd": 0.0}, None),
    ({"model": "m", "billing_provider": "p", "estimated_cost_usd": 0.25}, "m"),
    ({"input_tokens": 3, "api_call_count": 1}, "seed-model"),
    ({"input_tokens": 3, "model": "", "billing_provider": ""}, "seed-model"),
])
def test_main_usage_gate_and_route_inheritance(backend, scope_a, usage, expected_model):
    store, epoch = worker(backend, scope_a)
    store.sessions.update_model(SESSION, "seed-model")
    store.usage.set_billing_route(SESSION, provider="seed-provider", base_url="seed-url")
    store.worker_executions.record_usage(**ident(epoch), **usage)
    store.worker_executions.record_usage(**ident(epoch), **usage)
    route = store.usage.recent_route(SESSION)
    assert (route.route.model if route else None) == expected_model
    if route is not None:
        assert route.api_call_count == usage.get("api_call_count", 0)
    aggregate = accounting(store, SESSION)
    assert aggregate.tokens.input_tokens == usage.get("input_tokens", 0)
    assert aggregate.estimated_cost_usd == (usage.get("estimated_cost_usd") or 0.0)


@pytest.mark.parametrize("usage", [{}, {"api_call_count": 0}, {"model": "aux", "estimated_cost_usd": 0.0}, {"input_tokens": 5, "api_call_count": 2, "estimated_cost_usd": 0.5}])
def test_auxiliary_usage_is_unconditional_and_receipted(backend, scope_a, usage):
    store, epoch = worker(backend, scope_a)
    before = accounting(store, SESSION)
    store.worker_executions.record_auxiliary_usage(**ident(epoch), task="vision", **usage)
    store.worker_executions.record_auxiliary_usage(**ident(epoch), task="vision", **usage)
    rows = store.usage.auxiliary_usage_by_task(SESSION)
    assert list(rows) == ["vision"]
    # An auxiliary report without a call count counts one call.
    assert rows["vision"].api_call_count == usage.get("api_call_count", 1)
    assert rows["vision"].tokens == TokenCounts(input_tokens=usage.get("input_tokens", 0))
    assert rows["vision"].estimated_cost_usd == (usage.get("estimated_cost_usd") or 0.0)
    assert accounting(store, SESSION) == before


@pytest.mark.parametrize("empty_first", [False, True], ids=["absent-then-empty", "empty-then-absent"])
def test_empty_and_absent_billing_fields_share_worker_route(backend, scope_a, empty_first):
    store, epoch = worker(backend, scope_a)
    store.sessions.update_model(SESSION, "m")
    store.usage.set_billing_route(SESSION, provider="p", base_url=None)
    usages = [{"model": "m", "billing_provider": "p", "api_call_count": 1},
              {"model": "m", "billing_provider": "p", "billing_base_url": "", "billing_mode": "", "api_call_count": 1}]
    if empty_first:
        usages.reverse()
    for sequence, usage in enumerate(usages, start=1):
        store.worker_executions.record_usage(**ident(epoch, sequence=sequence), **usage)
    assert store.usage.recent_route(SESSION) == RouteUsage(BillingRoute(model="m", billing_provider="p"), api_call_count=2)
    assert accounting(store, SESSION).api_call_count == 2


def test_empty_to_none_stored_url_keeps_one_worker_route(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.sessions.update_model(SESSION, "m")
    store.usage.set_billing_route(SESSION, provider="p", base_url="", billing_mode="")
    usage = {"model": "m", "billing_provider": "p", "api_call_count": 1}
    store.worker_executions.record_usage(**ident(epoch), **usage)
    store.usage.set_billing_route(SESSION, provider="p", base_url=None)
    store.worker_executions.record_usage(**ident(epoch, sequence=2), **usage)
    assert store.usage.recent_route(SESSION) == RouteUsage(BillingRoute(model="m", billing_provider="p"), api_call_count=2)
    assert accounting(store, SESSION).api_call_count == 2
    stored = store.sessions.get(SESSION)
    assert (stored.billing_base_url, stored.billing_mode) == (None, "")


def test_interleaved_direct_and_worker_usage_share_empty_billing_route(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.sessions.update_model(SESSION, "m")
    store.usage.set_billing_route(SESSION, provider="p", base_url="", billing_mode="")
    usage = {"model": "m", "billing_provider": "p", "api_call_count": 1}
    store.worker_executions.record_usage(**ident(epoch), **usage)
    store.usage.queue_usage(SESSION, UsageDelta(api_call_count=1, route=BillingRoute(model="m", billing_provider="p")))
    store.worker_executions.record_usage(**ident(epoch, sequence=2), **usage)
    assert store.usage.recent_route(SESSION) == RouteUsage(BillingRoute(model="m", billing_provider="p"), api_call_count=3)
    assert accounting(store, SESSION).api_call_count == 3
    stored = store.sessions.get(SESSION)
    assert (stored.billing_base_url, stored.billing_mode) == ("", "")
