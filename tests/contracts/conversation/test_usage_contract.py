"""Contract suite for the ``UsageRepository`` capability.

Backend cases run the real fixture backend end to end: session rows come from the fixture's seeding seam, accounting is written through ``queue_usage`` and observed through the usage reads and the session row (``SessionRepository.get``). The record-validation case constructs records without calling the backend. Assertions cover inclusion AND exclusion; the concurrency test collects every outcome and asserts the exact-sum partition.

The contract offers no absolute write. The QUIRKs (COALESCE fill, first-accounted-route replacement, ghost-session upsert, cost-only attribution gate) are documented in the persistence/conversation/usage.py module docstring.
"""

from __future__ import annotations


import pytest

from persistence.conversation.usage import (
    BillingRoute,
    TokenCounts,
    UsageDelta,
    UsageTotalsQuery,
)
from tests.contracts.conversation.test_usage import accounting
from tests.contracts._scopes import profile_scope
from tests.contracts._concurrency import run_concurrently

STORM_ACTORS = 8
STORM_ROUNDS = 20

ROUTE_A = BillingRoute(model="model-a", billing_provider="prov-a", billing_base_url="https://prov-a.example", billing_mode="plan")


def full_delta() -> UsageDelta:
    return UsageDelta(
        tokens=TokenCounts(input_tokens=10, output_tokens=5, cache_read_tokens=2, cache_write_tokens=1, reasoning_tokens=3),
        api_call_count=1,
        estimated_cost_usd=0.5,
        actual_cost_usd=0.4,
        cost_status="final",
        cost_source="api",
        pricing_version="pv-2026-01",
        route=ROUTE_A,
    )


# ── delta writes: aggregate and attribution ───────────────────────────────────


def test_one_delta_lands_every_field_on_aggregate_and_attribution(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-1")
    # Pre-state: a seeded session reads back as the zero aggregate with no attribution.
    before = accounting(store_a, "s-usage-1")
    assert before.tokens == TokenCounts() and before.api_call_count == 0
    assert store_a.usage.recent_route("s-usage-1") is None
    result = store_a.usage.queue_usage("s-usage-1", full_delta())
    # CONTRACT: the delta write is a None-returning command (typed outcome).
    assert result is None
    usage = accounting(store_a, "s-usage-1")
    # CONTRACT: one delta call lands every counter and cost field on the session aggregate.
    assert usage.tokens == TokenCounts(input_tokens=10, output_tokens=5, cache_read_tokens=2, cache_write_tokens=1, reasoning_tokens=3)
    assert usage.api_call_count == 1
    assert usage.estimated_cost_usd == pytest.approx(0.5)
    assert usage.actual_cost_usd == pytest.approx(0.4)
    assert (usage.cost_status, usage.cost_source, usage.pricing_version) == ("final", "api", "pv-2026-01")
    # CONTRACT: the same call records the attribution row, readable through the dominant-route
    # surface with the full billing tuple.
    dominant = store_a.usage.recent_route("s-usage-1")
    assert dominant.route == ROUTE_A
    assert dominant.api_call_count == 1


def test_delta_mode_accumulates(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-2")
    delta = UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=4), api_call_count=1,
                       estimated_cost_usd=0.25, route=ROUTE_A)
    for _ in range(3):
        store_a.usage.queue_usage("s-usage-2", delta)
    usage = accounting(store_a, "s-usage-2")
    # CONTRACT: delta mode adds; three identical calls triple the counters, calls, and estimate.
    assert usage.tokens.input_tokens == 30
    assert usage.tokens.output_tokens == 12
    assert usage.api_call_count == 3
    assert usage.estimated_cost_usd == pytest.approx(0.75)
    # CONTRACT: the attribution row accumulates the per-call counts for that route.
    assert (store_a.usage.recent_route("s-usage-2")).api_call_count == 3


def test_cost_only_delta_lands_on_aggregate(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-cc")
    store_a.usage.queue_usage(
        "s-usage-cc", UsageDelta(actual_cost_usd=0.9, route=ROUTE_A))
    usage = accounting(store_a, "s-usage-cc")
    # CONTRACT: a billed-cost-only delta lands on the session aggregate and adds no counters.
    assert usage.actual_cost_usd == pytest.approx(0.9)
    assert usage.tokens == TokenCounts() and usage.api_call_count == 0


# ── attribution keying and the dominant route ─────────────────────────────────


def test_recent_route_is_later_recorded_not_most_calls(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-4")
    # Discriminating seed for recency rather than frequency: the busier route lands FIRST, the
    # later-recorded route has fewer calls and fewer tokens.
    store_a.usage.queue_usage(
        "s-usage-4",
        UsageDelta(tokens=TokenCounts(input_tokens=10), api_call_count=5,
                   route=BillingRoute(model="model-a", billing_provider="prov-a")))
    store_a.usage.queue_usage(
        "s-usage-4",
        UsageDelta(tokens=TokenCounts(input_tokens=1), api_call_count=2,
                   route=BillingRoute(model="model-b", billing_provider="prov-b")))
    # CONTRACT: the recent route is the most recently recorded main-loop
    # row, not the busiest by calls or the largest by tokens (a route retired
    # long ago must not keep reading back as current).
    assert (store_a.usage.recent_route("s-usage-4")).route.model == "model-b"
    # A later delta on the older route makes it the recent one again.
    store_a.usage.queue_usage(
        "s-usage-4",
        UsageDelta(tokens=TokenCounts(input_tokens=1), api_call_count=1,
                   route=BillingRoute(model="model-a", billing_provider="prov-a")))
    recent = store_a.usage.recent_route("s-usage-4")
    assert recent.route.model == "model-a"
    # CONTRACT: recency changes the winner, never the accumulated counts.
    assert recent.api_call_count == 6


def test_attribution_rows_are_distinct_per_billing_route(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-key")
    for url, calls in (("https://u1", 2), ("https://u2", 5)):
        store_a.usage.queue_usage(
            "s-usage-key",
            UsageDelta(tokens=TokenCounts(input_tokens=1), api_call_count=calls,
                       route=BillingRoute(model="model-a", billing_provider="prov-a", billing_base_url=url)))
    dominant = store_a.usage.recent_route("s-usage-key")
    # CONTRACT: the same model under a different billing base URL is a separate attribution row;
    # the calls did not merge (a merged row would show 7) and the busier route wins.
    assert dominant.route.billing_base_url == "https://u2"
    assert dominant.api_call_count == 5


def test_auxiliary_usage_isolated_from_aggregate_and_recent_route(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-usage-5")
    store_a.usage.queue_usage(
        "s-usage-5",
        UsageDelta(tokens=TokenCounts(input_tokens=10), api_call_count=1,
                   route=BillingRoute(model="model-a", billing_provider="prov-a")))
    before = accounting(store_a, "s-usage-5")
    store_a.usage.record_auxiliary_usage(
        "s-usage-5", "vision", model="model-v", billing_provider="prov-v",
        tokens=TokenCounts(input_tokens=7, output_tokens=6, cache_read_tokens=5, cache_write_tokens=4, reasoning_tokens=3),
        api_call_count=2, estimated_cost_usd=0.2)
    # CONTRACT: auxiliary usage never lands on the session aggregate: every counter, the call
    # count, and the costs are unchanged.
    assert accounting(store_a, "s-usage-5") == before
    # CONTRACT: auxiliary rows never win the main-loop dominant route, even with more calls.
    assert (store_a.usage.recent_route("s-usage-5")).route.model == "model-a"


# ── typed reads and negatives ─────────────────────────────────────────────────


def test_unknown_session_reads_are_typed_none(backend, scope_a):
    # CONTRACT: reads against an unknown session are typed None returns, never exceptions.
    store_a = backend.open(scope_a)
    assert accounting(store_a, "s-nope") is None
    assert store_a.usage.recent_route("s-nope") is None


def test_records_validate_their_own_invariants(backend):
    # CONTRACT: negative counters and negative floors are unrepresentable.
    with pytest.raises(ValueError):
        TokenCounts(input_tokens=-1)
    with pytest.raises(ValueError):
        UsageDelta(api_call_count=-1)
    with pytest.raises(ValueError):
        UsageTotalsQuery(min_message_count=-1)


# ── totals ────────────────────────────────────────────────────────────────────


def test_usage_totals_sums_and_filters(backend, scope_a):
    usage = backend.open(scope_a).usage
    usage.seed_session("root-with-msg", message_count=1)
    usage.queue_usage(
        "root-with-msg",
        UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=5), api_call_count=1,
                   estimated_cost_usd=9.0, actual_cost_usd=1.0, route=ROUTE_A))
    usage.seed_session("root-estimated", message_count=1)
    usage.queue_usage(
        "root-estimated",
        UsageDelta(tokens=TokenCounts(input_tokens=20, output_tokens=5), api_call_count=1,
                   estimated_cost_usd=0.5, route=ROUTE_A))
    usage.seed_session("root-empty", message_count=0)
    usage.queue_usage(
        "root-empty",
        UsageDelta(tokens=TokenCounts(input_tokens=100), api_call_count=1, route=ROUTE_A))
    usage.seed_session("child-session", message_count=1, parent_session_id="root-with-msg")
    usage.queue_usage(
        "child-session",
        UsageDelta(tokens=TokenCounts(input_tokens=1000), api_call_count=1, route=ROUTE_A))
    assert usage.flush_usage() is True
    totals = usage.usage_totals(UsageTotalsQuery(min_message_count=1))
    # CONTRACT: totals sum input+output across qualifying sessions, prefer the billed cost over
    # the estimate per session (root-with-msg carries both: 1.0 billed beats the 9.0 estimate,
    # and summing both would show 10.5), and exclude both sessions below the message floor and
    # child sessions.
    assert totals.tokens == 40
    assert totals.cost_usd == pytest.approx(1.5)
    # CONTRACT: the default floor is one message.
    assert usage.usage_totals() == totals
    # CONTRACT: lowering the floor admits the message-less session.
    assert (usage.usage_totals(UsageTotalsQuery(min_message_count=0))).tokens == 140
    # CONTRACT: archived sessions drop out of the default totals and return with include_archived.
    usage.set_session_archived("root-estimated", True)
    assert (usage.usage_totals(UsageTotalsQuery(min_message_count=1))).tokens == 15
    assert (usage.usage_totals(
        UsageTotalsQuery(min_message_count=1, include_archived=True))).tokens == 40


# ── concurrency: aggregate equals sum of deltas ───────────────────────────────


def test_concurrent_delta_storm_aggregate_equals_sum_of_deltas(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.usage.seed_session("s-storm-1")
    delta = UsageDelta(
        tokens=TokenCounts(input_tokens=3, output_tokens=2, reasoning_tokens=1),
        api_call_count=1, estimated_cost_usd=0.01,
        route=BillingRoute(model="model-storm", billing_provider="prov-storm"))

    def actor() -> int:
        done = 0
        for _ in range(STORM_ROUNDS):
            store_a.usage.queue_usage("s-storm-1", delta)
            done += 1
        return done

    results = run_concurrently(*[lambda: actor() for _ in range(STORM_ACTORS)], return_exceptions=True)
    assert store_a.usage.flush_usage() is True
    # CONTRACT: no actor raises; every attempt completes.
    assert results == [STORM_ROUNDS] * STORM_ACTORS
    total_calls = STORM_ACTORS * STORM_ROUNDS
    usage = accounting(store_a, "s-storm-1")
    # CONTRACT: after a concurrent storm of deltas the aggregate equals the exact sum:
    # updates are atomic, none lost, none double-applied.
    assert usage.tokens.input_tokens == 3 * total_calls
    assert usage.tokens.output_tokens == 2 * total_calls
    assert usage.tokens.reasoning_tokens == 1 * total_calls
    assert usage.api_call_count == total_calls
    assert usage.estimated_cost_usd == pytest.approx(0.01 * total_calls)
    # CONTRACT: the attribution row saw the same exact call count.
    dominant = store_a.usage.recent_route("s-storm-1")
    assert dominant.route.model == "model-storm"
    assert dominant.api_call_count == total_calls


# ── two-profile isolation ──────────────────────────────────────────────────────


def test_two_scopes_are_usage_isolated_through_one_provider(backend, scope_a, scope_b):
    # Colliding session id, distinct sentinel counters per scope. Routes carry a
    # provider: the recent-route read deliberately excludes providerless
    # attribution rows (junk filter), and this case proves isolation, not the
    # providerless shape.
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    for scope, tokens in ((scope_a, 11), (scope_b, 77)):
        backend.open(scope).usage.seed_session("shared-sess", message_count=1)
        backend.open(scope).usage.queue_usage(
            "shared-sess",
            UsageDelta(tokens=TokenCounts(input_tokens=tokens), api_call_count=1,
                       route=BillingRoute(model=f"model-{tokens}", billing_provider="prov-iso")))
    # CONTRACT: each scope reads its own sentinel through every read surface.
    assert (accounting(store_a, "shared-sess")).tokens.input_tokens == 11
    assert (accounting(store_b, "shared-sess")).tokens.input_tokens == 77
    assert (store_a.usage.recent_route("shared-sess")).route.model == "model-11"
    assert (store_b.usage.recent_route("shared-sess")).route.model == "model-77"
    # CONTRACT: cross-scope mutation impotence: a delta in scope B never leaks into scope A's
    # aggregate or totals.
    store_b.usage.queue_usage(
        "shared-sess", UsageDelta(tokens=TokenCounts(input_tokens=1000), api_call_count=1))
    assert (accounting(store_a, "shared-sess")).tokens.input_tokens == 11
    assert store_b.usage.flush_usage() is True
    totals_a = store_a.usage.usage_totals(UsageTotalsQuery())
    assert totals_a.tokens == 11
    # CONTRACT: a session seeded only in scope A is a typed None in scope B.
    store_a.usage.seed_session("only-in-a")
    assert accounting(store_b, "only-in-a") is None
    # A third scope routed through the same provider sees nothing at all.
    scope_c = profile_scope("profile-c")
    assert (backend.open(scope_c).usage.usage_totals(UsageTotalsQuery())).tokens == 0
