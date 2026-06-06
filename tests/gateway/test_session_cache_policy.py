"""Unit tests for gateway session cache policy seam."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from gateway.runtime.session_cache_policy import (
    active_agent_ids,
    apply_cache_eviction_plan,
    plan_idle_cache_evictions,
    plan_lru_cache_evictions,
)


class Agent(SimpleNamespace):
    pass


def test_lru_policy_evicts_only_excess_lru_entries():
    cache = OrderedDict(
        (
            ("s0", (Agent(), "sig0")),
            ("s1", (Agent(), "sig1")),
            ("s2", (Agent(), "sig2")),
            ("s3", (Agent(), "sig3")),
        )
    )

    plan, remaining = plan_lru_cache_evictions(cache, max_size=2)

    assert [key for key, _agent in plan] == ["s0", "s1"]
    assert remaining == 0
    assert list(cache.keys()) == ["s0", "s1", "s2", "s3"]

    apply_cache_eviction_plan(cache, plan)
    assert list(cache.keys()) == ["s2", "s3"]


def test_lru_policy_skips_active_without_substitute_eviction():
    active = Agent()
    idle_second = Agent()
    idle_third = Agent()
    cache = OrderedDict(
        (
            ("active", (active, "sig")),
            ("idle-second", (idle_second, "sig")),
            ("idle-third", (idle_third, "sig")),
        )
    )

    plan, remaining = plan_lru_cache_evictions(
        cache,
        running_agents={"active": active},
        max_size=2,
    )

    assert plan == []
    assert remaining == 1


def test_lru_policy_pending_sentinel_is_not_active():
    pending = object()
    evicted = Agent()
    kept = Agent()
    cache = OrderedDict((("evicted", (evicted, "sig")), ("kept", (kept, "sig"))))

    plan, remaining = plan_lru_cache_evictions(
        cache,
        running_agents={"constructing": pending},
        max_size=1,
        pending_sentinel=pending,
    )

    assert plan == [("evicted", evicted)]
    assert remaining == 0


def test_lru_policy_plain_dict_is_noop_for_legacy_fixture_tolerance():
    cache = {f"s{i}": (Agent(), f"sig{i}") for i in range(5)}

    plan, remaining = plan_lru_cache_evictions(cache, max_size=1)

    assert plan == []
    assert remaining == 0
    assert len(cache) == 5


def test_idle_policy_evicts_stale_skips_fresh_active_and_missing_timestamp():
    now = 100.0
    stale = Agent(_last_activity_ts=10.0)
    fresh = Agent(_last_activity_ts=99.0)
    active_stale = Agent(_last_activity_ts=0.0)
    missing_ts = Agent()
    cache = OrderedDict(
        (
            ("stale", (stale, "sig")),
            ("fresh", (fresh, "sig")),
            ("active-stale", (active_stale, "sig")),
            ("missing", (missing_ts, "sig")),
        )
    )

    plan = plan_idle_cache_evictions(
        cache,
        running_agents={"active-stale": active_stale},
        idle_ttl_secs=60.0,
        now=now,
    )

    assert plan == [("stale", stale)]
    apply_cache_eviction_plan(cache, plan)
    assert list(cache.keys()) == ["fresh", "active-stale", "missing"]


def test_active_agent_ids_ignores_none_and_pending_sentinel():
    pending = object()
    active = Agent()

    assert active_agent_ids(
        {"none": None, "pending": pending, "active": active},
        pending_sentinel=pending,
    ) == {id(active)}
