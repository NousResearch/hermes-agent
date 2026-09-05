from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


def _route(provider="zai", model="glm-5.3-flash", *, scope="default", effort=None):
    from agent.provider_health import ProviderRoute

    return ProviderRoute(provider, model, credential_scope=scope, reasoning_effort=effort)


def test_declared_quota_reset_is_shared_and_never_shortened(tmp_path):
    from agent.provider_health import ProviderHealthStore

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    store = ProviderHealthStore(tmp_path)
    route = _route()
    later = now + timedelta(days=6)
    earlier = now + timedelta(hours=1)

    first = store.record_failure(
        route,
        f"Weekly/Monthly Limit Exhausted; reset_at={later.isoformat()}",
        source="cron:eng-completion",
        now=now,
    )
    second = store.record_failure(
        route,
        f"quota exhausted; reset_at={earlier.isoformat()}",
        source="kanban:t-1",
        now=now + timedelta(minutes=1),
    )

    assert first.kind == "quota"
    assert second.until == first.until
    state = ProviderHealthStore(tmp_path).get(route)
    assert state is not None
    assert state.until == later
    assert state.reset_at == later
    assert state.source == "kanban:t-1"
    assert "quota exhausted" in state.reason


def test_generic_429_is_transient_not_confirmed_quota(tmp_path):
    from agent.provider_health import ProviderHealthStore

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    state = ProviderHealthStore(tmp_path).record_failure(
        _route(), "HTTP 429 Too Many Requests", source="cron:job", now=now
    )

    assert state.kind == "rate_limit"
    assert now + timedelta(minutes=14) < state.until < now + timedelta(minutes=17)


def test_route_decision_skips_open_primary_and_selects_sol_medium(tmp_path):
    from agent.provider_health import ProviderHealthStore

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    primary = _route()
    fallback = _route("openai-codex", "gpt-5.6-sol", effort="medium")
    store = ProviderHealthStore(tmp_path)
    store.record_failure(
        primary,
        "Weekly Limit Exhausted; reset_at=2026-09-11T00:00:00Z",
        source="cron:eng-completion",
        now=now,
    )

    decision = store.decide([primary, fallback], owner="kanban:t-1", now=now)

    assert decision.route == fallback
    assert decision.skipped == (primary,)
    assert decision.deferred_until is None


def test_all_open_routes_defer_durably_across_restart(tmp_path):
    from agent.provider_health import ProviderHealthStore

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    routes = [_route(), _route("openai-codex", "gpt-5.6-sol", effort="medium")]
    store = ProviderHealthStore(tmp_path)
    for route in routes:
        store.record_failure(
            route,
            "quota exhausted; reset_at=2026-09-11T00:00:00Z",
            source="kanban:t-1",
            now=now,
        )

    first = store.decide(routes, owner="kanban:t-1", now=now)
    second = ProviderHealthStore(tmp_path).decide(routes, owner="kanban:t-1", now=now)

    assert first.route is None
    assert second.route is None
    assert first.deferred_until == second.deferred_until == datetime(
        2026, 9, 11, tzinfo=timezone.utc
    )


def test_post_reset_probe_is_single_flight_and_failure_reopens(tmp_path):
    from agent.provider_health import ProviderHealthStore

    before = datetime(2026, 9, 5, tzinfo=timezone.utc)
    reset = before + timedelta(hours=1)
    route = _route()
    store = ProviderHealthStore(tmp_path)
    store.record_failure(
        route,
        f"quota exhausted; reset_at={reset.isoformat()}",
        source="cron:job",
        now=before,
    )

    owner = store.decide([route], owner="kanban:t-1", now=reset)
    rival = ProviderHealthStore(tmp_path).decide(
        [route], owner="cron:job", now=reset + timedelta(seconds=1)
    )

    assert owner.route == route
    assert owner.probe is True
    assert rival.route is None
    assert rival.deferred_until == reset + timedelta(minutes=5)

    reopened = store.record_failure(
        route,
        "quota exhausted; reset_at=2026-09-12T00:00:00Z",
        source="kanban:t-1",
        owner="kanban:t-1",
        now=reset + timedelta(minutes=1),
    )
    assert reopened.until == datetime(2026, 9, 12, tzinfo=timezone.utc)
    assert store.decide([route], owner="cron:job", now=reset + timedelta(minutes=2)).route is None


def test_credential_scope_and_profile_home_are_isolated(tmp_path):
    from agent.provider_health import ProviderHealthStore

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    blocked = _route(scope="pool-a")
    sibling_pool = _route(scope="pool-b")
    other_home = tmp_path / "other"
    store = ProviderHealthStore(tmp_path / "one")
    store.record_failure(
        blocked,
        "quota exhausted; reset_at=2026-09-11T00:00:00Z",
        source="kanban:t-1",
        now=now,
    )

    assert store.decide([sibling_pool], owner="kanban:t-2", now=now).route == sibling_pool
    assert ProviderHealthStore(other_home).decide(
        [blocked], owner="kanban:t-3", now=now
    ).route == blocked


def test_agent_failure_records_exact_active_route_and_credential_pool(tmp_path, monkeypatch):
    from agent.provider_health import ProviderHealthStore, ProviderRoute, record_agent_failure

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    agent = SimpleNamespace(
        provider="zai",
        model="glm-5.3-flash",
        _credential_pool_entry_id="zai-key-2",
        session_id="worker-session",
    )

    record_agent_failure(
        agent,
        "Weekly Limit Exhausted; reset_at=2026-09-11T00:00:00Z",
        reason="billing",
        now=now,
    )

    assert ProviderHealthStore(tmp_path).get(
        ProviderRoute("zai", "glm-5.3-flash", credential_scope="zai-key-2")
    ) is not None
    assert ProviderHealthStore(tmp_path).get(
        ProviderRoute("zai", "glm-5.3-flash", credential_scope="default")
    ) is not None


def test_agent_fallback_skips_known_unavailable_route_and_success_clears_probe(tmp_path, monkeypatch):
    from agent.provider_health import (
        ProviderHealthStore,
        ProviderRoute,
        agent_route_decision,
        record_agent_success,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-7")
    reset = datetime(2026, 9, 5, tzinfo=timezone.utc)
    route = ProviderRoute("zai", "glm-5.3-flash")
    store = ProviderHealthStore(tmp_path)
    store.record_failure(
        route,
        f"quota exhausted; reset_at={reset.isoformat()}",
        source="agent:old",
        now=reset - timedelta(hours=1),
    )
    agent = SimpleNamespace(provider=route.provider, model=route.model)

    blocked = agent_route_decision(agent, route, now=reset - timedelta(seconds=1))
    assert blocked.route is None
    decision = agent_route_decision(agent, route, now=reset)
    assert decision.route == route
    assert decision.probe is True

    record_agent_success(agent)
    assert store.get(route) is None
