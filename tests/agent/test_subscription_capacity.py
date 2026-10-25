from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from threading import Barrier

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.subscription_capacity import (
    CapacityRequest,
    DEFAULT_LEASE_TTL,
    ProviderPolicy,
    SubscriptionCapacityRouter,
    default_subscription_policies,
)
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


NOW = datetime(2026, 10, 25, 13, 0, tzinfo=timezone.utc)


def _policy(
    provider="openai-codex", *, floor=10.0, models=("gpt-",), sources=frozenset({"usage_api"}),
    identity=None, max_bounded=1,
):
    return ProviderPolicy(
        provider=provider,
        protected_floor_percent=floor,
        eligible_tasks=frozenset({"builder", "reviewer"}),
        model_prefixes=models,
        allowed_sources=sources,
        expected_identity=identity,
        max_bounded_leases=max_bounded,
    )


def _snapshot(
    provider="openai-codex", *, source="usage_api", age=timedelta(0), used=(20.0, 30.0),
    reset=(timedelta(hours=2), timedelta(days=5)), identity=None, windows=None,
):
    quota_windows = windows or tuple(
        AccountUsageWindow(f"w{i}", percent, NOW + reset[i]) for i, percent in enumerate(used)
    )
    return AccountUsageSnapshot(
        provider=provider, source=source, fetched_at=NOW - age,
        windows=quota_windows, identity=identity,
    )


def _router(tmp_path, policies, snapshots, *, now=lambda: NOW):
    calls = []

    def fetch(provider):
        calls.append(provider)
        value = snapshots[provider]
        return value() if callable(value) else value

    return SubscriptionCapacityRouter(
        tmp_path / "capacity.sqlite", policies, coordination_scope="test-subscriptions", fetcher=fetch, now=now,
    ), calls


def test_default_floors_keep_codex_interactive_allowance_and_require_configured_claude_floor():
    codex, claude = default_subscription_policies(claude_floor_percent=4.0)
    assert codex.protected_floor_percent == 20.0
    assert claude.protected_floor_percent == 4.0
    with pytest.raises(ValueError, match="positive"):
        default_subscription_policies(claude_floor_percent=0)


def test_effective_remaining_is_minimum_of_global_and_matching_model_windows(tmp_path):
    snapshot = _snapshot(windows=(
        AccountUsageWindow("session", 15, NOW + timedelta(hours=3)),
        AccountUsageWindow("week", 25, NOW + timedelta(days=3)),
        AccountUsageWindow("opus", 99, NOW + timedelta(days=3), model_prefixes=("claude-opus",)),
    ))
    router, _ = _router(tmp_path, (_policy(),), {"openai-codex": snapshot})

    result = router.acquire(
        CapacityRequest("req-1", "builder", "worker-a"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )

    assert result.lease is not None
    assessment = next(item for item in result.assessments if item.provider == "openai-codex")
    assert assessment.effective_remaining_percent == 75
    assert assessment.post_floor_slack_percent == 65


@pytest.mark.parametrize(
    ("snapshot", "policy", "reason"),
    [
        (_snapshot(age=timedelta(seconds=61)), _policy(), "stale"),
        (_snapshot(source="other"), _policy(), "source"),
        (_snapshot(reset=(timedelta(seconds=-1), timedelta(days=1))), _policy(), "expired"),
        (_snapshot(identity=None), _policy(identity="acct-1"), "identity"),
        (_snapshot(used=(-1, 20)), _policy(), "invalid"),
    ],
)
def test_stale_untrusted_expired_or_unidentified_usage_fails_closed(tmp_path, snapshot, policy, reason):
    router, _ = _router(tmp_path, (policy,), {"openai-codex": snapshot})
    result = router.acquire(
        CapacityRequest(f"req-{reason}", "builder", "worker"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert result.lease is None
    assert reason in result.assessments[0].reason


def test_routing_uses_slack_then_active_leases_then_earlier_reset(tmp_path):
    policies = (
        _policy(),
        _policy("anthropic", floor=5, models=("claude-",), sources=frozenset({"oauth_usage_api"}), max_bounded=3),
    )
    snapshots = {
        "openai-codex": _snapshot(used=(20, 20), reset=(timedelta(hours=4), timedelta(days=4))),
        "anthropic": _snapshot(
            "anthropic", source="oauth_usage_api", used=(17, 22),
            reset=(timedelta(hours=2), timedelta(days=4)),
        ),
    }
    router, _ = _router(tmp_path, policies, snapshots)
    # Slack is tied at 70%; Anthropic resets sooner and wins the deterministic near-tie rule.
    result = router.acquire(
        CapacityRequest("req-choice", "builder", "worker", reserved_percent=1),
        candidate_models={"openai-codex": "gpt-6-sol", "anthropic": "claude-sonnet-4-6"},
    )
    assert result.lease is not None
    assert result.lease.provider == "anthropic"
    assert result.lease.model == "claude-sonnet-4-6"


def test_unknown_gateway_alias_is_rejected_without_fetch_or_paid_worker_side_effect(tmp_path):
    router, calls = _router(tmp_path, (_policy(),), {"openai-codex": _snapshot()})
    result = router.acquire(
        CapacityRequest("req-paid", "builder", "worker"),
        candidate_models={"job/code": "openrouter/paid-model"},
    )
    assert result.lease is None
    assert calls == []
    assert result.assessments[0].reason == "not an admitted subscription provider"


def test_ineligible_model_is_rejected_before_reading_quota(tmp_path):
    router, calls = _router(tmp_path, (_policy(),), {"openai-codex": _snapshot()})
    result = router.acquire(
        CapacityRequest("req-model", "builder", "worker"),
        candidate_models={"openai-codex": "claude-sonnet-4-6"},
    )
    assert result.lease is None
    assert calls == []
    assert result.assessments[0].reason == "task or model is not eligible"


def test_atomic_cross_process_style_lease_allows_only_one_unbounded_worker(tmp_path):
    barrier = Barrier(2)
    policy = _policy()
    snapshot = _snapshot()
    routers = tuple(
        SubscriptionCapacityRouter(
            tmp_path / "shared.sqlite", (policy,), coordination_scope="test-subscriptions",
            fetcher=lambda provider: snapshot, now=lambda: NOW,
        )
        for _ in range(2)
    )

    def attempt(index):
        barrier.wait()
        return routers[index].acquire(
            CapacityRequest(f"req-{index}", "builder", f"worker-{index}"),
            candidate_models={"openai-codex": "gpt-6-sol"},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))

    assert sum(result.lease is not None for result in results) == 1
    refusal = next(result for result in results if result.lease is None)
    assert "unbounded" in refusal.assessments[0].reason


def test_bounded_reservations_are_atomic_and_cannot_cross_floor(tmp_path):
    router, _ = _router(tmp_path, (_policy(max_bounded=3),), {"openai-codex": _snapshot(used=(70, 70))})
    first = router.acquire(
        CapacityRequest("req-a", "builder", "a", reserved_percent=12),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    second = router.acquire(
        CapacityRequest("req-b", "builder", "b", reserved_percent=9),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert first.lease is not None
    assert second.lease is None
    assert second.assessments[0].reason == "protected floor would be crossed"


def test_phase_boundary_forces_refresh_and_stops_when_floor_is_reached(tmp_path):
    snapshots = iter((_snapshot(used=(20, 20)), _snapshot(used=(91, 91))))
    router, calls = _router(tmp_path, (_policy(),), {"openai-codex": lambda: next(snapshots)})
    result = router.acquire(
        CapacityRequest("req-phase", "reviewer", "worker"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert result.lease is not None

    phase = router.check_phase(result.lease)

    assert calls == ["openai-codex", "openai-codex"]
    assert not phase.admitted
    assert phase.reason == "protected floor would be crossed"


def test_bound_worker_reservation_survives_initial_ttl_and_can_renew(tmp_path, monkeypatch):
    clock = [NOW]

    def snapshot():
        return _snapshot(
            used=(20, 20), reset=(timedelta(hours=4), timedelta(days=5)),
            age=NOW - clock[0],
        )

    router, _ = _router(
        tmp_path, (_policy(),), {"openai-codex": snapshot}, now=lambda: clock[0],
    )
    first = router.acquire(
        CapacityRequest("req-long", "builder", "worker"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert first.lease is not None
    assert router.authorize_dispatcher_process(first.lease)
    assert router.register_worker_process(first.lease, os.getpid())

    clock[0] += timedelta(minutes=31)
    second = router.acquire(
        CapacityRequest("req-second", "builder", "other"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert second.lease is None
    assert "unbounded" in second.assessments[0].reason
    bound_pid = os.getpid()
    assert router.worker_process_matches(first.lease, bound_pid)
    with monkeypatch.context() as copied_process:
        copied_process.setattr("agent.subscription_capacity.os.getpid", lambda: bound_pid + 1)
        assert router.renew_bound_lease(first.lease) is None
    renewed = router.renew_bound_lease(first.lease)
    assert renewed is not None
    assert renewed.expires_at == clock[0] + DEFAULT_LEASE_TTL
    assert router.check_phase(renewed).admitted


def test_release_requires_the_lease_owner(tmp_path):
    router, _ = _router(tmp_path, (_policy(),), {"openai-codex": _snapshot()})
    result = router.acquire(
        CapacityRequest("req-release", "builder", "owner"),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert result.lease is not None
    assert not router.release(result.lease.lease_id, owner="someone-else")
    assert router.release(result.lease.lease_id, owner="owner")


def test_request_id_cannot_change_its_reservation_contract(tmp_path):
    router, _ = _router(tmp_path, (_policy(max_bounded=2),), {"openai-codex": _snapshot()})
    first = router.acquire(
        CapacityRequest("req-stable", "builder", "owner", reserved_percent=5),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    changed = router.acquire(
        CapacityRequest("req-stable", "builder", "owner", reserved_percent=10),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert first.lease is not None
    assert changed.lease is None
    assert changed.assessments[-1].reason == "request id belongs to a different lease"


@pytest.mark.parametrize(
    ("second_policy", "second_snapshot", "reason"),
    [
        (_policy(floor=85, max_bounded=2), _snapshot(used=(20, 20)), "protected floor"),
        (_policy(max_bounded=2), _snapshot(used=(95, 95)), "protected floor"),
    ],
)
def test_request_replay_in_a_new_router_rechecks_current_policy_and_fresh_usage(
    tmp_path, second_policy, second_snapshot, reason,
):
    db_path = tmp_path / "shared.sqlite"
    first = SubscriptionCapacityRouter(
        db_path, (_policy(max_bounded=2),), coordination_scope="operator-subscriptions",
        fetcher=lambda provider: _snapshot(used=(20, 20)), now=lambda: NOW,
    )
    request = CapacityRequest("req-replay", "builder", "owner", reserved_percent=5)
    admitted = first.acquire(request, candidate_models={"openai-codex": "gpt-6-sol"})
    assert admitted.lease is not None

    calls = []

    def second_fetch(provider):
        calls.append(provider)
        return second_snapshot

    restarted = SubscriptionCapacityRouter(
        db_path, (second_policy,), coordination_scope="operator-subscriptions",
        fetcher=second_fetch, now=lambda: NOW,
    )
    replay = restarted.acquire(request, candidate_models={"openai-codex": "gpt-6-sol"})

    assert replay.lease is None
    assert calls == ["openai-codex"]
    assert reason in replay.assessments[-1].reason


def test_request_replay_in_the_same_router_bypasses_admission_cache(tmp_path):
    snapshots = iter((_snapshot(used=(20, 20)), _snapshot(used=(95, 95))))
    router, calls = _router(
        tmp_path, (_policy(max_bounded=2),), {"openai-codex": lambda: next(snapshots)},
    )
    request = CapacityRequest("req-fresh-replay", "builder", "owner", reserved_percent=5)
    first = router.acquire(request, candidate_models={"openai-codex": "gpt-6-sol"})
    replay = router.acquire(request, candidate_models={"openai-codex": "gpt-6-sol"})

    assert first.lease is not None
    assert replay.lease is None
    assert calls == ["openai-codex", "openai-codex"]
    assert replay.assessments[-1].reason == "protected floor would be crossed"


def test_snapshot_cache_is_scoped_to_the_active_hermes_home_a_to_b_to_a(tmp_path):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    fetched = []

    def fetch(provider):
        home = hermes_home_key()
        fetched.append(home)
        used = (10, 10) if home == hermes_home_key(home_a) else (80, 80)
        return _snapshot(used=used)

    router = SubscriptionCapacityRouter(
        tmp_path / "capacity.sqlite", (_policy(),), coordination_scope="operator-subscriptions",
        fetcher=fetch, now=lambda: NOW,
    )
    token_a = set_hermes_home_override(home_a)
    try:
        a_first = router.snapshot("openai-codex")
    finally:
        reset_hermes_home_override(token_a)
    token_b = set_hermes_home_override(home_b)
    try:
        b = router.snapshot("openai-codex")
    finally:
        reset_hermes_home_override(token_b)
    token_a = set_hermes_home_override(home_a)
    try:
        a_again = router.snapshot("openai-codex")
    finally:
        reset_hermes_home_override(token_a)

    assert a_first is a_again
    assert a_first.windows[0].used_percent == 10
    assert b.windows[0].used_percent == 80
    assert fetched == [hermes_home_key(home_a), hermes_home_key(home_b)]


def test_capacity_database_refuses_an_accidental_coordination_scope_change(tmp_path):
    db_path = tmp_path / "capacity.sqlite"
    SubscriptionCapacityRouter(db_path, (_policy(),), coordination_scope="account-a", now=lambda: NOW)
    with pytest.raises(ValueError, match="belongs to coordination scope"):
        SubscriptionCapacityRouter(db_path, (_policy(),), coordination_scope="account-b", now=lambda: NOW)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "forged-request"), ("provider", "anthropic"), ("task", "reviewer"),
        ("owner", "forged-owner"), ("model", "gpt-forged"), ("reserved_percent", 7.0),
        ("expires_at", NOW + timedelta(days=1)),
    ],
)
def test_phase_check_rejects_a_supplied_lease_that_differs_from_durable_record(tmp_path, field, value):
    router, calls = _router(tmp_path, (_policy(max_bounded=2),), {"openai-codex": _snapshot()})
    result = router.acquire(
        CapacityRequest("req-durable", "builder", "owner", reserved_percent=5),
        candidate_models={"openai-codex": "gpt-6-sol"},
    )
    assert result.lease is not None
    assessment = router.check_phase(replace(result.lease, **{field: value}))
    assert not assessment.admitted
    assert assessment.reason == "lease record does not match supplied lease"
    assert calls == ["openai-codex"]
