"""Independent-review regressions: native identity, eligibility and concurrency."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import quota_routing as quota
from agent.quota_preflight import candidate_quota, preflight_quota_route, retain_quota_fallback
from tests.agent.test_quota_preflight import FALLBACK, fallback_client, make_agent
from tests.agent.test_quota_routing import URL, snapshot


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    quota._cache.clear()
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 200_000)
    yield
    quota._cache.clear()
    assert not quota._inflight


def pool_for(*keys):
    entries = [SimpleNamespace(runtime_api_key=key, runtime_base_url=URL) for key in keys]
    return SimpleNamespace(provider="openai-codex", entries=lambda: entries, current=lambda: entries[0])


def test_preflight_does_not_bypass_multi_account_rotation(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr("agent.credential_pool._codex_principal_identity", lambda key: ("workspace", key))
    agent._credential_pool = pool_for(agent.api_key, "other-account")
    with patch("agent.quota_preflight.current_quota") as probe, patch("agent.route_binding.bind_route_entry") as bind:
        assert not preflight_quota_route(agent)
    probe.assert_not_called()
    bind.assert_not_called()
    assert agent._fallback_index == 0


def test_duplicate_account_pool_can_preflight(monkeypatch):
    agent = make_agent()
    monkeypatch.setattr("agent.credential_pool._codex_principal_identity", lambda key: ("workspace", "member"))
    agent._credential_pool = pool_for(agent.api_key, "refreshed-token")
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")
    ):
        assert preflight_quota_route(agent)
    assert agent._quota_primary_pool is not None


def test_retention_reads_live_pool_not_stale_init_key():
    agent = SimpleNamespace(_quota_fallback_active=True, _quota_primary_pool=pool_for("refreshed-token"))
    with patch("agent.quota_preflight.account_quota", return_value=quota.QuotaVerdict("available", "test")) as probe:
        assert not retain_quota_fallback(agent, dict(provider="openai-codex", base_url=URL, api_key="stale-init"))
    probe.assert_called_once_with("openai-codex", URL, "refreshed-token")


@pytest.mark.parametrize("state", ["unknown", "fetch_failed"])
def test_unknown_retention_still_obeys_native_cooldown(state):
    import time
    agent = make_agent()
    agent._fallback_activated = True
    agent._quota_fallback_active = True
    agent._rate_limited_until = time.monotonic() + 60
    with patch("agent.quota_preflight.account_quota", return_value=quota.QuotaVerdict(state, "test")), patch(
        "agent.agent_runtime_helpers._rebuild_primary_client"
    ) as rebuild:
        assert not agent._restore_primary_runtime()
    rebuild.assert_not_called()


def test_failed_speculative_walk_preserves_index_cooldown_and_reactive_retry():
    agent = make_agent()
    original_chain = list(agent._fallback_chain)
    agent._rate_limited_until = 0
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), patch(
        "agent.quota_preflight.candidate_quota", return_value=quota.QuotaVerdict("exhausted", "test")
    ), patch("agent.route_binding.bind_route_entry") as bind:
        assert not preflight_quota_route(agent)
        assert not preflight_quota_route(agent)
    bind.assert_not_called()
    assert agent._fallback_index == 0
    assert agent._rate_limited_until == 0
    assert agent._fallback_chain == original_chain
    from agent.chat_completion_helpers import try_activate_fallback
    with patch("agent.quota_preflight.candidate_quota", return_value=quota.QuotaVerdict("unknown", "test")), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")
    ):
        assert try_activate_fallback(agent)


def test_quota_notice_does_not_claim_provider_failure():
    agent = make_agent()
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")
    ), patch("agent.chat_completion_helpers._buffer_fallback_notice") as notice:
        assert preflight_quota_route(agent)
    assert "account quota exhausted" in notice.call_args.args[1]
    assert "provider failure" not in notice.call_args.args[1]


def test_exhausted_reactive_fallback_does_not_pin_unobserved_primary():
    agent = make_agent()
    agent._fallback_activated = True
    agent._quota_fallback_active = False
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), patch(
        "agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client(), "fallback-model")
    ):
        assert preflight_quota_route(agent)
    assert agent._quota_fallback_active is False


@pytest.mark.parametrize("excluded", ["unavailable", "malformed", "entitlement", "pool", "local", "identity"])
def test_native_exclusions_precede_quota_network(excluded):
    from agent.fallback_candidates import _should_skip_fallback_candidate
    agent = make_agent()
    entry = dict(FALLBACK)
    key = ("synthetic",)
    unavailable = {key} if excluded == "unavailable" else set()
    with patch("agent.quota_preflight.candidate_quota") as probe, patch(
        "agent.fallback_cooldown._is_entitlement_rejected", return_value=excluded == "entitlement"
    ), patch("agent.chat_completion_helpers._candidate_pool_exhausted", return_value=excluded == "pool"), patch(
        "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
        return_value="missing" if excluded == "local" else None
    ), patch("agent.backend_identity.should_skip_candidate", return_value=excluded == "identity"):
        assert _should_skip_fallback_candidate(agent, entry, key, "" if excluded == "malformed" else "custom", "fallback-model", unavailable)
    probe.assert_not_called()


@pytest.mark.parametrize("fallback_active", [False, True])
def test_keyless_codex_entry_does_not_borrow_unproven_explicit_key(fallback_active):
    agent = make_agent()
    agent._fallback_activated = fallback_active
    with patch("agent.account_usage.fetch_account_usage") as fetch:
        assert candidate_quota(agent, dict(provider="openai-codex", model="other-model")).state == "unknown"
    fetch.assert_not_called()


def test_keyless_entry_cannot_borrow_explicit_fallback_key_even_with_pool():
    agent = make_agent()
    agent._fallback_activated = True
    agent._credential_pool = pool_for(agent.api_key)
    with patch("agent.account_usage.fetch_account_usage") as fetch:
        assert candidate_quota(agent, dict(provider="openai-codex", model="other-model")).state == "unknown"
    fetch.assert_not_called()


def test_independent_group_refresh_and_cache_hits_do_not_wait_for_slow_network(monkeypatch):
    entered, release = Event(), Event()
    def fetch(provider, *, api_key, **kwargs):
        if api_key == "slow":
            entered.set()
            assert release.wait(5)
        return snapshot()
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    quota.account_quota("openai-codex", URL, "cached")
    with ThreadPoolExecutor(max_workers=3) as workers:
        slow = workers.submit(quota.account_quota, "openai-codex", URL, "slow")
        assert entered.wait(3)
        try:
            for key in ("independent", "cached"):
                assert workers.submit(quota.account_quota, "openai-codex", URL, key).result(timeout=2).state == "exhausted"
        finally:
            release.set()
        assert slow.result(timeout=3).state == "exhausted"


def test_flight_capacity_is_bounded_and_released(monkeypatch):
    monkeypatch.setattr(quota, "MAX_CACHE_ENTRIES", 1)
    entered, release = Event(), Event()
    def fetch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return snapshot()
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(quota.account_quota, "openai-codex", URL, "first")
        assert entered.wait(3)
        try:
            assert quota.account_quota("openai-codex", URL, "second").state == "unknown"
            assert len(quota._inflight) == 1
        finally:
            release.set()
        pending.result(timeout=3)
    assert not quota._inflight


@pytest.mark.parametrize("child", [False, True])
def test_authorized_chain_is_never_expanded_or_repriced(child):
    entry = {**FALLBACK, "routing_constraint": "private-reviewed-only", "cost_ceiling": 0}
    agent = make_agent(child=child, chain=[entry])
    original = [dict(item) for item in agent._fallback_chain]
    with patch("agent.account_usage.fetch_account_usage", return_value=snapshot()), patch(
        "agent.route_binding.bind_route_entry", return_value=None
    ) as bind:
        assert not preflight_quota_route(agent)
    assert agent._fallback_chain == original
    assert bind.call_count == 1
    assert bind.call_args.args[1] == original[0]
    assert bind.call_args.args[2:] == ("custom", "fallback-model")


def test_voice_snapshot_restores_quota_provenance():
    from agent.voice_turn_route import _capture, _reinstall
    agent = make_agent()
    pool = pool_for(agent.api_key)
    agent._quota_fallback_active = False
    agent._quota_primary_pool = pool
    state = _capture(agent)
    agent._quota_fallback_active = True
    agent._quota_primary_pool = None
    with patch("agent.route_binding.reinstall_runtime_snapshot"):
        _reinstall(agent, state)
    assert agent._quota_fallback_active is False
    assert agent._quota_primary_pool is pool
