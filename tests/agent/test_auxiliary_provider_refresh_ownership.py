"""Auxiliary renewal preserves pooled ownership and provider-owned runtime hooks."""

from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from providers.base import ProviderProfile


def test_nonpooled_provider_runtime_refresh_evicts_its_cached_clients(monkeypatch):
    reasons = []
    evicted = []

    class RuntimeProvider(ProviderProfile):
        def refresh_runtime_credentials(self, *, reason):
            reasons.append(reason)
            return "fresh-runtime-token", None

    profile = RuntimeProvider(name="runtime-refresh-fixture")
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(aux, "_evict_cached_clients", evicted.append)

    assert aux._refresh_provider_credentials(profile.name) is True
    assert reasons == ["auxiliary_unauthorized"]
    assert evicted == [profile.name]


@pytest.mark.parametrize("refreshed", [False, True])
def test_pooled_refresh_never_falls_through_to_an_unbound_runtime_hook(monkeypatch, refreshed):
    runtime_reasons = []
    pool_calls = []
    evicted = []

    class PooledProvider(ProviderProfile):
        def refresh_runtime_credentials(self, *, reason):
            runtime_reasons.append(reason)
            return "unrelated-runtime-token", None

    profile = PooledProvider(name="pooled-refresh-fixture", refresh_credential=lambda entry: {})

    def refresh_matching(**kwargs):
        pool_calls.append(kwargs)
        return SimpleNamespace(runtime_api_key="rotated-token") if refreshed else None

    pool = SimpleNamespace(
        entries=lambda: [SimpleNamespace(id="issuing-grant", runtime_api_key="expired-token")],
        try_refresh_matching=refresh_matching,
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda name: pool)
    monkeypatch.setattr(aux, "_evict_cached_clients", evicted.append)

    assert aux._refresh_provider_credentials(
        profile.name, failed_api_key="expired-token", credential_id="issuing-grant",
    ) is refreshed
    assert pool_calls == [{"api_key_hint": "expired-token", "credential_id": "issuing-grant"}]
    assert runtime_reasons == []
    assert evicted == ([profile.name] if refreshed else [])
