"""Plugin providers in the credential pool (#116408): opaque metadata round-trips and refresh goes
through the profile's ``refresh_credential`` hook — eligibility derives from the hook, never a name set."""

from __future__ import annotations

from dataclasses import replace

import pytest

import providers
from providers.base import ProviderProfile

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from hermes_cli.auth_plugin_providers import is_refreshable_oauth_provider


def _entry(**over):
    base = dict(provider="example-oauth", id="abc123", label="acme", auth_type=AUTH_TYPE_OAUTH, priority=0,
                source="manual:example_device", access_token="tok-1", refresh_token="rt-1",
                extra={"tenant": "acme", "region": "eu"})
    return PooledCredential(**{**base, **over})


def test_plugin_metadata_survives_load_save_load():
    payload = _entry().to_dict()
    again = PooledCredential.from_dict("example-oauth", payload).to_dict()
    assert again["tenant"] == "acme" and again["region"] == "eu"
    assert PooledCredential.from_dict("example-oauth", again).extra == {"tenant": "acme", "region": "eu"}
    # Core-known extra keys keep their attribute surface; unknown ones stay opaque payload.
    assert PooledCredential.from_dict("nous", {"access_token": "t", "org_id": "o1"}).org_id == "o1"


@pytest.fixture
def plugin_profiles():
    seen = []

    def refresh_credential(entry):
        seen.append(entry.refresh_token)
        return {"access_token": "tok-2", "refresh_token": "rt-2"}

    providers.register_provider(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1",
                                                refresh_credential=refresh_credential))
    providers.register_provider(ProviderProfile(name="example-oauth-nohook", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1"))
    yield seen
    for name in ("example-oauth", "example-oauth-nohook"):
        providers._REGISTRY.pop(name, None)
    providers._PROVIDER_LIST_CACHE = None


def test_pool_refresh_dispatches_to_profile_hook(plugin_profiles, monkeypatch):
    assert is_refreshable_oauth_provider("example-oauth") is True
    assert is_refreshable_oauth_provider("example-oauth-nohook") is False
    assert is_refreshable_oauth_provider("anthropic") is True  # built-ins unchanged

    entry = _entry()
    pool = CredentialPool("example-oauth", [entry])
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)
    refreshed = pool._refresh_entry_impl(entry, force=True)
    assert plugin_profiles == ["rt-1"]
    assert (refreshed.access_token, refreshed.refresh_token, refreshed.extra) == ("tok-2", "rt-2", entry.extra)

    # Without the hook the pool must not pretend it refreshed anything.
    nohook = replace(entry, provider="example-oauth-nohook")
    assert CredentialPool("example-oauth-nohook", [nohook])._refresh_entry_impl(nohook, force=True) is nohook
