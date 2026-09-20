"""Plugin providers in the credential pool (#116408): opaque metadata round-trips and refresh goes
through the profile's ``refresh_credential`` hook — eligibility derives from the hook, never a name set."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

import providers
from providers.base import ProviderProfile

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    STATUS_OK,
    CredentialPool,
    PooledCredential,
)
from hermes_cli.auth import read_credential_pool
from hermes_cli.auth_constants import AuthError
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
    # A stray row-level ``provider`` is pool bookkeeping, not plugin metadata: it must not be swept
    # into ``extra`` and written back over the owning provider's row.
    assert "provider" not in PooledCredential.from_dict(
        "example-oauth", {**payload, "provider": "other"}).to_dict()


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


def _register_hook(hook):
    providers.register_provider(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1",
                                                refresh_credential=hook))


def _token_endpoint_shape(entry):
    # The natural token-endpoint response: field keys plus keys that are not dataclass fields.
    return {"access_token": "tok-2", "refresh_token": "rt-2", "expires_in": 3600,
            "expires_at_ms": 4102444800000, "token_type": "Bearer"}


def _raise_transient(entry):
    raise RuntimeError("token endpoint said 503")


def _raise_terminal(entry):
    raise AuthError("invalid_grant", provider="example-oauth", code="example_refresh_failed",
                    relogin_required=True)


@pytest.mark.parametrize("hook, status, tokens, extra_key", [
    (_token_endpoint_shape, STATUS_OK, ("tok-2", "rt-2"), "expires_in"),
    (_raise_transient, STATUS_EXHAUSTED, ("tok-1", "rt-1"), None),
    (_raise_terminal, STATUS_DEAD, ("tok-1", "rt-1"), None),
], ids=["token-endpoint-shape-rotates", "transient-error-benches", "relogin-required-is-terminal"])
def test_plugin_refresh_outcome(plugin_profiles, caplog, hook, status, tokens, extra_key):
    _register_hook(hook)
    entry = _entry(expires_at_ms=1)
    pool = CredentialPool("example-oauth", [entry])
    pool._persist()
    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        pool._refresh_entry(entry, force=True)
    row = PooledCredential.from_dict("example-oauth", read_credential_pool("example-oauth")[0])
    assert (row.last_status, row.access_token, row.refresh_token) == (status, *tokens)
    if extra_key:
        assert row.extra[extra_key] == 3600 and row.expires_at_ms == 4102444800000
    if status == STATUS_DEAD:
        assert "hermes auth add example-oauth" in caplog.text
    else:
        assert "hermes auth add" not in caplog.text


def test_plugin_refresh_adopts_peer_rotation_without_spending_token(plugin_profiles):
    """Two Hermes processes share one auth.json: the second refresh adopts the first's rotated pair
    instead of POSTing the same single-use refresh token again."""
    calls = []

    def hook(entry):
        calls.append(entry.refresh_token)
        return {"access_token": f"tok-{len(calls) + 1}", "refresh_token": f"rt-{len(calls) + 1}"}

    _register_hook(hook)
    first = CredentialPool("example-oauth", [_entry(expires_at_ms=1)])
    first._persist()
    second = CredentialPool("example-oauth", [_entry(expires_at_ms=1)])  # stale copy, same on-disk row

    assert first._refresh_entry(first.entries()[0], force=True).access_token == "tok-2"
    adopted = second._refresh_entry(second.entries()[0], force=True)
    assert (adopted.access_token, adopted.refresh_token) == ("tok-2", "rt-2")
    assert calls == ["rt-1"]
