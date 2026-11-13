"""A plugin's returned expiry controls dispatch without requiring legacy hooks to add it."""

import pytest

import providers
from agent.credential_pool import load_pool
from hermes_cli import auth
from providers.base import ProviderProfile


@pytest.mark.parametrize("expiry", [None, 4102444800000, 0], ids=["omitted", "future", "expired"])
@pytest.mark.parametrize("path", ["post", "peer-refresh", "peer-recovery"])
def test_rotated_plugin_tokens_only_require_an_expiry_when_returned(monkeypatch, expiry, path):
    provider = "example-expiry-oauth"
    calls = []

    def refresh(entry):
        calls.append(entry.refresh_token)
        result = {"access_token": f"access-{len(calls)}", "refresh_token": f"refresh-{len(calls)}"}
        if expiry is not None:
            result["expires_at_ms"] = expiry
        return result

    profile = ProviderProfile(name=provider, auth_type="oauth_external", refresh_credential=refresh)
    monkeypatch.setitem(providers._REGISTRY, provider, profile)
    auth.write_credential_pool(provider, [{
        "id": "owned", "auth_type": "oauth", "source": "manual:example",
        "access_token": "access-old", "refresh_token": "refresh-old", "expires_at_ms": 1,
    }])
    pool = load_pool(provider)
    peer = load_pool(provider)

    selected = pool.select()
    assert calls == ["refresh-old"]
    assert auth.read_credential_pool(provider)[0]["refresh_token"] == "refresh-1"
    result = selected
    if path == "peer-refresh":
        result = peer.try_refresh_matching(credential_id="owned")
    elif path == "peer-recovery":
        result = peer._recover_failed_refresh(peer.entries()[0], RuntimeError("offline"))
    if expiry == 0:
        assert selected is None
        assert result is None
    else:
        assert selected is not None and selected.access_token == "access-1"
        assert result is not None and result.access_token == "access-1"
        assert calls == ["refresh-old"]
        if path == "post":
            refreshed = pool.try_refresh_matching(credential_id="owned")
            assert refreshed is not None and refreshed.access_token == "access-2"
            assert calls == ["refresh-old", "refresh-1"]
