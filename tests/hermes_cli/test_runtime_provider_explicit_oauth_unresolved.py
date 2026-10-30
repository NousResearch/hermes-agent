"""An explicitly chosen OAuth provider profile with no usable sign-in fails with a typed AuthError.

Regression: a plugin OAuth provider (``auth_type="oauth_external"`` + ``refresh_credential``) whose
pool row was DEAD (refresh grant rejected) fell through the ladder to the OpenRouter fallback, so
with ``OPENROUTER_API_KEY`` set every request for that provider's model went to OpenRouter (402)
before the operator's ``fallback_providers`` chain was ever consulted.
"""

from __future__ import annotations

import json

import pytest

import hermes_cli.auth
import providers
from hermes_cli import runtime_provider as rp
from hermes_cli.auth_constants import AuthError
from hermes_constants import get_hermes_home
from providers import register_provider
from providers.base import ProviderProfile

NAME = "example-signin-oauth"


def _refresh_rejected(entry):
    raise AuthError("refresh rejected", provider=NAME, code="invalid_grant", relogin_required=True)


@pytest.fixture
def oauth_provider(monkeypatch, request):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-unrelated")
    register_provider(ProviderProfile(name=NAME, auth_type="oauth_external", base_url="https://example.invalid/v1",
                                      refresh_credential=_refresh_rejected))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(NAME, None), hermes_cli.auth.PROVIDER_REGISTRY.pop(NAME, None)))
    assert hermes_cli.auth.PROVIDER_REGISTRY[NAME].auth_type == "oauth_external"
    return NAME


def _write_pool(rows: list[dict]) -> None:
    (get_hermes_home() / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {NAME: rows}}),
                                                 encoding="utf-8")


_DEAD_ROW = {"id": "dead01", "label": NAME, "auth_type": "oauth", "priority": 0, "source": "manual:oauth",
             "access_token": "ya29.revoked", "refresh_token": "1//rt-revoked", "last_status": "dead",
             "last_error_reason": "invalid_grant", "last_error_message": "Token has been expired or revoked."}


def test_dead_sign_in_raises_instead_of_routing_to_openrouter(oauth_provider):
    _write_pool([_DEAD_ROW])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    err = excinfo.value
    assert err.provider == oauth_provider
    assert err.relogin_required is True
    assert "Token has been expired or revoked." in str(err)
    assert f"auth add {oauth_provider}" in str(err)


def test_never_signed_in_raises_instead_of_routing_to_openrouter(oauth_provider):
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert excinfo.value.provider == oauth_provider
    assert excinfo.value.relogin_required is False
    assert "not signed in" in str(excinfo.value)


def test_operator_fallback_chain_still_runs(oauth_provider):
    """The typed error hands the decision to ``fallback_providers`` — an explicit OpenRouter entry works."""
    _write_pool([_DEAD_ROW])
    config = {"fallback_providers": [{"provider": "openrouter", "model": "fallback/model"}]}
    runtime, entry = rp.resolve_runtime_with_fallback(config, requested=oauth_provider, target_model="some-model")
    assert entry == {"provider": "openrouter", "model": "fallback/model"}
    assert runtime["provider"] == "openrouter"


def test_auto_keeps_the_openrouter_default(oauth_provider):
    _write_pool([_DEAD_ROW])
    assert rp.resolve_runtime_provider(requested="auto")["provider"] == "openrouter"
