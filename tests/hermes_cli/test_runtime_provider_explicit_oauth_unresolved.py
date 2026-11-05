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


def test_an_explicit_base_url_is_served_not_refused(oauth_provider):
    """``--base-url`` (a fallback entry, a cron job, a ``/model`` providers block) names the endpoint itself."""
    runtime = rp.resolve_runtime_provider(requested=oauth_provider, explicit_base_url="http://127.0.0.1:9/v1",
                                          explicit_api_key="k-explicit")
    assert runtime["base_url"] == "http://127.0.0.1:9/v1"
    assert runtime["api_key"] == "k-explicit"


@pytest.mark.parametrize(("code", "rate_limited"), [(429, True), (401, False)])
def test_a_benched_sign_in_names_its_cooldown_not_a_missing_login(oauth_provider, code, rate_limited):
    import time

    from hermes_cli.auth import is_rate_limited_auth_error

    _write_pool([{**_DEAD_ROW, "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": code}])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert "cooling down" in str(excinfo.value)
    assert "not signed in" not in str(excinfo.value)
    assert is_rate_limited_auth_error(excinfo.value) is rate_limited


def test_the_providers_own_url_still_means_this_provider(oauth_provider):
    """``/model`` stores the provider's own URL and sends it every turn: that is not an explicit endpoint."""
    _write_pool([_DEAD_ROW])
    with pytest.raises(AuthError):
        rp.resolve_runtime_provider(requested=oauth_provider, explicit_base_url="https://example.invalid/v1/")


@pytest.mark.parametrize("url", [" https://EXAMPLE.invalid/v1", "https://example.invalid:443/v1/"])
def test_the_providers_own_url_is_recognised_in_any_spelling(oauth_provider, url):
    _write_pool([_DEAD_ROW])
    with pytest.raises(AuthError):
        rp.resolve_runtime_provider(requested=oauth_provider, explicit_base_url=url)


def test_the_providers_own_url_with_its_own_key_is_served(oauth_provider):
    """``--api-key`` names the credential itself, so the stored sign-in's state does not matter."""
    _write_pool([_DEAD_ROW])
    runtime = rp.resolve_runtime_provider(requested=oauth_provider, explicit_base_url="https://example.invalid/v1",
                                          explicit_api_key="k-explicit")
    assert runtime["api_key"] == "k-explicit"


def test_a_rate_limited_sign_in_carries_its_wait_for_the_cron_hold(oauth_provider):
    import time

    reset = time.time() + 600
    _write_pool([{**_DEAD_ROW, "last_status": "exhausted", "last_status_at": time.time(),
                  "last_error_code": 429, "last_error_reset_at": reset}])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert 590 < excinfo.value.retry_after <= 600


def test_a_billing_bench_names_billing_not_a_sign_in(oauth_provider):
    import time

    _write_pool([{**_DEAD_ROW, "last_status": "exhausted", "last_status_at": time.time(),
                  "last_error_code": 403, "failure_reason": "billing"}])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert "Sign in again" not in str(excinfo.value)


def test_a_billing_bench_on_a_429_is_not_a_rate_limit_wait(oauth_provider):
    import time

    from hermes_cli.auth import is_rate_limited_auth_error

    _write_pool([{**_DEAD_ROW, "last_status": "exhausted", "last_status_at": time.time(),
                  "last_error_code": 429, "failure_reason": "billing"}])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert not is_rate_limited_auth_error(excinfo.value)


def test_a_revoked_old_row_does_not_hide_a_live_row_cooling_down(oauth_provider):
    import time

    live = {**_DEAD_ROW, "id": "live01", "access_token": "ya29.live", "refresh_token": "1//rt-live",
            "last_status": "exhausted", "last_status_at": time.time(), "last_error_code": 429,
            "last_error_reset_at": time.time() + 600}
    _write_pool([_DEAD_ROW, live])
    with pytest.raises(AuthError) as excinfo:
        rp.resolve_runtime_provider(requested=oauth_provider, target_model="some-model")
    assert excinfo.value.retry_after is not None and "sign-in was lost" not in str(excinfo.value)
