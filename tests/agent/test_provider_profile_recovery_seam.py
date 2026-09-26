"""A pure model-provider plugin participates in error classification and the 401 refresh path through its
``ProviderProfile`` hooks — no PluginManager hook lifecycle, no core provider-name table (#116408 §4)."""

from __future__ import annotations

import httpx
import openai
import pytest

import providers
from providers.base import ProviderProfile

from agent.credential_pool import AUTH_TYPE_OAUTH, STATUS_EXHAUSTED, CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason, classify_api_error


def _error(status: int, code: str) -> openai.APIStatusError:
    body = {"error": {"message": "vendor refused", "code": code}}
    resp = httpx.Response(status, json=body, request=httpx.Request("POST", "https://example.invalid/v1/chat"))
    return openai.APIStatusError("vendor refused", response=resp, body=body)


@pytest.fixture
def plugin_profiles():
    def classify(error, *, status_code, error_code, message, body, model):
        if status_code == 403 and error_code == "quota_exhausted":
            return {"reason": "billing", "retryable": False, "should_rotate_credential": True}
        return None

    providers.register_provider(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1", classify_api_error=classify,
                                                refresh_credential=lambda entry: None))
    providers.register_provider(ProviderProfile(name="example-nohook", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1"))
    yield
    for name in ("example-oauth", "example-nohook"):
        providers._REGISTRY.pop(name, None)
    providers._PROVIDER_LIST_CACHE = None


def test_profile_hook_reclassifies_only_its_own_provider(plugin_profiles):
    exc = _error(403, "quota_exhausted")
    mine = classify_api_error(exc, provider="example-oauth", model="m")
    assert (mine.reason, mine.should_rotate_credential, mine.retryable) == (FailoverReason.billing, True, False)
    # Declining (None) and a profile without the hook both keep the built-in verdict.
    assert classify_api_error(_error(403, "other"), provider="example-oauth", model="m").reason == FailoverReason.auth
    assert classify_api_error(exc, provider="example-nohook", model="m").reason == FailoverReason.auth
    assert classify_api_error(exc, provider="anthropic", model="m").reason == FailoverReason.auth


def test_plugin_refresh_returning_none_benches_instead_of_phantom_success(plugin_profiles, monkeypatch):
    entry = PooledCredential(provider="example-oauth", id="abc123", label="acme", auth_type=AUTH_TYPE_OAUTH, priority=0,
                             source="manual:example_device", access_token="tok-1", refresh_token="rt-1", extra={})
    pool = CredentialPool("example-oauth", [entry])
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)
    assert pool._refresh_entry_impl(entry, force=True) is None
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED


def test_profile_hook_should_fallback_is_non_retryable_by_default(monkeypatch):
    """The fallback walk only runs for non-retryable verdicts outside the retryable-client reasons
    (the built-in terminal verdicts pin ``retryable=False``). A hook asking for fallback on such a
    reason gets the built-in default instead of a retry against the dead route; a rate-limit hook
    keeps the built-in retry-then-fallback shape."""
    def classify(error, *, status_code, error_code, message, body, model):
        return {"reason": "billing", "should_fallback": True}

    providers.register_provider(ProviderProfile(name="example-fallback", auth_type="oauth_external",
                                                base_url="https://example.invalid/v1", classify_api_error=classify))
    try:
        verdict = classify_api_error(_error(403, "quota_exhausted"), provider="example-fallback", model="m")
    finally:
        providers._REGISTRY.pop("example-fallback", None)
        providers._PROVIDER_LIST_CACHE = None
    assert (verdict.reason, verdict.should_fallback, verdict.retryable) == (FailoverReason.billing, True, False)

    providers.register_provider(ProviderProfile(
        name="example-ratelimit", auth_type="oauth_external", base_url="https://example.invalid/v1",
        classify_api_error=lambda error, **kw: {"reason": "rate_limit", "should_fallback": True}))
    try:
        limited = classify_api_error(_error(429, "slow_down"), provider="example-ratelimit", model="m")
    finally:
        providers._REGISTRY.pop("example-ratelimit", None)
        providers._PROVIDER_LIST_CACHE = None
    assert (limited.reason, limited.should_fallback, limited.retryable) == (FailoverReason.rate_limit, True, True)
