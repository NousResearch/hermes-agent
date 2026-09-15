"""``_pricing_cache`` keys on the credential, not just the base URL.

Nous ``/v1/models`` answers each caller with the catalog their org may reach,
so an anonymous read, and two different tokens, must not share a cache entry.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import hermes_cli.models as models_mod
from hermes_cli import models_pricing
from hermes_cli.models_pricing import fetch_models_with_pricing, peek_cached_pricing

BASE = "https://inference-api.example.com"

# What the endpoint serves anonymously vs. to a policy-restricted caller.
_FULL = ["vendor/allowed", "vendor/blocked"]
_FILTERED = ["vendor/allowed"]


@pytest.fixture(autouse=True)
def _clear_pricing_cache():
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    yield
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()


@pytest.fixture
def catalog(monkeypatch):
    """Serve the filtered catalog to an authenticated read, the full one to an
    anonymous read, and record every request."""
    requests: list[str | None] = []

    def _fake_urlopen(req, timeout=8.0):
        auth = req.get_header("Authorization")
        requests.append(auth)
        ids = _FILTERED if auth else _FULL
        payload = {
            "data": [
                {"id": mid, "pricing": {"prompt": "0.000002", "completion": "0.00001"}}
                for mid in ids
            ]
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: False
        return resp

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _fake_urlopen)
    return requests


@pytest.fixture
def per_org_catalog(monkeypatch):
    """Serve each token the catalog its own org may reach."""
    requests: list[str | None] = []

    def _fake_urlopen(req, timeout=8.0):
        auth = req.get_header("Authorization")
        requests.append(auth)
        org = "a" if auth == "Bearer tok-a" else "b"
        payload = {
            "data": [
                {
                    "id": f"org-{org}/only",
                    "pricing": {"prompt": "0.000002", "completion": "0.00001"},
                }
            ]
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: False
        return resp

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _fake_urlopen)
    return requests


def test_one_token_does_not_receive_another_tokens_catalog(per_org_catalog):
    """Two orgs in one process — a long-lived gateway or desktop backend after
    a profile switch or re-login."""
    a = fetch_models_with_pricing(api_key="tok-a", base_url=BASE)
    b = fetch_models_with_pricing(api_key="tok-b", base_url=BASE)

    assert list(a) == ["org-a/only"]
    assert list(b) == ["org-b/only"], "token B was handed token A's catalog"
    assert len(per_org_catalog) == 2, "token B must reach the network"


def test_credential_value_does_not_appear_in_the_cache_key():
    """Guards against keying on the raw token."""
    assert "sk-super-secret" not in models_pricing._pricing_auth_fingerprint("sk-super-secret")


def test_anonymous_and_authenticated_reads_are_separate(catalog):
    """Also pins the header: anonymous must send none."""
    anon = fetch_models_with_pricing(api_key="", base_url=BASE)
    authed = fetch_models_with_pricing(api_key="sk-test", base_url=BASE)

    assert sorted(anon) == sorted(_FULL)
    assert sorted(authed) == sorted(_FILTERED)
    assert catalog == [None, "Bearer sk-test"]


@pytest.mark.parametrize("api_key", ["sk-test", ""])
def test_repeated_read_still_hits_the_cache(catalog, api_key):
    """Widening the key must not cost the caching it was there for."""
    first = fetch_models_with_pricing(api_key=api_key, base_url=BASE)
    second = fetch_models_with_pricing(api_key=api_key, base_url=BASE)

    assert first == second
    assert len(catalog) == 1, "second read should be served from cache"


def test_force_refresh_replaces_only_its_own_entry(catalog):
    """A forced authenticated re-read must leave the anonymous entry intact."""
    fetch_models_with_pricing(api_key="", base_url=BASE)
    fetch_models_with_pricing(api_key="sk-test", base_url=BASE)
    fetch_models_with_pricing(api_key="sk-test", base_url=BASE, force_refresh=True)

    assert len(catalog) == 3
    anon = fetch_models_with_pricing(api_key="", base_url=BASE)
    assert sorted(anon) == sorted(_FULL)
    assert len(catalog) == 3, "the anonymous entry should have survived"


@pytest.mark.parametrize("provider", ["nous", "openrouter"])
@pytest.mark.parametrize("api_key", ["sk-test", ""])
def test_provider_cached_only_reads_its_fetched_catalog(monkeypatch, catalog, provider, api_key):
    monkeypatch.setattr(models_pricing, "_pricing_provider_cache_keys", {})
    monkeypatch.setattr(models_pricing, "_resolve_openrouter_api_key", lambda: api_key)
    monkeypatch.setattr(models_pricing, "_resolve_nous_pricing_credentials", lambda: (api_key, BASE))
    fetched = models_pricing.get_pricing_for_provider(provider)
    assert sorted(fetched) == sorted(_FILTERED if api_key else _FULL)
    if provider == "nous":
        monkeypatch.setattr("hermes_cli.auth._nous_inference_env_override", lambda: None)
        monkeypatch.setattr(models_pricing, "get_cached_nous_inference_base_url", lambda: "")
        assert models_pricing.pricing_cache_scope(provider) == BASE

    def unexpected_auth():
        pytest.fail("cached-only reads must not resolve or refresh credentials")

    monkeypatch.setattr(models_pricing, "_resolve_openrouter_api_key", unexpected_auth)
    monkeypatch.setattr(models_pricing, "_resolve_nous_pricing_credentials", unexpected_auth)
    assert models_pricing.get_pricing_for_provider(provider, cached_only=True) == fetched
    assert len(catalog) == 1
    if provider == "nous":
        now = models_pricing.time.monotonic()
        monkeypatch.setattr(models_pricing.time, "monotonic", lambda: now + 301)
        assert models_pricing.get_pricing_for_provider(provider, cached_only=True) == {}
        assert len(catalog) == 1


@pytest.mark.parametrize("provider", ["nous", "openrouter"])
def test_provider_cached_only_keeps_profile_ownership(monkeypatch, tmp_path, per_org_catalog, provider):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(models_pricing, "_pricing_provider_cache_keys", {})
    expected = {}
    for profile in ("a", "b"):
        token = set_hermes_home_override(tmp_path / profile)
        try:
            monkeypatch.setattr(models_pricing, "_resolve_openrouter_api_key", lambda: f"tok-{profile}")
            monkeypatch.setattr(models_pricing, "_resolve_nous_pricing_credentials", lambda: (f"tok-{profile}", BASE))
            expected[profile] = models_pricing.get_pricing_for_provider(provider)
        finally:
            reset_hermes_home_override(token)
    assert expected["a"] != expected["b"]
    for profile in ("a", "b"):
        token = set_hermes_home_override(tmp_path / profile)
        try:
            assert models_pricing.get_pricing_for_provider(provider, cached_only=True) == expected[profile]
        finally:
            reset_hermes_home_override(token)
    assert len(per_org_catalog) == 2


class TestPeekCachedPricing:
    def test_returns_empty_when_nothing_cached(self):
        assert peek_cached_pricing(BASE) == {}

    def test_accepts_a_v1_suffixed_url(self, catalog):
        """The agent holds a /v1-suffixed base URL; fetchers key on the root."""
        fetch_models_with_pricing(api_key="sk-test", base_url=BASE)
        assert sorted(peek_cached_pricing(BASE + "/v1")) == sorted(_FILTERED)

    def test_prefers_the_authenticated_catalog(self, catalog):
        fetch_models_with_pricing(api_key="", base_url=BASE)
        fetch_models_with_pricing(api_key="sk-test", base_url=BASE)
        assert sorted(peek_cached_pricing(BASE)) == sorted(_FILTERED)

    def test_falls_back_to_the_anonymous_catalog(self, catalog):
        fetch_models_with_pricing(api_key="", base_url=BASE)
        assert sorted(peek_cached_pricing(BASE)) == sorted(_FULL)

    def test_never_fetches(self, catalog):
        peek_cached_pricing(BASE)
        assert catalog == []


class TestNousCatalogExpiry:
    """A Nous catalog reflects the org's policy, which an admin can change while
    a long-lived process holds the entry."""

    def test_entry_expires_so_a_policy_change_is_picked_up(self, catalog, monkeypatch):
        from hermes_cli.models_pricing import _NOUS_CATALOG_TTL_SECONDS

        fetch_models_with_pricing(
            api_key="sk-test", base_url=BASE,
            cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
        )
        assert len(catalog) == 1

        now = models_mod.time.monotonic()
        monkeypatch.setattr(
            models_mod.time, "monotonic",
            lambda: now + _NOUS_CATALOG_TTL_SECONDS + 1,
        )
        fetch_models_with_pricing(
            api_key="sk-test", base_url=BASE,
            cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
        )
        assert len(catalog) == 2, "expired entry should be re-read"

    def test_no_ttl_keeps_the_entry_indefinitely(self, catalog, monkeypatch):
        """Other providers' catalogs carry no policy and must not start
        re-fetching."""
        fetch_models_with_pricing(api_key="sk-test", base_url=BASE)
        now = models_mod.time.monotonic()
        monkeypatch.setattr(models_mod.time, "monotonic", lambda: now + 86_400)
        fetch_models_with_pricing(api_key="sk-test", base_url=BASE)
        assert len(catalog) == 1

    def test_peek_prefers_the_newest_credential(self, per_org_catalog):
        """After a rotation the older entry is still resident and, being
        insertion-ordered, comes first."""
        fetch_models_with_pricing(api_key="tok-a", base_url=BASE, cache_ttl_seconds=300)
        fetch_models_with_pricing(api_key="tok-b", base_url=BASE, cache_ttl_seconds=300)
        assert list(peek_cached_pricing(BASE)) == ["org-b/only"]

    def test_peek_skips_an_expired_entry(self, catalog, monkeypatch):
        """Reading _pricing_cache directly walked straight past the TTL."""
        from hermes_cli.models_pricing import _NOUS_CATALOG_TTL_SECONDS

        fetch_models_with_pricing(
            api_key="sk-test", base_url=BASE,
            cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
        )
        now = models_mod.time.monotonic()
        monkeypatch.setattr(
            models_mod.time, "monotonic",
            lambda: now + _NOUS_CATALOG_TTL_SECONDS + 1,
        )
        assert peek_cached_pricing(BASE) == {}
