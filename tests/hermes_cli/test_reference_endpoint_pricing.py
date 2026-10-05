"""Contract tests for the pricing fallbacks in ``models_pricing.get_pricing_for_provider``:

- a ``custom:<canonical>`` slug whose endpoint is the canonical upstream keeps the canonical
  fetcher; an unproven one (no endpoint, or a non-official host) stays unpriced;
- a configured ``providers.<slug>`` endpoint with no fetcher of its own resolves through the
  reference catalog (OpenRouter live rates + models.dev index), and ONLY when it is configured;
- ``cached_only`` serves the reference set the prewarm cached without any provider I/O.
"""

import time

import pytest

import hermes_cli.models_pricing as mp


@pytest.fixture
def openrouter_fetch(monkeypatch):
    """Fake OpenRouter fetcher + empty models.dev index; returns the fetch call list."""
    calls = []

    def fake_fetch(*, force_refresh=False):
        calls.append(force_refresh)
        return {"vendor/model": {"prompt": "0.000001", "completion": "0.000002"}}

    monkeypatch.setattr(mp, "_fetch_openrouter_pricing", fake_fetch)
    monkeypatch.setattr(mp, "_models_dev_cost_index_cache", {})
    monkeypatch.setattr(mp, "_pricing_cache", {})
    monkeypatch.setattr(mp, "_pricing_cache_retry_after", {})
    return calls


def _config_with(monkeypatch, providers):
    from hermes_cli import config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {"providers": providers})


def test_custom_prefix_with_canonical_endpoint_keeps_fetcher(monkeypatch):
    """``custom:openrouter`` pointed at the official openrouter.ai endpoint reaches the
    canonical pricing fetcher, not {}."""
    fetched = []
    monkeypatch.setitem(
        mp._PRICING_FETCHERS, "openrouter",
        lambda *, force_refresh=False: fetched.append(force_refresh)
        or {"a/b": {"prompt": "1", "completion": "2"}},
    )
    pricing = mp.get_pricing_for_provider(
        "custom:openrouter", base_url="https://openrouter.ai/api"
    )
    assert pricing == {"a/b": {"prompt": "1", "completion": "2"}}
    assert fetched == [False]


def test_unproven_custom_canonical_slug_stays_unpriced(monkeypatch, openrouter_fetch):
    """The trust boundary refuses a ``custom:openrouter`` row with no (or a non-official)
    endpoint: the canonical fetcher, reseller rates, and the cold-catalog index all stay dark
    instead of mislabeling the endpoint's models."""
    def no_fetcher(**_kwargs):
        raise AssertionError("canonical fetcher reached despite unproven custom slug")

    monkeypatch.setitem(mp._PRICING_FETCHERS, "openrouter", no_fetcher)
    assert mp.get_pricing_for_provider("custom:openrouter") == {}
    assert mp.get_pricing_for_provider("custom:openrouter", cached_only=True) == {}
    assert openrouter_fetch == []


def test_custom_proxy_host_is_refused(monkeypatch):
    """A lookalike host must not pass the trust boundary: exact official hostname only."""
    assert mp.resolve_pricing_provider("custom:openrouter", base_url="https://openrouter.ai.attacker.invalid/api") == ""
    assert mp.resolve_pricing_provider("custom:openrouter", base_url="https://api.example/v1") == ""
    assert mp.resolve_pricing_provider("custom:openrouter", base_url="") == ""
    assert (
        mp.resolve_pricing_provider("custom:openrouter", base_url="https://openrouter.ai/api")
        == "openrouter"
    )


def test_configured_endpoint_resolves_reference_catalog(monkeypatch, openrouter_fetch):
    """A ``providers.<slug>`` row without a fetcher prices through the reference catalog."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    pricing = mp.get_pricing_for_provider("tokenrouter")
    assert pricing == {"vendor/model": {"prompt": "0.000001", "completion": "0.000002"}}
    assert openrouter_fetch == [False]


def test_unconfigured_slug_gets_no_reference_prices(monkeypatch, openrouter_fetch):
    """A provider the user did not configure as an endpoint is never priced at reseller rates."""
    _config_with(monkeypatch, {})
    assert mp.get_pricing_for_provider("some-provider") == {}
    assert openrouter_fetch == []


def test_cached_only_serves_cached_reference_without_fetch(monkeypatch):
    """The picker's cold path reads the prewarm's reference set without dialing."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    cached = {"vendor/model": {"prompt": "1", "completion": "2"}}
    key = mp._reference_cache_key("tokenrouter")
    monkeypatch.setattr(mp, "_pricing_cache", {key: cached})
    monkeypatch.setattr(mp, "_pricing_cache_retry_after", {})

    def no_fetch(**_kwargs):
        raise AssertionError("provider I/O started on a cached_only read")

    monkeypatch.setattr(mp, "_fetch_openrouter_pricing", no_fetch)
    assert mp.get_pricing_for_provider("tokenrouter", cached_only=True) == cached


def test_reference_result_is_cached_for_later_cached_only_reads(monkeypatch, openrouter_fetch):
    """One fetch populates the cache the next (cached_only) render reads."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    first = mp.get_pricing_for_provider("tokenrouter")
    second = mp.get_pricing_for_provider("tokenrouter", cached_only=True)
    assert first == second
    assert openrouter_fetch == [False]  # second read did not fetch again


def test_cached_only_finds_authenticated_catalog_entries(monkeypatch):
    """A credentialed fetch caches under ``root␀auth:<fp>`` — the desktop picker's cached_only
    read must find it, or every second-open renders blank after a successful prewarm."""
    root = "https://inference-api.nousresearch.com"
    authed = {"vendor/model": {"prompt": "1", "completion": "2"}}
    monkeypatch.setattr(mp, "_pricing_cache", {root + "\x00auth:deadbeef": authed})
    monkeypatch.setattr(mp, "_pricing_cache_retry_after", {})
    monkeypatch.setattr(mp, "_pricing_provider_cache_keys", {})

    def no_fetch(**_kwargs):
        raise AssertionError("provider I/O started on a cached_only read")

    monkeypatch.setattr(mp, "fetch_models_with_pricing", no_fetch)
    from hermes_cli.models import _pricing_profile_key
    mp._pricing_provider_cache_keys[(_pricing_profile_key(), "nous")] = root
    assert mp.get_pricing_for_provider("nous", cached_only=True) == authed


def _seed_profile(monkeypatch, key):
    from hermes_cli import models as models_mod
    monkeypatch.setattr(models_mod, "hermes_home_key", lambda: key, raising=False)
    import sys
    hc = sys.modules.get("hermes_constants")
    if hc is not None:
        monkeypatch.setattr(hc, "hermes_home_key", lambda: key, raising=False)


def test_reference_catalog_is_profile_scoped(monkeypatch, openrouter_fetch):
    """Two profiles with the same endpoint slug never share a reference entry: the cache key
    folds in the profile identity, so profile-b's read cannot answer from profile-a's fetch."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    _seed_profile(monkeypatch, "profile-a")
    first = mp.get_pricing_for_provider("tokenrouter")
    _seed_profile(monkeypatch, "profile-b")
    monkeypatch.setattr(mp, "_pricing_cache", {})
    monkeypatch.setattr(mp, "_pricing_cache_retry_after", {})
    second = mp.get_pricing_for_provider("tokenrouter")
    assert first == second  # same catalog, freshly fetched
    assert openrouter_fetch == [False, False]  # profile-b did NOT read profile-a's entry


def test_reference_catalog_is_credential_scoped(monkeypatch, openrouter_fetch):
    """Two OpenRouter credentials in one process never share a reference entry: the key folds
    in the credential fingerprint, so a second key's read refetches rather than inheriting the
    first key's org-scoped catalog."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key-a")
    first = mp.get_pricing_for_provider("tokenrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key-b")
    monkeypatch.setattr(mp, "_pricing_cache", {})
    monkeypatch.setattr(mp, "_pricing_cache_retry_after", {})
    second = mp.get_pricing_for_provider("tokenrouter")
    assert first == second
    assert openrouter_fetch == [False, False]  # key-b did NOT read key-a's entry


def test_reference_catalog_entry_expires(monkeypatch, openrouter_fetch):
    """A live reference catalog is cached for a bounded window, not the life of the process."""
    _config_with(monkeypatch, {"tokenrouter": {"base_url": "https://api.example/v1"}})
    mp.get_pricing_for_provider("tokenrouter")
    key = mp._reference_cache_key("tokenrouter")
    retry_after = mp._pricing_cache_retry_after[key]
    assert 0 < retry_after - time.monotonic() <= mp._REFERENCE_CATALOG_TTL_SECONDS


def test_loopback_endpoint_is_never_priced_at_reseller_rates(monkeypatch, openrouter_fetch):
    """A self-hosted loopback endpoint (free on the user's own GPU) must not render OpenRouter
    retail rates: _configured_endpoint_slug rejects it before the reference catalog is built."""
    _config_with(monkeypatch, {"selfhosted": {"base_url": "http://127.0.0.1:8000/v1"}})
    assert mp.get_pricing_for_provider("selfhosted") == {}
    assert openrouter_fetch == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "http://0.0.0.0:8000/v1",
        "http://127.255.255.254/v1",
    ],
)
def test_loopback_base_urls_are_rejected(base_url):
    assert mp._is_loopback_base_url(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example/v1",
        "https://openrouter.ai/api",
        "http://192.168.1.10:8000/v1",
        "http://tokenrouter.example.com/v1",
    ],
)
def test_non_loopback_base_urls_are_accepted(base_url):
    assert not mp._is_loopback_base_url(base_url)
