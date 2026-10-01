"""Contract tests for the pricing fallbacks in ``models_pricing.get_pricing_for_provider``:

- a ``custom:<canonical>`` slug keeps the canonical provider's fetcher;
- a configured ``providers.<slug>`` endpoint with no fetcher of its own resolves through the
  reference catalog (OpenRouter live rates + models.dev index), and ONLY when it is configured;
- ``cached_only`` serves the reference set the prewarm cached without any provider I/O.
"""

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


def test_custom_prefix_keeps_canonical_fetcher(monkeypatch):
    """``custom:openrouter`` must reach the openrouter pricing fetcher, not return {}."""
    fetched = []
    monkeypatch.setitem(
        mp._PRICING_FETCHERS, "openrouter",
        lambda *, force_refresh=False: fetched.append(force_refresh)
        or {"a/b": {"prompt": "1", "completion": "2"}},
    )
    pricing = mp.get_pricing_for_provider("custom:openrouter")
    assert pricing == {"a/b": {"prompt": "1", "completion": "2"}}
    assert fetched == [False]


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
    monkeypatch.setattr(mp, "_pricing_cache", {"reference:tokenrouter": cached})
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
