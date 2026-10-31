"""Tests for the catalog deadline in ``models_pricing._cached_catalog``.

A deadline on a non-empty catalog is a re-fetch trigger, not a correctness
requirement. Past it the entry must still be served — a ``cached_only`` read performs
no provider I/O by design, so evicting the entry leaves the picker with no price and no
Free badge until something performs a live fetch — and the re-fetch happens behind the
read instead.
"""

from __future__ import annotations

from threading import Event
from time import monotonic

import pytest

from hermes_cli import models_pricing
from hermes_cli.models import _pricing_profile_key

CACHE_KEY = "https://inference.example.test"
CATALOG = {"model-a": {"prompt": "1", "completion": "2"}}


@pytest.fixture
def catalog(monkeypatch):
    """One non-empty catalog, already past its deadline, with a recording fetcher."""
    fetched = Event()
    calls: list[bool] = []

    def _fetcher(*, force_refresh=False):
        calls.append(force_refresh)
        fetched.set()
        return {}

    monkeypatch.setattr(models_pricing, "_pricing_cache", {CACHE_KEY: dict(CATALOG)})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {CACHE_KEY: monotonic() - 1.0})
    monkeypatch.setattr(models_pricing, "_catalog_refresh_inflight", set())
    monkeypatch.setattr(
        models_pricing,
        "_pricing_provider_cache_keys",
        {(_pricing_profile_key(), "nous"): CACHE_KEY},
    )
    monkeypatch.setattr(models_pricing, "_PRICING_FETCHERS", {"nous": _fetcher})
    return calls, fetched


def test_expired_non_empty_catalog_is_still_served(catalog):
    assert models_pricing._cached_catalog(CACHE_KEY) == CATALOG
    assert CACHE_KEY in models_pricing._pricing_cache


def test_expired_catalog_starts_one_refresh_behind_the_read(catalog):
    calls, fetched = catalog

    models_pricing._cached_catalog(CACHE_KEY)

    assert fetched.wait(5.0), "no refresh was started for an expired catalog"
    assert calls == [True], "the refresh must force a fetch"
    # The deadline moves out, so a burst of reads cannot pile up refresh threads while the
    # endpoint is unreachable.
    assert models_pricing._pricing_cache_retry_after[CACHE_KEY] - monotonic() > 60
    for _ in range(4):
        models_pricing._cached_catalog(CACHE_KEY)
    assert calls == [True]


def test_fresh_catalog_is_served_without_a_refresh(catalog):
    calls, _ = catalog
    models_pricing._pricing_cache_retry_after[CACHE_KEY] = monotonic() + 300
    assert models_pricing._cached_catalog(CACHE_KEY) == CATALOG
    assert calls == []


def test_expired_empty_catalog_is_still_evicted(monkeypatch):
    """An empty catalog is a failed fetch, not stale data — it must not be served."""
    monkeypatch.setattr(models_pricing, "_pricing_cache", {CACHE_KEY: {}})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {CACHE_KEY: monotonic() - 1.0})
    monkeypatch.setattr(models_pricing, "_catalog_refresh_inflight", set())

    assert models_pricing._cached_catalog(CACHE_KEY) is None
    assert CACHE_KEY not in models_pricing._pricing_cache


def test_refresh_is_not_scheduled_for_an_unattributable_key(catalog):
    calls, _ = catalog
    models_pricing._pricing_provider_cache_keys.clear()

    assert models_pricing._schedule_catalog_refresh(CACHE_KEY) is False
    assert calls == []
