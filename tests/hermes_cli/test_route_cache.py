"""Opt-in short-TTL cache for config-derived GET payloads (`HERMES_ROUTE_CACHE=1`).

The polled dashboard routes (status ~1/s, profiles, model options) re-read config and re-walk
provider/profile inventories on every call, while their payloads change on a human timescale.
The cache is opt-in so the default install, the test suite and single-shot callers keep seeing
live values.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli.web_routers import _route_cache


@pytest.fixture
def enabled(monkeypatch):
    """Turn the cache on for one test and start with an empty store."""
    monkeypatch.setattr(_route_cache, "_ENABLED", True)
    _route_cache.clear()
    yield _route_cache
    _route_cache.clear()


def test_disabled_by_default_reads_live_values(monkeypatch):
    monkeypatch.setattr(_route_cache, "_ENABLED", False)
    _route_cache.clear()
    assert _route_cache.enabled() is False
    _route_cache.put(("status", "default"), {"a": 1}, ttl=60.0)
    assert _route_cache.get(("status", "default")) is None


def test_round_trip_within_ttl(enabled):
    enabled.put(("status", "default"), {"a": 1}, ttl=30.0)
    assert enabled.get(("status", "default")) == {"a": 1}


def test_entry_expires_after_its_own_ttl(enabled, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(enabled.time, "monotonic", lambda: clock["t"])
    enabled.put(("status", "default"), {"a": 1}, ttl=5.0)
    clock["t"] += 4.9
    assert enabled.get(("status", "default")) == {"a": 1}
    clock["t"] += 0.2
    assert enabled.get(("status", "default")) is None


def test_keys_do_not_alias(enabled):
    enabled.put(("status", "default"), {"profile": "default"}, ttl=30.0)
    enabled.put(("status", "work"), {"profile": "work"}, ttl=30.0)
    enabled.put(("model_options", "default"), {"models": []}, ttl=30.0)
    assert enabled.get(("status", "default")) == {"profile": "default"}
    assert enabled.get(("status", "work")) == {"profile": "work"}
    assert enabled.get(("model_options", "default")) == {"models": []}
    assert enabled.get(("status", "missing")) is None


@pytest.mark.parametrize("value,ttl", [(None, 30.0), ({"a": 1}, 0.0), ({"a": 1}, -1.0)])
def test_nothing_is_stored_for_an_unusable_entry(enabled, value, ttl):
    enabled.put(("status", "default"), value, ttl=ttl)
    assert enabled.get(("status", "default")) is None


def test_store_stays_bounded(enabled, monkeypatch):
    monkeypatch.setattr(enabled, "_MAX_ENTRIES", 4)
    for i in range(10):
        enabled.put(("status", f"p{i}"), {"i": i}, ttl=30.0)
    live = sum(1 for i in range(10) if enabled.get(("status", f"p{i}")) is not None)
    assert live <= 4, f"store kept {live} entries past its bound"


def test_clear_drops_every_entry(enabled):
    enabled.put(("status", "default"), {"a": 1}, ttl=30.0)
    enabled.clear()
    assert enabled.get(("status", "default")) is None
