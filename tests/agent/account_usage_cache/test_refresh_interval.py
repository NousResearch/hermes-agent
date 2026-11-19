"""The account-usage cache's refresh cadence is the caller's to choose.

``REFRESH_AFTER_S`` (two minutes) is a model picker's cadence. The Desktop status bar shows the
same window where the user watches it move, so it asks once a minute instead — same throttle, one
parameter, still one inflight fetch per provider.
"""
from __future__ import annotations

from agent import account_usage_cache as cache


def _key() -> tuple[str, str, str, str]:
    return ("test-home", "opencode-go", "", "")


def test_the_interval_parameter_overrides_the_default(monkeypatch):
    monkeypatch.setattr(cache, "_last_try", {})
    monkeypatch.setattr(cache, "_inflight", set())

    key = _key()
    assert cache._claim_refresh(key, 1_000.0) is True  # the first try claims the slot
    cache._release_refresh(key)  # the fetch finished (a request long gone by now)
    assert cache._claim_refresh(key, 1_030.0) is False  # inside the two-minute default
    assert cache._claim_refresh(key, 1_030.0, 20.0) is True  # 30 s later, a 20 s cadence allows it


def test_an_inflight_refresh_is_never_duplicated(monkeypatch):
    monkeypatch.setattr(cache, "_last_try", {})
    monkeypatch.setattr(cache, "_inflight", {_key()})

    assert cache._claim_refresh(_key(), 10_000.0, 0.0) is False


def test_refresh_account_usage_async_passes_the_interval_through(monkeypatch):
    """The cadence reaches the claim, and a refused claim starts no thread."""
    seen: list[float] = []

    def _claim(key, now, interval):
        seen.append(interval)
        return False

    monkeypatch.setattr(cache, "_claim_refresh", _claim)

    assert cache.refresh_account_usage_async(["opencode-go"], interval=60.0) == []
    assert seen == [60.0]
