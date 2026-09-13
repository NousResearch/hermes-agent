"""Regression tests for issue #17335.

The ``quiet_mode=True`` fast path in :func:`model_tools.get_tool_definitions`
memoizes results to avoid re-walking the registry on every Gateway call. The
cached object must NOT be aliased into callers' return values \u2014 long-lived
Gateway processes mutate the returned list (``run_agent`` appends memory and
LCM context-engine tool schemas to ``self.tools``), and a shared list would
poison subsequent agent inits with duplicate tool names. Providers that
enforce uniqueness (DeepSeek, Xiaomi MiMo, Moonshot/Kimi) then reject the
API call with HTTP 400.

These tests pin:
- the cache-hit path returns a fresh list (existing #17098 behavior)
- the first uncached call also returns a fresh list (the fix)
- every call returns a list that is not the cached one, even after mutation
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import model_tools


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with an empty quiet_mode cache."""
    model_tools._tool_defs_cache.clear()
    yield
    model_tools._tool_defs_cache.clear()


class TestQuietModeCacheIsolation:

    def test_first_uncached_call_returns_fresh_list(self):
        """The first quiet_mode call must not alias the cached object \u2014
        otherwise a caller mutating the returned list mutates the cache."""
        first = model_tools.get_tool_definitions(quiet_mode=True)
        assert isinstance(first, list)
        # Find the cached value to compare identity.
        assert len(model_tools._tool_defs_cache) == 1
        cached = next(iter(model_tools._tool_defs_cache.values()))
        assert first is not cached, (
            "issue #17335: first quiet_mode call returned the cached list "
            "by reference \u2014 mutations will leak into subsequent calls."
        )

    def test_cache_hit_returns_fresh_list(self):
        """The cache-hit path already returned a copy pre-fix; pin it."""
        first = model_tools.get_tool_definitions(quiet_mode=True)
        second = model_tools.get_tool_definitions(quiet_mode=True)
        assert first is not second
        cached = next(iter(model_tools._tool_defs_cache.values()))
        assert second is not cached



    def test_cache_bounded_by_eviction(self):
        """The cache evicts the oldest entry when it reaches the cap,
        keeping the cache bounded instead of growing unbounded over a
        long-lived Gateway's lifetime (#19251)."""
        cap = model_tools._TOOL_DEFS_CACHE_MAX
        # Fill cache to the cap with distinct keys by varying enabled_toolsets.
        for i in range(cap):
            model_tools.get_tool_definitions(
                enabled_toolsets=[f"fake_toolset_{i}"], quiet_mode=True,
            )
        assert len(model_tools._tool_defs_cache) == cap

        # Adding one more must evict the oldest, not clear everything and
        # not grow past the cap.
        model_tools.get_tool_definitions(
            enabled_toolsets=["fake_toolset_overflow"], quiet_mode=True,
        )
        assert len(model_tools._tool_defs_cache) == cap, (
            "Eviction should keep the cache at the cap, not clear it or grow"
        )

    def test_non_quiet_mode_does_not_use_cache(self):
        """Sanity: quiet_mode=False (TUI path) skips the cache entirely \u2014
        explains why the bug only hit Gateway."""
        model_tools.get_tool_definitions(quiet_mode=False)
        assert len(model_tools._tool_defs_cache) == 0

    def test_concurrent_capacity_misses_evict_atomically(self, monkeypatch):
        """Two profile/toolset misses at capacity cannot race on eviction."""
        barrier = Barrier(2)

        def compute(*args, **kwargs):
            barrier.wait(timeout=2)
            return []

        monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute)
        for index in range(model_tools._TOOL_DEFS_CACHE_MAX):
            model_tools._tool_defs_cache[("old", index)] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    model_tools.get_tool_definitions,
                    enabled_toolsets=[f"concurrent_{index}"],
                    quiet_mode=True,
                )
                for index in range(2)
            ]
            assert [future.result(timeout=2) for future in futures] == [[], []]

        assert len(model_tools._tool_defs_cache) == model_tools._TOOL_DEFS_CACHE_MAX


class TestCronSessionCacheDimension:
    """Regression tests: the quiet_mode cache key must vary with cron-session
    state (HERMES_CRON_SESSION, a per-task ContextVar set only while a cron
    job's own agent turn is running), not just toolset/config fingerprint.

    Without this, the FIRST quiet_mode call for a given toolset fingerprint
    (often a non-cron probe, since a long-lived gateway serves many session
    types) permanently caches its check_fn-filtered tool list for every LATER
    call with the same fingerprint — including genuinely cron-scoped calls
    where HERMES_CRON_SESSION is bound and should unlock cronjob_manage.
    Confirmed via a live deployment: cronjob_manage silently unavailable to
    every cron job for the gateway process's entire lifetime.
    """

    def test_cron_session_and_non_cron_session_get_distinct_cache_entries(self):
        from gateway.session_context import _VAR_MAP

        var = _VAR_MAP["HERMES_CRON_SESSION"]

        # Call 1: no cron session bound (e.g. an interactive/gateway probe).
        model_tools.get_tool_definitions(enabled_toolsets=["cronjob"], quiet_mode=True)
        assert len(model_tools._tool_defs_cache) == 1

        # Call 2: identical toolset args, but HERMES_CRON_SESSION is now bound.
        token = var.set("1")
        try:
            model_tools.get_tool_definitions(enabled_toolsets=["cronjob"], quiet_mode=True)
        finally:
            var.reset(token)

        assert len(model_tools._tool_defs_cache) == 2, (
            "a cron-session call reused a non-cron-session cache entry for the "
            "same toolset fingerprint — cronjob_manage-style check_fns that key "
            "off HERMES_CRON_SESSION will silently vanish for every cron job "
            "after the first tool-list computation of the gateway's lifetime."
        )
