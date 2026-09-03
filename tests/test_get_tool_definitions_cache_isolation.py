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
import os


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
        """The cache lock keeps concurrent capacity misses bounded."""
        barrier = Barrier(2)

        class UnlockedConfig:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def compute(*args, **kwargs):
            barrier.wait(timeout=2)
            return []

        # Isolate the cache's concurrency contract from config's RLock, which
        # intentionally serializes normal schema builds around config parsing.
        monkeypatch.setattr("hermes_cli.config._CONFIG_LOCK", UnlockedConfig())
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

    def test_same_metadata_rewrite_invalidates_dynamic_schema(
        self, tmp_path, monkeypatch
    ):
        config_path = tmp_path / "config.yaml"
        before = "delegation:\n  routes:\n    alpha: {}\n"
        after = "delegation:\n  routes:\n    bravo: {}\n"
        assert len(before) == len(after)
        config_path.write_text(before, encoding="utf-8")
        original = config_path.stat()

        monkeypatch.setattr(
            "hermes_cli.config.get_config_path", lambda: config_path
        )
        monkeypatch.setattr(
            model_tools, "check_fn_cache_scope", lambda: "cache-rewrite-test"
        )

        first = model_tools.get_tool_definitions(
            enabled_toolsets=["delegation"], quiet_mode=True
        )
        first_delegate = next(
            item["function"] for item in first if item["function"]["name"] == "delegate_task"
        )
        assert first_delegate["parameters"]["properties"]["route"]["enum"] == ["alpha"]

        config_path.write_text(after, encoding="utf-8")
        os.utime(config_path, ns=(original.st_atime_ns, original.st_mtime_ns))
        rewritten = config_path.stat()
        assert rewritten.st_size == original.st_size
        assert rewritten.st_mtime_ns == original.st_mtime_ns

        second = model_tools.get_tool_definitions(
            enabled_toolsets=["delegation"], quiet_mode=True
        )
        second_delegate = next(
            item["function"] for item in second if item["function"]["name"] == "delegate_task"
        )
        assert second_delegate["parameters"]["properties"]["route"]["enum"] == ["bravo"]

    def test_config_rewrite_during_schema_build_never_poisons_old_key(
        self, tmp_path, monkeypatch
    ):
        """Schema parsed after a rewrite must not be cached under the prior digest."""
        config_path = tmp_path / "config.yaml"
        alpha = "delegation:\n  routes:\n    alpha: {}\n"
        bravo = "delegation:\n  routes:\n    bravo: {}\n"
        assert len(alpha) == len(bravo)
        config_path.write_text(alpha, encoding="utf-8")
        original = config_path.stat()

        monkeypatch.setattr(
            "hermes_cli.config.get_config_path",
            lambda: config_path,
        )
        monkeypatch.setattr(
            model_tools,
            "check_fn_cache_scope",
            lambda: "cache-interleaving-test",
        )
        real_compute = model_tools._compute_tool_definitions
        compute_count = 0

        def rewrite_during_first_compute(*args, **kwargs):
            nonlocal compute_count
            compute_count += 1
            if compute_count == 1:
                config_path.write_text(bravo, encoding="utf-8")
                os.utime(
                    config_path,
                    ns=(original.st_atime_ns, original.st_mtime_ns),
                )
            return real_compute(*args, **kwargs)

        monkeypatch.setattr(
            model_tools,
            "_compute_tool_definitions",
            rewrite_during_first_compute,
        )

        first = model_tools.get_tool_definitions(
            enabled_toolsets=["delegation"], quiet_mode=True
        )
        first_delegate = next(
            item["function"]
            for item in first
            if item["function"]["name"] == "delegate_task"
        )
        assert first_delegate["parameters"]["properties"]["route"]["enum"] == [
            "bravo"
        ]
        assert compute_count == 2

        config_path.write_text(alpha, encoding="utf-8")
        os.utime(
            config_path,
            ns=(original.st_atime_ns, original.st_mtime_ns),
        )
        restored = model_tools.get_tool_definitions(
            enabled_toolsets=["delegation"], quiet_mode=True
        )
        restored_delegate = next(
            item["function"]
            for item in restored
            if item["function"]["name"] == "delegate_task"
        )
        assert restored_delegate["parameters"]["properties"]["route"]["enum"] == [
            "alpha"
        ]
        assert compute_count == 3

    def test_repeated_cache_hit_fingerprint_changes_return_last_complete_snapshot(
        self, monkeypatch
    ):
        snapshots = iter(("A", "B", "A", "B"))
        cached = [{"function": {"name": "cached_tool"}}]

        monkeypatch.setattr(
            model_tools,
            "_tool_defs_config_fingerprints",
            lambda: next(snapshots),
        )
        monkeypatch.setattr(
            model_tools,
            "check_fn_cache_scope",
            lambda: "cache-oscillation-test",
        )
        monkeypatch.setattr(
            model_tools,
            "_tool_defs_cache_key",
            lambda *_args: "cache-key",
        )
        monkeypatch.setattr(
            model_tools,
            "_compute_tool_definitions",
            lambda *_args, **_kwargs: pytest.fail("cached snapshots must not recompute"),
        )
        model_tools._tool_defs_cache["cache-key"] = cached

        result = model_tools.get_tool_definitions(quiet_mode=True)

        assert result == cached
        assert result is not cached

    def test_oscillation_fallback_refreshes_last_resolved_tool_names(
        self, monkeypatch
    ):
        """Continuously-changing-config fallback must keep _last_resolved_tool_names
        aligned with the cached definitions it returns, not leave them stale.

        The oscillation-fallback branch (attempt==1, cache hit, fingerprint still
        changing) is reached by pre-seeding the cache under the known key and
        arranging the fingerprint mock to always return a different value after
        the initial binding — mirroring the existing
        ``test_repeated_cache_hit_fingerprint_changes_return_last_complete_snapshot``
        scenario but adding the _last_resolved_tool_names assertion.
        """
        # Each ``with _CONFIG_LOCK`` block calls _tool_defs_config_fingerprints()
        # once for `before`.  The second call (after `if cached is not None:`)
        # checks whether the config changed while we held the result.
        # With the cache pre-populated, the sequence is:
        #   attempt 0: before=A  [hit]  check=B  → mismatch → attempt==0 → continue
        #   attempt 1: before=A  [hit]  check=B  → mismatch → attempt==1 → fallback
        snapshots = iter(("fp-A", "fp-B", "fp-A", "fp-B"))
        cached = [{"function": {"name": "expected_tool"}}]

        monkeypatch.setattr(
            model_tools,
            "_tool_defs_config_fingerprints",
            lambda: next(snapshots),
        )
        monkeypatch.setattr(
            model_tools,
            "check_fn_cache_scope",
            lambda: "oscillation-names-test",
        )
        monkeypatch.setattr(
            model_tools,
            "_tool_defs_cache_key",
            lambda *_args: "osc-names-key",
        )
        monkeypatch.setattr(
            model_tools,
            "_compute_tool_definitions",
            lambda *_args, **_kwargs: pytest.fail("pre-seeded cache must not recompute"),
        )
        model_tools._tool_defs_cache["osc-names-key"] = cached
        # Force a stale value so we can verify it gets overwritten.
        model_tools._last_resolved_tool_names = ["stale_tool"]

        result = model_tools.get_tool_definitions(quiet_mode=True)

        assert result == cached
        assert result is not cached
        assert model_tools._last_resolved_tool_names == ["expected_tool"], (
            "oscillation fallback must refresh _last_resolved_tool_names "
            "to match the returned cached definitions"
        )
