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

    def test_caller_mutation_does_not_poison_cache(self):
        """Simulate run_agent appending LCM tool schemas to the returned
        list. A second call must NOT see those appended entries."""
        first = model_tools.get_tool_definitions(quiet_mode=True)
        baseline_len = len(first)
        # Caller mutates the returned list (this is what run_agent does
        # when it injects memory + context-engine tool schemas).
        first.append({"type": "function", "function": {"name": "lcm_grep"}})
        first.append({"type": "function", "function": {"name": "lcm_expand"}})

        second = model_tools.get_tool_definitions(quiet_mode=True)
        # Length must match the original \u2014 cache pollution would make
        # second 2 entries longer.
        assert len(second) == baseline_len, (
            f"issue #17335: cache was polluted by caller mutation. "
            f"first len={baseline_len}, mutated len={len(first)}, "
            f"second-call len={len(second)} \u2014 expected {baseline_len}."
        )
        names = [t.get("function", {}).get("name") for t in second]
        assert "lcm_grep" not in names
        assert "lcm_expand" not in names

    def test_repeated_caller_mutation_does_not_accumulate(self):
        """The original Gateway symptom: every agent init in a long-lived
        process appends LCM schemas, accumulating duplicates over time."""
        baseline = len(model_tools.get_tool_definitions(quiet_mode=True))
        for _ in range(5):
            tools = model_tools.get_tool_definitions(quiet_mode=True)
            tools.append({"type": "function", "function": {"name": "lcm_grep"}})
        final = model_tools.get_tool_definitions(quiet_mode=True)
        assert len(final) == baseline, (
            f"Cache accumulated mutations across {5} agent inits: "
            f"baseline={baseline}, final={len(final)}."
        )

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

    def test_availability_invalidation_refreshes_filtered_schema_cache(self):
        """Explicit invalidation must refresh an already-filtered cache entry."""
        from tools.registry import invalidate_check_fn_cache, registry

        name = "test_availability_invalidation"
        available = {"value": False}

        def check():
            return available["value"]

        registry.register(
            name=name,
            toolset=name,
            schema={"description": "availability probe", "parameters": {"type": "object"}},
            handler=lambda _args: "{}",
            check_fn=check,
        )
        try:
            invalidate_check_fn_cache()
            missing = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name not in {tool["function"]["name"] for tool in missing}

            available["value"] = True
            invalidate_check_fn_cache()
            refreshed = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name in {tool["function"]["name"] for tool in refreshed}
        finally:
            registry.deregister(name)
            invalidate_check_fn_cache()

    def test_availability_cache_isolated_by_active_profile(self, tmp_path):
        """A filtered schema from one profile cannot leak into another turn."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.registry import invalidate_check_fn_cache, registry

        name = "test_profile_availability"
        first_home = tmp_path / "first"
        second_home = tmp_path / "second"
        first_home.mkdir()
        second_home.mkdir()

        def check():
            from hermes_constants import get_hermes_home

            return get_hermes_home() == first_home

        registry.register(
            name=name,
            toolset=name,
            schema={"description": "profile availability probe", "parameters": {"type": "object"}},
            handler=lambda _args: "{}",
            check_fn=check,
        )
        try:
            invalidate_check_fn_cache()
            first_token = set_hermes_home_override(first_home)
            try:
                visible = model_tools.get_tool_definitions(
                    enabled_toolsets=[name], quiet_mode=True,
                )
                assert name in {tool["function"]["name"] for tool in visible}
            finally:
                reset_hermes_home_override(first_token)

            second_token = set_hermes_home_override(second_home)
            try:
                hidden = model_tools.get_tool_definitions(
                    enabled_toolsets=[name], quiet_mode=True,
                )
                assert name not in {tool["function"]["name"] for tool in hidden}
            finally:
                reset_hermes_home_override(second_token)
        finally:
            registry.deregister(name)
            invalidate_check_fn_cache()

    def test_availability_cache_refreshes_with_check_fn_ttl(self, monkeypatch):
        """The outer filtered-schema cache cannot outlive check_fn freshness."""
        import tools.registry as registry_module
        from tools.registry import invalidate_check_fn_cache, registry

        name = "test_availability_ttl"
        available = {"value": False}
        now = {"value": 30.1}

        def check():
            return available["value"]

        monkeypatch.setattr(registry_module.time, "monotonic", lambda: now["value"])
        registry.register(
            name=name,
            toolset=name,
            schema={"description": "TTL availability probe", "parameters": {"type": "object"}},
            handler=lambda _args: "{}",
            check_fn=check,
        )
        try:
            invalidate_check_fn_cache()
            missing = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name not in {tool["function"]["name"] for tool in missing}

            now["value"] = 60.0
            assert name not in {
                tool["function"]["name"]
                for tool in model_tools.get_tool_definitions(
                    enabled_toolsets=[name], quiet_mode=True,
                )
            }

            available["value"] = True
            now["value"] = 60.2
            refreshed = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name in {tool["function"]["name"] for tool in refreshed}
        finally:
            registry.deregister(name)
            invalidate_check_fn_cache()

    def test_availability_cache_expires_last_good_after_grace(self, monkeypatch):
        """The outer cache cannot advertise a failed check past its grace."""
        import tools.registry as registry_module
        from tools.registry import invalidate_check_fn_cache, registry

        name = "test_availability_grace"
        available = {"value": True}
        now = {"value": 0.1}

        def check():
            return available["value"]

        monkeypatch.setattr(registry_module.time, "monotonic", lambda: now["value"])
        registry.register(
            name=name,
            toolset=name,
            schema={"description": "grace availability probe", "parameters": {"type": "object"}},
            handler=lambda _args: "{}",
            check_fn=check,
        )
        try:
            invalidate_check_fn_cache()
            visible = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name in {tool["function"]["name"] for tool in visible}

            available["value"] = False
            now["value"] = 60.0
            still_visible = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name in {tool["function"]["name"] for tool in still_visible}

            now["value"] = 60.2
            hidden = model_tools.get_tool_definitions(
                enabled_toolsets=[name], quiet_mode=True,
            )
            assert name not in {tool["function"]["name"] for tool in hidden}
        finally:
            registry.deregister(name)
            invalidate_check_fn_cache()
