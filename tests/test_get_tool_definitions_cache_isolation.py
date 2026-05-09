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

    def test_non_quiet_mode_does_not_use_cache(self):
        """Sanity: quiet_mode=False (TUI path) skips the cache entirely —
        explains why the bug only hit Gateway."""
        model_tools.get_tool_definitions(quiet_mode=False)
        assert len(model_tools._tool_defs_cache) == 0


class TestGatewayAccessControlCacheIsolation:
    def _patch_access_config(self, monkeypatch):
        import hermes_cli.config as config_mod

        cfg = {
            "skills": {
                "access_control": {
                    "restricted_toolsets": ["terminal"],
                    "allowed_identities": {"whatsapp": ["owner@lid"]},
                }
            }
        }
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: cfg)

    def _tool_names(self):
        tools = model_tools.get_tool_definitions(
            enabled_toolsets=["terminal"],
            quiet_mode=True,
        )
        return {t["function"]["name"] for t in tools}

    def test_restricted_toolset_hidden_for_unauthorized_gateway_identity(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        self._patch_access_config(monkeypatch)
        tokens = set_session_vars(platform="whatsapp", chat_id="stranger@lid", user_id="stranger@lid")
        try:
            names = self._tool_names()
            assert "terminal" not in names
        finally:
            clear_session_vars(tokens)

    def test_restricted_toolset_visible_for_authorized_gateway_identity(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        self._patch_access_config(monkeypatch)
        tokens = set_session_vars(platform="whatsapp", chat_id="owner@lid", user_id="owner@lid")
        try:
            names = self._tool_names()
            assert "terminal" in names
        finally:
            clear_session_vars(tokens)

    def test_quiet_mode_cache_is_scoped_by_gateway_identity(self, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        self._patch_access_config(monkeypatch)
        first = set_session_vars(platform="whatsapp", chat_id="owner@lid", user_id="owner@lid")
        try:
            assert "terminal" in self._tool_names()
        finally:
            clear_session_vars(first)

        second = set_session_vars(platform="whatsapp", chat_id="stranger@lid", user_id="stranger@lid")
        try:
            assert "terminal" not in self._tool_names()
        finally:
            clear_session_vars(second)
