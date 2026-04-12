"""Tests for tool result memoization (caching of idempotent tool results).

Tools can opt in to result caching by setting ``can_memoize=True`` at
registration.  When enabled, ``handle_function_call()`` caches the dispatch
result keyed by ``(tool_name, args_hash)`` and returns the cached value on
subsequent calls with identical arguments — saving the cost of re-executing
idempotent handlers like file reads or web fetches.
"""

import json

import pytest

from model_tools import handle_function_call, clear_memo_cache
from tools.registry import ToolRegistry, ToolEntry


# ── Unit tests for ToolEntry.can_memoize field ─────────────────────────────


class TestToolEntryCanMemoize:
    """ToolEntry stores per-tool memoization opt-in flag."""

    def test_default_can_memoize_is_false(self):
        entry = ToolEntry(
            name="t", toolset="test", schema={}, handler=lambda: None,
            check_fn=None, requires_env=[], is_async=False,
            description="", emoji="", max_result_size_chars=None,
            can_memoize=False,
        )
        assert entry.can_memoize is False

    def test_can_memoize_set_true(self):
        entry = ToolEntry(
            name="t", toolset="test", schema={}, handler=lambda: None,
            check_fn=None, requires_env=[], is_async=False,
            description="", emoji="", max_result_size_chars=None,
            can_memoize=True,
        )
        assert entry.can_memoize is True


class TestRegistryRegisterCanMemoize:
    """registry.register() passes can_memoize to ToolEntry."""

    def test_register_with_can_memoize(self):
        reg = ToolRegistry()
        reg.register(
            name="_test_memo_tool",
            toolset="test",
            schema={"name": "_test_memo_tool", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
            can_memoize=True,
        )
        assert reg._tools["_test_memo_tool"].can_memoize is True

    def test_register_without_can_memoize_defaults_false(self):
        reg = ToolRegistry()
        reg.register(
            name="_test_no_memo",
            toolset="test",
            schema={"name": "_test_no_memo", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
        )
        assert reg._tools["_test_no_memo"].can_memoize is False


# ── Integration tests for memoization in handle_function_call ──────────────


class TestMemoCacheHitMiss:
    """Memoizable tools return cached results on repeat calls."""

    def test_memoizable_tool_returns_cached_result(self):
        """Second call with same args returns the cached result without
        re-invoking the handler."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_hit",
            toolset="test",
            schema={"name": "_test_memo_hit", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        # Patch the global registry
        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_hit"] = reg._tools["_test_memo_hit"]
        clear_memo_cache()

        try:
            r1 = handle_function_call("_test_memo_hit", {"x": 1})
            r2 = handle_function_call("_test_memo_hit", {"x": 1})
            # Both calls should return the same (first) result
            assert r1 == r2
            assert json.loads(r1)["call"] == 1
            # Handler should only have been called once
            assert call_count == 1
        finally:
            global_registry._tools.pop("_test_memo_hit", None)
            clear_memo_cache()

    def test_different_args_produces_cache_miss(self):
        """Different arguments bypass the cache and invoke the handler."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count, "x": args.get("x")})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_miss",
            toolset="test",
            schema={"name": "_test_memo_miss", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_miss"] = reg._tools["_test_memo_miss"]
        clear_memo_cache()

        try:
            r1 = handle_function_call("_test_memo_miss", {"x": 1})
            r2 = handle_function_call("_test_memo_miss", {"x": 2})
            # Different results — handler called twice
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
            assert call_count == 2
        finally:
            global_registry._tools.pop("_test_memo_miss", None)
            clear_memo_cache()

    def test_non_memoizable_tool_always_invokes_handler(self):
        """A tool with can_memoize=False always runs the handler."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_no_memo_dispatch",
            toolset="test",
            schema={"name": "_test_no_memo_dispatch", "description": "t"},
            handler=counting_handler,
            can_memoize=False,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_no_memo_dispatch"] = reg._tools["_test_no_memo_dispatch"]
        clear_memo_cache()

        try:
            handle_function_call("_test_no_memo_dispatch", {"x": 1})
            handle_function_call("_test_no_memo_dispatch", {"x": 1})
            assert call_count == 2
        finally:
            global_registry._tools.pop("_test_no_memo_dispatch", None)
            clear_memo_cache()


class TestClearMemoCache:
    """clear_memo_cache() invalidates all cached results."""

    def test_clear_forces_re_execution(self):
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_clear",
            toolset="test",
            schema={"name": "_test_memo_clear", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_clear"] = reg._tools["_test_memo_clear"]
        clear_memo_cache()

        try:
            r1 = handle_function_call("_test_memo_clear", {"x": 1})
            clear_memo_cache()
            r2 = handle_function_call("_test_memo_clear", {"x": 1})
            # After clearing, handler runs again
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
        finally:
            global_registry._tools.pop("_test_memo_clear", None)
            clear_memo_cache()
