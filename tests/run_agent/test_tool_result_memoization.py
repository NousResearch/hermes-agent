"""Tests for tool result memoization (caching of idempotent tool results).

Tools can opt in to result caching by setting ``can_memoize=True`` at
registration.  When enabled, ``handle_function_call()`` caches the dispatch
result keyed by ``(tool_name, args_hash)`` within the current agent session
and turn, returning the cached value on subsequent calls with identical
arguments — saving the cost of re-executing idempotent handlers like file
reads or web fetches.

Cache semantics:
  - Scoped per session_id: two sessions with the same tool+args do NOT share
    results (prevents cross-session data leakage in gateway mode).
  - Bounded: at most _MEMO_MAX_ENTRIES_PER_SESSION entries per session (LRU).
  - Per-turn: clear_memo_cache(session_id) must be called at turn start so
    stale results from prior turns are not served in the current turn.
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

_TEST_SESSION = "test-session-memo"


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
        clear_memo_cache(_TEST_SESSION)

        try:
            r1 = handle_function_call("_test_memo_hit", {"x": 1}, session_id=_TEST_SESSION)
            r2 = handle_function_call("_test_memo_hit", {"x": 1}, session_id=_TEST_SESSION)
            # Both calls should return the same (first) result
            assert r1 == r2
            assert json.loads(r1)["call"] == 1
            # Handler should only have been called once
            assert call_count == 1
        finally:
            global_registry._tools.pop("_test_memo_hit", None)
            clear_memo_cache(_TEST_SESSION)

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
        clear_memo_cache(_TEST_SESSION)

        try:
            r1 = handle_function_call("_test_memo_miss", {"x": 1}, session_id=_TEST_SESSION)
            r2 = handle_function_call("_test_memo_miss", {"x": 2}, session_id=_TEST_SESSION)
            # Different results — handler called twice
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
            assert call_count == 2
        finally:
            global_registry._tools.pop("_test_memo_miss", None)
            clear_memo_cache(_TEST_SESSION)

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
        clear_memo_cache(_TEST_SESSION)

        try:
            handle_function_call("_test_no_memo_dispatch", {"x": 1}, session_id=_TEST_SESSION)
            handle_function_call("_test_no_memo_dispatch", {"x": 1}, session_id=_TEST_SESSION)
            assert call_count == 2
        finally:
            global_registry._tools.pop("_test_no_memo_dispatch", None)
            clear_memo_cache(_TEST_SESSION)


class TestClearMemoCache:
    """clear_memo_cache() invalidates cached results."""

    def test_clear_session_forces_re_execution(self):
        """Clearing the session cache forces the handler to run again."""
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
        clear_memo_cache(_TEST_SESSION)

        try:
            r1 = handle_function_call("_test_memo_clear", {"x": 1}, session_id=_TEST_SESSION)
            clear_memo_cache(_TEST_SESSION)
            r2 = handle_function_call("_test_memo_clear", {"x": 1}, session_id=_TEST_SESSION)
            # After clearing, handler runs again
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
        finally:
            global_registry._tools.pop("_test_memo_clear", None)
            clear_memo_cache(_TEST_SESSION)

    def test_clear_all_forces_re_execution(self):
        """clear_memo_cache() with no args clears all sessions."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_clear_all",
            toolset="test",
            schema={"name": "_test_memo_clear_all", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_clear_all"] = reg._tools["_test_memo_clear_all"]
        clear_memo_cache()

        try:
            r1 = handle_function_call("_test_memo_clear_all", {"x": 1}, session_id=_TEST_SESSION)
            clear_memo_cache()  # no session_id — clears everything
            r2 = handle_function_call("_test_memo_clear_all", {"x": 1}, session_id=_TEST_SESSION)
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
        finally:
            global_registry._tools.pop("_test_memo_clear_all", None)
            clear_memo_cache()


class TestSessionIsolation:
    """Cache is scoped per session_id — sessions cannot observe each other's results."""

    def test_different_sessions_do_not_share_cache(self):
        """Two sessions with the same tool+args each get their own handler call."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_session_iso",
            toolset="test",
            schema={"name": "_test_memo_session_iso", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_session_iso"] = reg._tools["_test_memo_session_iso"]
        clear_memo_cache("session-A")
        clear_memo_cache("session-B")

        try:
            r1 = handle_function_call("_test_memo_session_iso", {"x": 1}, session_id="session-A")
            r2 = handle_function_call("_test_memo_session_iso", {"x": 1}, session_id="session-B")
            # Each session invokes the handler independently
            assert call_count == 2
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
        finally:
            global_registry._tools.pop("_test_memo_session_iso", None)
            clear_memo_cache("session-A")
            clear_memo_cache("session-B")

    def test_same_session_second_call_is_cached(self):
        """Within a session, a second call with the same args is served from cache."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_session_hit",
            toolset="test",
            schema={"name": "_test_memo_session_hit", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_session_hit"] = reg._tools["_test_memo_session_hit"]
        clear_memo_cache("session-X")

        try:
            r1 = handle_function_call("_test_memo_session_hit", {"x": 1}, session_id="session-X")
            r2 = handle_function_call("_test_memo_session_hit", {"x": 1}, session_id="session-X")
            assert call_count == 1
            assert r1 == r2
        finally:
            global_registry._tools.pop("_test_memo_session_hit", None)
            clear_memo_cache("session-X")


class TestErrorEnvelopeHandling:
    """Results with {\"error\": None} (success envelope) are cached; truthy errors are not."""

    def test_error_none_result_is_cached(self):
        """A result with error=None is a success envelope and must be cached."""
        call_count = 0

        def handler_with_error_none(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"error": None, "data": "payload", "call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_error_none",
            toolset="test",
            schema={"name": "_test_memo_error_none", "description": "t"},
            handler=handler_with_error_none,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_error_none"] = reg._tools["_test_memo_error_none"]
        clear_memo_cache(_TEST_SESSION)

        try:
            r1 = handle_function_call("_test_memo_error_none", {"x": 1}, session_id=_TEST_SESSION)
            r2 = handle_function_call("_test_memo_error_none", {"x": 1}, session_id=_TEST_SESSION)
            # error=None is a success envelope — second call must be a cache hit
            assert call_count == 1, (
                f"Handler was called {call_count} times; expected 1 (second call should be cached). "
                "This means results with {{\"error\": None}} are incorrectly treated as errors."
            )
            assert r1 == r2
        finally:
            global_registry._tools.pop("_test_memo_error_none", None)
            clear_memo_cache(_TEST_SESSION)

    def test_truthy_error_result_is_not_cached(self):
        """A result with a truthy error value must not be cached."""
        call_count = 0

        def failing_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"error": "something went wrong", "call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_real_error",
            toolset="test",
            schema={"name": "_test_memo_real_error", "description": "t"},
            handler=failing_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_real_error"] = reg._tools["_test_memo_real_error"]
        clear_memo_cache(_TEST_SESSION)

        try:
            handle_function_call("_test_memo_real_error", {"x": 1}, session_id=_TEST_SESSION)
            handle_function_call("_test_memo_real_error", {"x": 1}, session_id=_TEST_SESSION)
            # Error results must not be cached — handler runs twice
            assert call_count == 2
        finally:
            global_registry._tools.pop("_test_memo_real_error", None)
            clear_memo_cache(_TEST_SESSION)


class TestNoneSessionIdSkipsMemoization:
    """Memoization is skipped when session_id is None to prevent cross-caller contamination."""

    def test_none_session_id_never_caches(self):
        """Calls with session_id=None always invoke the handler — no shared bucket."""
        call_count = 0

        def counting_handler(args, **kw):
            nonlocal call_count
            call_count += 1
            return json.dumps({"call": call_count})

        reg = ToolRegistry()
        reg.register(
            name="_test_memo_none_session",
            toolset="test",
            schema={"name": "_test_memo_none_session", "description": "t"},
            handler=counting_handler,
            can_memoize=True,
        )

        from model_tools import registry as global_registry
        global_registry._tools["_test_memo_none_session"] = reg._tools["_test_memo_none_session"]
        clear_memo_cache()

        try:
            r1 = handle_function_call("_test_memo_none_session", {"x": 1}, session_id=None)
            r2 = handle_function_call("_test_memo_none_session", {"x": 1}, session_id=None)
            # session_id=None must bypass the cache — handler invoked both times
            assert call_count == 2, (
                f"Handler called {call_count} times; expected 2 (session_id=None must not cache). "
                "Two unrelated callers with no session_id must not share a cache bucket."
            )
            assert json.loads(r1)["call"] == 1
            assert json.loads(r2)["call"] == 2
        finally:
            global_registry._tools.pop("_test_memo_none_session", None)
            clear_memo_cache()
