"""Tests for per-tool timeout enforcement in registry.dispatch().

ToolEntry now accepts an optional ``timeout`` field (seconds).  When set,
registry.dispatch() wraps the handler invocation in a timeout guard.  If the
handler exceeds the limit, a JSON error is returned instead of hanging.
"""

import asyncio
import json
import time

import pytest

from tools.registry import ToolRegistry, ToolEntry


# ── Unit tests for ToolEntry.timeout field ─────────────────────────────────


class TestToolEntryTimeout:
    """ToolEntry stores per-tool timeout."""

    def test_default_timeout_is_none(self):
        entry = ToolEntry(
            name="t", toolset="test", schema={}, handler=lambda: None,
            check_fn=None, requires_env=[], is_async=False,
            description="", emoji="", max_result_size_chars=None,
            timeout=None,
        )
        assert entry.timeout is None

    def test_custom_timeout_stored(self):
        entry = ToolEntry(
            name="t", toolset="test", schema={}, handler=lambda: None,
            check_fn=None, requires_env=[], is_async=False,
            description="", emoji="", max_result_size_chars=None,
            timeout=30,
        )
        assert entry.timeout == 30


class TestRegistryRegisterTimeout:
    """registry.register() passes timeout to ToolEntry."""

    def test_register_with_timeout(self):
        reg = ToolRegistry()
        reg.register(
            name="_test_timeout_tool",
            toolset="test",
            schema={"name": "_test_timeout_tool", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
            timeout=10,
        )
        assert reg._tools["_test_timeout_tool"].timeout == 10

    def test_register_without_timeout_defaults_none(self):
        reg = ToolRegistry()
        reg.register(
            name="_test_no_timeout",
            toolset="test",
            schema={"name": "_test_no_timeout", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
        )
        assert reg._tools["_test_no_timeout"].timeout is None


# ── Integration tests for dispatch timeout enforcement ─────────────────────


class TestDispatchTimeoutEnforcement:
    """registry.dispatch() enforces per-tool timeout."""

    def test_fast_tool_completes_within_timeout(self):
        """A handler that finishes quickly returns its result normally."""
        reg = ToolRegistry()
        reg.register(
            name="_test_fast",
            toolset="test",
            schema={"name": "_test_fast", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
            timeout=5,
        )
        result = reg.dispatch("_test_fast", {})
        parsed = json.loads(result)
        assert parsed["ok"] is True

    def test_slow_tool_exceeds_timeout(self):
        """A handler that sleeps past its timeout returns a timeout error."""
        reg = ToolRegistry()
        reg.register(
            name="_test_slow",
            toolset="test",
            schema={"name": "_test_slow", "description": "t"},
            handler=lambda args, **kw: (time.sleep(10), json.dumps({"ok": True}))[1],
            timeout=1,
        )
        result = reg.dispatch("_test_slow", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "timed out" in parsed["error"].lower() or "timeout" in parsed["error"].lower()

    def test_no_timeout_means_no_limit(self):
        """A tool with timeout=None runs without a timeout guard."""
        reg = ToolRegistry()
        reg.register(
            name="_test_no_limit",
            toolset="test",
            schema={"name": "_test_no_limit", "description": "t"},
            handler=lambda args, **kw: json.dumps({"ok": True}),
            timeout=None,
        )
        result = reg.dispatch("_test_no_limit", {})
        parsed = json.loads(result)
        assert parsed["ok"] is True

    def test_timeout_error_includes_tool_name(self):
        """Timeout error message includes the tool name for debugging."""
        reg = ToolRegistry()
        reg.register(
            name="_test_named_slow",
            toolset="test",
            schema={"name": "_test_named_slow", "description": "t"},
            handler=lambda args, **kw: (time.sleep(10), json.dumps({}))[1],
            timeout=1,
        )
        result = reg.dispatch("_test_named_slow", {})
        parsed = json.loads(result)
        assert "_test_named_slow" in parsed["error"]

    def test_timeout_error_includes_timeout_value(self):
        """Timeout error message includes the timeout duration."""
        reg = ToolRegistry()
        reg.register(
            name="_test_timeout_info",
            toolset="test",
            schema={"name": "_test_timeout_info", "description": "t"},
            handler=lambda args, **kw: (time.sleep(10), json.dumps({}))[1],
            timeout=1,
        )
        result = reg.dispatch("_test_timeout_info", {})
        parsed = json.loads(result)
        assert "1" in parsed["error"]

    def test_dispatch_returns_promptly_on_timeout(self):
        """dispatch() must return within ~timeout seconds, not wait for the thread."""
        reg = ToolRegistry()
        reg.register(
            name="_test_prompt_return",
            toolset="test",
            schema={"name": "_test_prompt_return", "description": "t"},
            handler=lambda args, **kw: (time.sleep(10), json.dumps({"ok": True}))[1],
            timeout=1,
        )
        start = time.monotonic()
        result = reg.dispatch("_test_prompt_return", {})
        elapsed = time.monotonic() - start
        # Must return well before the handler's 10s sleep completes.
        assert elapsed < 3.0, f"dispatch blocked for {elapsed:.2f}s; expected ~1s"
        parsed = json.loads(result)
        assert "error" in parsed

    def test_async_tool_timeout_is_enforced(self):
        """Async handlers with a timeout are subject to the same timeout guard."""
        reg = ToolRegistry()

        async def slow_async(args, **kw):
            await asyncio.sleep(10)
            return json.dumps({"ok": True})

        reg.register(
            name="_test_async_slow",
            toolset="test",
            schema={"name": "_test_async_slow", "description": "t"},
            handler=slow_async,
            is_async=True,
            timeout=1,
        )
        start = time.monotonic()
        result = reg.dispatch("_test_async_slow", {})
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f"async dispatch blocked for {elapsed:.2f}s; expected ~1s"
        parsed = json.loads(result)
        assert "error" in parsed
        assert "_test_async_slow" in parsed["error"]

    def test_async_tool_completes_within_timeout(self):
        """A fast async handler with a timeout returns its result normally."""
        reg = ToolRegistry()

        async def fast_async(args, **kw):
            return json.dumps({"async": True})

        reg.register(
            name="_test_async_fast",
            toolset="test",
            schema={"name": "_test_async_fast", "description": "t"},
            handler=fast_async,
            is_async=True,
            timeout=5,
        )
        result = reg.dispatch("_test_async_fast", {})
        parsed = json.loads(result)
        assert parsed["async"] is True
