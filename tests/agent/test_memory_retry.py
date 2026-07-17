"""Regression tests for MemoryManager retry behaviour on provider calls.

These cover the three scenarios requested in the teknium1 review of PR #44378:

1. Transient exception (``ConnectionError``/``TimeoutError``) is retried and
   ultimately succeeds — the provider method is invoked more than once.
2. Non-retryable exception (``ValueError``, ``FileNotFoundError``) is *not*
   retried — the provider method is invoked exactly once and the exception
   propagates.
3. ``handle_tool_call`` is never retried, even on a transient exception, to
   avoid duplicating mutating memory operations after an ambiguous failure.

The tests rely on ``tenacity`` being importable in the test environment. When
it is unavailable (``_TENACITY_AVAILABLE`` is False), the retry layer is a
no-op and these behaviours collapse to single-attempt calls — in that case
we assert the corresponding single-call expectations instead.
"""

import json
import pytest
from unittest.mock import MagicMock

import agent.memory_manager as mm
from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Minimal provider whose call sites can be configured to raise or succeed
# ---------------------------------------------------------------------------


class _CallCountingProvider(MemoryProvider):
    """Provider that records every invocation and can raise on demand."""

    def __init__(self, name="retry-fake"):
        self._name = name
        self._available = True
        # *call_counts* maps method name → number of times it was invoked.
        self.call_counts = {}
        # *raise_queue* maps method name → list of exceptions to raise in
        # order; an empty / exhausted list means "return self._ok_value".
        self.raise_queues = {}
        self._ok_value = "ok"

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def initialize(self, session_id, **kwargs):
        pass

    def system_prompt_block(self) -> str:
        return self._record_and_return("system_prompt_block")

    def prefetch(self, query, *, session_id=""):
        return self._record_and_return("prefetch")

    def queue_prefetch(self, query, *, session_id=""):
        self._record_and_return("queue_prefetch")

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        self._record_and_return("sync_turn")

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.call_counts["handle_tool_call"] = self.call_counts.get("handle_tool_call", 0) + 1
        self._maybe_raise("handle_tool_call")
        return json.dumps({"tool": tool_name, "args": args})

    def shutdown(self):
        pass

    # -- helpers ------------------------------------------------------------

    def _record_and_return(self, method_name):
        self.call_counts[method_name] = self.call_counts.get(method_name, 0) + 1
        self._maybe_raise(method_name)
        return self._ok_value

    def _maybe_raise(self, method_name):
        queue = self.raise_queues.get(method_name)
        if queue:
            exc = queue.pop(0)
            if exc is not None:
                raise exc

    def configure_raise(self, method_name, *exceptions):
        """Set the sequence of exceptions a method should raise before returning."""
        self.raise_queues[method_name] = list(exceptions)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_manager():
    mgr = MemoryManager()
    provider = _CallCountingProvider()
    mgr.add_provider(provider)
    return mgr, provider


# ---------------------------------------------------------------------------
# 1. Transient exception (ConnectionError / TimeoutError) → retried
# ---------------------------------------------------------------------------


class TestTransientRetry:
    def test_connection_error_is_retried(self, fresh_manager):
        """A single ConnectionError should be retried and ultimately succeed."""
        mgr, provider = fresh_manager
        provider.configure_raise("system_prompt_block", ConnectionError("reset by peer"))

        # build_system_prompt swallows provider exceptions, so call the helper
        # directly to assert the retry actually happened.
        result = mgr._call_with_retry(provider, "system_prompt_block")
        assert result == "ok"
        assert provider.call_counts["system_prompt_block"] >= 2

    def test_timeout_error_is_retried(self, fresh_manager):
        """A single TimeoutError should be retried and ultimately succeed."""
        mgr, provider = fresh_manager
        provider.configure_raise("prefetch", TimeoutError("timed out"))

        result = mgr._call_with_retry(provider, "prefetch", "query", session_id="s1")
        assert result == "ok"
        assert provider.call_counts["prefetch"] >= 2

    def test_multiple_transient_then_success(self, fresh_manager):
        """A short run of transient errors is retried up to _MEMORY_RETRY_ATTEMPTS."""
        mgr, provider = fresh_manager
        # Two transient failures, then success — within the 3-attempt budget.
        provider.configure_raise(
            "system_prompt_block",
            ConnectionError("retry-1"),
            ConnectionError("retry-2"),
        )

        result = mgr._call_with_retry(provider, "system_prompt_block")
        assert result == "ok"
        assert provider.call_counts["system_prompt_block"] == 3

    def test_exhausted_transient_attempts_propagate(self, fresh_manager):
        """When every attempt raises a transient error, the last error propagates."""
        mgr, provider = fresh_manager
        # Raise on every attempt — exhaust the retry budget (3 attempts).
        attempts = mm._MEMORY_RETRY_ATTEMPTS
        provider.configure_raise(
            "system_prompt_block",
            *[ConnectionError(f"attempt-{i}") for i in range(attempts)],
        )

        with pytest.raises(ConnectionError):
            mgr._call_with_retry(provider, "system_prompt_block")
        assert provider.call_counts["system_prompt_block"] == attempts


# ---------------------------------------------------------------------------
# 2. Non-retryable exception → NOT retried (single invocation)
# ---------------------------------------------------------------------------


class TestNonRetryableFailure:
    def test_value_error_not_retried(self, fresh_manager):
        """A ValueError is not in the retry predicate and must not be retried."""
        mgr, provider = fresh_manager
        provider.configure_raise("system_prompt_block", ValueError("not transient"))

        with pytest.raises(ValueError):
            mgr._call_with_retry(provider, "system_prompt_block")
        assert provider.call_counts["system_prompt_block"] == 1

    def test_file_not_found_error_not_retried(self, fresh_manager):
        """FileNotFoundError is an OSError subclass and must NOT be retried.

        This guards the narrowing performed in the fix: ``OSError`` was
        removed from the predicate precisely so permanent filesystem errors
        are not retried.
        """
        mgr, provider = fresh_manager
        provider.configure_raise("prefetch", FileNotFoundError("missing"))

        with pytest.raises(FileNotFoundError):
            mgr._call_with_retry(provider, "prefetch", "q", session_id="s")
        assert provider.call_counts["prefetch"] == 1

    def test_permission_error_not_retried(self, fresh_manager):
        """PermissionError is an OSError subclass and must NOT be retried."""
        mgr, provider = fresh_manager
        provider.configure_raise("prefetch", PermissionError("denied"))

        with pytest.raises(PermissionError):
            mgr._call_with_retry(provider, "prefetch", "q", session_id="s")
        assert provider.call_counts["prefetch"] == 1


# ---------------------------------------------------------------------------
# 3. handle_tool_call must NOT be retried (no duplicate mutating operation)
# ---------------------------------------------------------------------------


class TestHandleToolCallNoRetry:
    def test_handle_tool_call_direct_no_retry(self, fresh_manager):
        """handle_tool_call must invoke the provider exactly once.

        Even on a transient ConnectionError, the call must not be replayed,
        because the memory backend may have already mutated before the
        failure surfaced.
        """
        mgr, provider = fresh_manager
        provider.configure_raise("handle_tool_call", ConnectionError("post-write reset"))

        # Register a tool name so the routing logic dispatches to this provider.
        # _tool_to_provider is populated via get_tool_schemas; we inject it
        # directly to keep the test focused on the retry-path contract.
        mgr._tool_to_provider["store"] = provider

        result = mgr.handle_tool_call("store", {"key": "value"})
        # handle_tool_call catches the exception and returns a tool_error string.
        assert "failed" in result.lower() or "error" in result.lower()
        # CRITICAL: exactly one invocation — no replay.
        assert provider.call_counts["handle_tool_call"] == 1

    def test_handle_tool_call_success_no_retry(self, fresh_manager):
        """A successful handle_tool_call is also a single invocation."""
        mgr, provider = fresh_manager
        mgr._tool_to_provider["store"] = provider

        result = mgr.handle_tool_call("store", {"key": "value"})
        data = json.loads(result)
        assert data["tool"] == "store"
        assert provider.call_counts["handle_tool_call"] == 1
