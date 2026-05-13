"""Tests for Telegram polling conflict-retry failure path (issue #25221).

When _handle_polling_conflict's inner retry start_polling() fails, the failure
must be routed into the network reconnect ladder — not silently swallowed. This
was the bug: a failed retry inside the conflict handler left the gateway alive
but with dead polling, because no reconnect was scheduled.

The regression test simulates start_polling raising inside the conflict-retry
block, then asserts that _handle_polling_network_error is scheduled (not that
the conflict handler retries indefinitely on its own — that's the old behaviour).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter():
    """Minimal adapter for conflict-retry testing.

    TelegramAdapter.__init__ calls super().__init__(config, platform) which
    sets self.platform and self.config, then the __init__ body sets up all the
    _polling_* state.  We need to go through __init__ properly since it also
    sets up logging (via a logger property) and the parent class __init__ sets
    up _running, _fatal_error_code, etc.
    """
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)

    # Reset conflict state that __init__ may have set to defaults
    adapter._polling_conflict_count = 0
    adapter._polling_network_error_count = 0
    adapter._polling_error_callback_ref = MagicMock()
    adapter._background_tasks: set = set()

    # Mock the PTB Application — tests never make real network calls
    mock_app = MagicMock()
    mock_app.updater = MagicMock()
    mock_app.updater.running = True
    adapter._app = mock_app
    adapter._bot = SimpleNamespace(id=999, username="test_bot")

    return adapter


class TestConflictRetryStartPollingFailure:
    """Issue #25221 — retry start_polling failure inside conflict handler must
    schedule network reconnect, not return silently."""

    @pytest.mark.asyncio
    async def test_conflict_retry_start_polling_failure_schedules_network_reconnect(self):
        """When the conflict-retry start_polling raises, route the error into
        _handle_polling_network_error so the reconnect ladder advances.

        Before the fix: retry_err was swallowed (just logged and returned),
        leaving the adapter alive but with dead polling and no retry scheduled.

        After the fix: _handle_polling_network_error is scheduled as a background
        task, continuing the exponential-backoff reconnect chain.
        """
        adapter = _make_adapter()
        # Start at conflict count 1 so we're inside the retry loop
        adapter._polling_conflict_count = 1

        # Simulate start_polling raising in the conflict-retry path
        async def mock_start_polling(**kwargs):
            raise RuntimeError("conflict retry start_polling failed")

        async def mock_stop():
            pass

        adapter._app.updater.start_polling = mock_start_polling
        adapter._app.updater.stop = mock_stop
        adapter._drain_polling_connections = AsyncMock()
        adapter._handle_polling_network_error = AsyncMock()

        # Capture what gets scheduled via asyncio.ensure_future
        scheduled_coroutines = []
        original_ensure = asyncio.ensure_future

        def tracking_ensure_future(coro, *, loop=None):
            scheduled_coroutines.append(coro)
            return original_ensure(coro, loop=loop)

        with patch("asyncio.ensure_future", tracking_ensure_future):
            await adapter._handle_polling_conflict(RuntimeError("simulated 409 conflict"))

        # Run the scheduled coroutines so the mock gets called
        for coro in scheduled_coroutines:
            await coro

        # After the fix: a background task calling _handle_polling_network_error
        # must have been scheduled. Before the fix, no such task is scheduled.
        assert scheduled_coroutines, (
            "retry_err from conflict-retry start_polling must schedule "
            "a background task that calls _handle_polling_network_error, "
            "not silently return. A task should be added to _background_tasks."
        )
        assert adapter._handle_polling_network_error.called, (
            "the scheduled task must call _handle_polling_network_error"
        )

    @pytest.mark.asyncio
    async def test_conflict_retry_start_polling_failure_does_not_set_fatal_immediately(self):
        """Conflict-retry start_polling failure is a transient/retryable error —
        it should NOT immediately set fatal_error. The network reconnect ladder
        (via _handle_polling_network_error) handles escalation after retries
        are exhausted."""
        adapter = _make_adapter()
        adapter._polling_conflict_count = 1

        async def mock_start_polling(**kwargs):
            raise RuntimeError("conflict retry start_polling failed")

        adapter._app.updater.start_polling = mock_start_polling
        adapter._app.updater.stop = AsyncMock()
        adapter._drain_polling_connections = AsyncMock()
        adapter._handle_polling_network_error = AsyncMock()

        await adapter._handle_polling_conflict(RuntimeError("simulated 409"))

        # Fatal should NOT be set — this is still in the transient retry window
        assert not adapter.has_fatal_error, (
            "conflict-retry start_polling failure must not set fatal_error immediately; "
            "it should hand off to the network reconnect ladder."
        )

    @pytest.mark.asyncio
    async def test_conflict_retry_start_polling_failure_leaves_adapter_alive(self):
        """The adapter should remain in a non-fatal state, running but with
        polling dead until the network reconnect ladder resumes it."""
        adapter = _make_adapter()
        adapter._polling_conflict_count = 1
        adapter._running = True  # should remain True

        async def mock_start_polling(**kwargs):
            raise RuntimeError("conflict retry failed")

        adapter._app.updater.start_polling = mock_start_polling
        adapter._app.updater.stop = AsyncMock()
        adapter._drain_polling_connections = AsyncMock()
        adapter._handle_polling_network_error = AsyncMock()

        await adapter._handle_polling_conflict(RuntimeError("conflict"))

        assert adapter._running, "adapter should stay running after conflict-retry failure"
        assert not adapter.has_fatal_error


class TestConflictRetrySuccessResets:
    """When conflict-retry start_polling succeeds, conflict count resets."""

    @pytest.mark.asyncio
    async def test_successful_conflict_retry_resets_conflict_count(self):
        """After a successful retry, _polling_conflict_count must reset to 0
        so subsequent conflicts start fresh."""
        adapter = _make_adapter()
        adapter._polling_conflict_count = 2  # simulate mid-retry

        async def mock_start_polling(**kwargs):
            return  # success

        adapter._app.updater.start_polling = mock_start_polling
        adapter._app.updater.stop = AsyncMock()
        adapter._drain_polling_connections = AsyncMock()
        adapter._handle_polling_network_error = AsyncMock()

        await adapter._handle_polling_conflict(RuntimeError("conflict"))

        assert adapter._polling_conflict_count == 0, (
            "Successful conflict retry must reset _polling_conflict_count to 0"
        )