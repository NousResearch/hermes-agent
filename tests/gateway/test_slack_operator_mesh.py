"""
Tests for Slack operator mesh state management (slice 19: error/unavailable state).

Covers: operator state tracking, error/unavailable handling, mesh routing helpers.
"""

import asyncio
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig

# Mock the slack-bolt package if it's not installed
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)

    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import gateway.platforms.slack as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = MagicMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    return a


class TestOperatorMeshState:
    """Test operator state tracking in multi-agent mesh."""

    @pytest.mark.asyncio
    async def test_set_operator_state_creates_entry(self, adapter):
        """Setting operator state creates a tracking entry."""
        await adapter.set_operator_state("worker01", "available")

        state = await adapter.get_operator_state("worker01")
        assert state is not None
        assert state["state"] == "available"
        assert state["error"] is None
        assert "since" in state

    @pytest.mark.asyncio
    async def test_set_operator_state_with_error(self, adapter):
        """Setting error state records the error message."""
        await adapter.set_operator_state("worker01", "error", error="Connection timeout")

        state = await adapter.get_operator_state("worker01")
        assert state["state"] == "error"
        assert state["error"] == "Connection timeout"

    @pytest.mark.asyncio
    async def test_set_operator_state_updates_existing(self, adapter):
        """Setting state on existing operator updates the record."""
        await adapter.set_operator_state("worker01", "available")
        first_since = (await adapter.get_operator_state("worker01"))["since"]

        # Small delay to ensure timestamp changes
        await asyncio.sleep(0.01)

        await adapter.set_operator_state("worker01", "busy")
        state = await adapter.get_operator_state("worker01")

        assert state["state"] == "busy"
        assert state["since"] > first_since

    @pytest.mark.asyncio
    async def test_get_operator_state_unknown_returns_none(self, adapter):
        """Getting state for unknown operator returns None."""
        state = await adapter.get_operator_state("unknown_worker")
        assert state is None

    @pytest.mark.asyncio
    async def test_list_operators_returns_all(self, adapter):
        """List operators returns all tracked operators."""
        await adapter.set_operator_state("worker01", "available")
        await adapter.set_operator_state("worker02", "busy")
        await adapter.set_operator_state("worker03", "error", error="OOM")

        operators = await adapter.list_operators()
        assert len(operators) == 3
        assert operators["worker01"]["state"] == "available"
        assert operators["worker02"]["state"] == "busy"
        assert operators["worker03"]["state"] == "error"

    @pytest.mark.asyncio
    async def test_remove_operator_deletes_entry(self, adapter):
        """Removing operator deletes their state entry."""
        await adapter.set_operator_state("worker01", "available")

        removed = await adapter.remove_operator("worker01")
        assert removed is True

        state = await adapter.get_operator_state("worker01")
        assert state is None

    @pytest.mark.asyncio
    async def test_remove_operator_unknown_returns_false(self, adapter):
        """Removing unknown operator returns False."""
        removed = await adapter.remove_operator("unknown_worker")
        assert removed is False

    @pytest.mark.asyncio
    async def test_is_operator_available_true_for_available(self, adapter):
        """is_operator_available returns True for available operators."""
        await adapter.set_operator_state("worker01", "available")

        assert adapter.is_operator_available("worker01") is True

    @pytest.mark.asyncio
    async def test_is_operator_available_false_for_others(self, adapter):
        """is_operator_available returns False for non-available states."""
        await adapter.set_operator_state("worker01", "busy")
        await adapter.set_operator_state("worker02", "error")
        await adapter.set_operator_state("worker03", "unavailable")

        assert adapter.is_operator_available("worker01") is False
        assert adapter.is_operator_available("worker02") is False
        assert adapter.is_operator_available("worker03") is False

    def test_is_operator_available_false_for_unknown(self, adapter):
        """is_operator_available returns False for unknown operators."""
        assert adapter.is_operator_available("unknown_worker") is False

    @pytest.mark.asyncio
    async def test_mark_operator_error_sets_retryable_error(self, adapter):
        """mark_operator_error with retryable=True sets 'error' state."""
        await adapter.mark_operator_error("worker01", "Transient failure", retryable=True)

        state = await adapter.get_operator_state("worker01")
        assert state["state"] == "error"
        assert state["error"] == "Transient failure"

    @pytest.mark.asyncio
    async def test_mark_operator_error_sets_unavailable_for_non_retryable(self, adapter):
        """mark_operator_error with retryable=False sets 'unavailable' state."""
        await adapter.mark_operator_error("worker01", "Fatal crash", retryable=False)

        state = await adapter.get_operator_state("worker01")
        assert state["state"] == "unavailable"
        assert state["error"] == "Fatal crash"

    @pytest.mark.asyncio
    async def test_clear_operator_error_sets_available(self, adapter):
        """clear_operator_error marks operator as available."""
        await adapter.set_operator_state("worker01", "error", error="Something failed")

        cleared = await adapter.clear_operator_error("worker01")
        assert cleared is True

        state = await adapter.get_operator_state("worker01")
        assert state["state"] == "available"
        assert state["error"] is None

    @pytest.mark.asyncio
    async def test_clear_operator_error_unknown_returns_false(self, adapter):
        """clear_operator_error for unknown operator returns False."""
        cleared = await adapter.clear_operator_error("unknown_worker")
        assert cleared is False


class TestOperatorMeshStateLimits:
    """Test operator state tracking limits and cleanup."""

    @pytest.mark.asyncio
    async def test_operator_state_max_limit(self, adapter):
        """Operator state respects max limit by removing oldest."""
        # Set a small limit for testing
        adapter._OPERATOR_STATE_MAX = 5

        # Add 5 operators
        for i in range(5):
            await adapter.set_operator_state(f"worker{i}", "available")

        # Verify all 5 exist
        operators = await adapter.list_operators()
        assert len(operators) == 5

        # Small delay
        await asyncio.sleep(0.01)

        # Add 6th - should evict oldest (worker0)
        await adapter.set_operator_state("worker5", "busy")

        operators = await adapter.list_operators()
        assert len(operators) == 5
        assert "worker0" not in operators
        assert "worker5" in operators


class TestOperatorMeshDisconnectCleanup:
    """Test operator state cleanup on disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_operator_states(self, adapter):
        """Disconnect clears all operator states."""
        await adapter.set_operator_state("worker01", "available")
        await adapter.set_operator_state("worker02", "busy")

        # Mock the socket handler cleanup
        adapter._socket_watchdog_task = None
        adapter._handler = None

        await adapter.disconnect()

        operators = await adapter.list_operators()
        assert len(operators) == 0


class TestOperatorMeshConcurrency:
    """Test concurrent access to operator state."""

    @pytest.mark.asyncio
    async def test_concurrent_state_updates(self, adapter):
        """Concurrent state updates are handled safely."""
        async def update_state(worker_id: str, state: str):
            for _ in range(10):
                await adapter.set_operator_state(worker_id, state)
                await asyncio.sleep(0.001)

        # Launch concurrent updates
        await asyncio.gather(
            update_state("worker01", "available"),
            update_state("worker01", "busy"),
            update_state("worker02", "available"),
            update_state("worker02", "error"),
        )

        # Verify final state is valid
        state1 = await adapter.get_operator_state("worker01")
        state2 = await adapter.get_operator_state("worker02")

        assert state1 is not None
        assert state1["state"] in ("available", "busy")
        assert state2 is not None
        assert state2["state"] in ("available", "error")

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, adapter):
        """Concurrent reads and writes are handled safely."""
        async def writer():
            for i in range(20):
                await adapter.set_operator_state(f"worker{i % 5}", "busy" if i % 2 else "available")
                await asyncio.sleep(0.001)

        async def reader():
            for _ in range(20):
                await adapter.list_operators()
                await asyncio.sleep(0.001)

        await asyncio.gather(writer(), reader())

        # Should complete without errors
        operators = await adapter.list_operators()
        assert len(operators) <= 5
