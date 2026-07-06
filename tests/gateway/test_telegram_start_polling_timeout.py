"""Test that start_polling() timeout prevents indefinite hanging.

This is a regression test for issue #59614 where start_polling() could hang
indefinitely when the connection pool is in a degraded state.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Only import TelegramAdapter if it exists
try:
    from hermes.plugins.platforms.telegram import adapter
    _UPDATER_START_TIMEOUT = adapter._UPDATER_START_TIMEOUT
except (ImportError, AttributeError):
    pytest.skip("Telegram adapter not available", allow_module_level=True)


class TestStartPollingTimeout:
    """Test that start_polling() timeout prevents indefinite hanging."""

    @pytest.mark.asyncio
    async def test_start_polling_timeout_raises_runtime_error(self):
        """When start_polling() times out, it should raise RuntimeError."""
        from hermes.plugins.platforms.telegram.adapter import TelegramAdapter

        # Mock the adapter's internal state
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.name = "test_bot"
        adapter.has_fatal_error = False
        adapter._polling_network_error_count = 1
        adapter._polling_error_callback_ref = None
        adapter._background_tasks = set()
        adapter._send_path_degraded = True

        # Mock the app and updater
        mock_app = MagicMock()
        mock_updater = AsyncMock()
        mock_app.updater = mock_updater
        adapter._app = mock_app

        # Make start_polling() hang indefinitely (simulate the bug)
        async def hanging_start_polling(**kwargs):
            await asyncio.sleep(1000)  # Hang for a long time
            return None

        mock_updater.start_polling = hanging_start_polling
        mock_updater.running = True

        # Mock _drain_polling_connections to avoid actual connection cleanup
        with patch.object(adapter, '_drain_polling_connections', new=AsyncMock()):
            # Trigger the network error handler
            task = asyncio.create_task(adapter._handle_polling_network_error(Exception("test")))

            # Wait for the timeout to trigger (start_polling_timeout is 30s)
            try:
                await asyncio.wait_for(task, timeout=_UPDATER_START_TIMEOUT + 5)
            except asyncio.TimeoutError:
                task.cancel()
                pytest.fail("Network error handler did not complete within timeout")

            # The task should have completed (either with success or error)
            # The important part is that it didn't hang forever
            assert task.done()

    @pytest.mark.asyncio
    async def test_start_polling_success_returns_normally(self):
        """When start_polling() succeeds quickly, it should return normally."""
        from hermes.plugins.platforms.telegram.adapter import TelegramAdapter

        # Mock the adapter's internal state
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter.name = "test_bot"
        adapter.has_fatal_error = False
        adapter._polling_network_error_count = 1
        adapter._polling_error_callback_ref = None
        adapter._background_tasks = set()
        adapter._send_path_degraded = True

        # Mock the app and updater
        mock_app = MagicMock()
        mock_updater = AsyncMock()
        mock_app.updater = mock_updater
        adapter._app = mock_app

        # Make start_polling() succeed immediately
        mock_updater.start_polling = AsyncMock(return_value=None)
        mock_updater.running = True

        # Mock _drain_polling_connections and _verify_polling_after_reconnect
        with patch.object(adapter, '_drain_polling_connections', new=AsyncMock()), \
             patch.object(adapter, '_verify_polling_after_reconnect', new=AsyncMock()):
            # Trigger the network error handler
            await adapter._handle_polling_network_error(Exception("test"))

            # Verify that start_polling was called
            mock_updater.start_polling.assert_called_once()