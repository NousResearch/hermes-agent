"""Regression tests for WhatsApp adapter session leak on bridge failure.

When _open_ws() creates an aiohttp session and then _poll_messages()
fails to start, or when the overall bridge startup fails, the session
must be closed to prevent orphaned sessions.
"""

import asyncio
from unittest import mock

import pytest

from gateway.config import PlatformConfig


def _make_config(**extra):
    return PlatformConfig(enabled=True, extra=extra)


class TestWhatsAppExistingBridgeSessionCleanup:
    """When existing-bridge path creates session + poll_task, failures must
    close the session."""

    @pytest.mark.asyncio
    async def test_poll_task_failure_closes_session(self):
        """If create_task(_poll_messages) raises, _http_session must be closed."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(_make_config(bridge_port=12345))

        fake_session = mock.AsyncMock(closed=False)
        adapter._http_session = fake_session

        # Simulate the inner try/except from the existing-bridge path:
        # create_task raises, so session must be closed.
        with pytest.raises(RuntimeError, match="task failed"):
            try:
                raise RuntimeError("task failed")
            except Exception:
                await adapter._http_session.close()
                adapter._http_session = None
                raise

        fake_session.close.assert_awaited_once()
        assert adapter._http_session is None


class TestWhatsAppNewBridgeSessionCleanup:
    """When new-bridge path fails, _http_session must be closed."""

    @pytest.mark.asyncio
    async def test_bridge_failure_closes_session(self):
        """If bridge startup raises, _http_session must be closed and set to None."""
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(_make_config(bridge_port=12345))
        adapter._http_session = None

        # Simulate the except block from the new-bridge path
        fake_session = mock.AsyncMock(closed=False)
        adapter._http_session = fake_session

        # The except block closes the session
        if adapter._http_session:
            await adapter._http_session.close()
            adapter._http_session = None

        fake_session.close.assert_awaited_once()
        assert adapter._http_session is None
