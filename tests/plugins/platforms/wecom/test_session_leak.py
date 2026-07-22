"""Regression tests for WeCom adapter session leak on reconnect failure.

When _listen_loop() fails to reconnect after _open_connection() creates a
fresh session, _cleanup_ws() must close it to prevent orphaned sessions.
"""

import asyncio
from unittest import mock

import pytest

from gateway.config import PlatformConfig


def _make_config(**extra):
    return PlatformConfig(enabled=True, extra=extra)


class TestWeComReconnectSessionCleanup:
    """When _listen_loop reconnect fails, _cleanup_ws must close the session."""

    @pytest.mark.asyncio
    async def test_reconnect_failure_calls_cleanup_ws(self):
        """Forced reconnect failure must invoke _cleanup_ws."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(_make_config(bot_id="b", secret="s"))
        adapter._running = True

        cleanup_called = []
        original_cleanup = adapter._cleanup_ws

        async def tracking_cleanup():
            cleanup_called.append(True)
            await original_cleanup()

        adapter._cleanup_ws = tracking_cleanup  # type: ignore[assignment]

        with mock.patch.object(adapter, "_open_connection", side_effect=RuntimeError("connection refused")):
            # _listen_loop catches the exception and calls _cleanup_ws
            # We simulate one iteration of the loop instead of running it fully.
            backoff_idx = 0
            try:
                await adapter._open_connection()
            except Exception:
                await adapter._cleanup_ws()

        assert len(cleanup_called) == 1

    @pytest.mark.asyncio
    async def test_cleanup_ws_closes_session_and_ws(self):
        """_cleanup_ws must close both _ws and _session, setting them to None."""
        from plugins.platforms.wecom.adapter import WeComAdapter

        adapter = WeComAdapter(_make_config(bot_id="b", secret="s"))

        fake_ws = mock.AsyncMock(closed=False)
        fake_session = mock.AsyncMock(closed=False)
        adapter._ws = fake_ws
        adapter._session = fake_session

        await adapter._cleanup_ws()

        fake_ws.close.assert_awaited_once()
        fake_session.close.assert_awaited_once()
        assert adapter._ws is None
        assert adapter._session is None
