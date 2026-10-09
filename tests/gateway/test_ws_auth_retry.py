"""Tests for auth-aware retry in Mattermost WS and Matrix sync loops.

Both Mattermost's _ws_loop and Matrix's _sync_loop previously caught all
exceptions with a broad ``except Exception`` and retried forever. Permanent
auth failures (401, 403, M_UNKNOWN_TOKEN) would loop indefinitely instead
of stopping. These tests verify that auth errors now stop the reconnect.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Mattermost: _ws_loop auth-aware retry
# ---------------------------------------------------------------------------

class TestMattermostWSAuthRetry:
    """gateway/platforms/mattermost.py — _ws_loop()"""

    def test_401_handshake_stops_reconnect(self):
        """A WSServerHandshakeError with status 401 should stop the loop."""
        import aiohttp

        exc = aiohttp.WSServerHandshakeError(
            request_info=MagicMock(),
            history=(),
            status=401,
            message="Unauthorized",
            headers=MagicMock(),
        )

        from plugins.platforms.mattermost.adapter import MattermostAdapter
        adapter = MattermostAdapter.__new__(MattermostAdapter)
        adapter._closing = False

        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            raise exc

        adapter._ws_connect_and_listen = fake_connect

        asyncio.run(adapter._ws_loop())

        # Should have attempted once and stopped, not retried
        assert call_count == 1


# ---------------------------------------------------------------------------
# Matrix: _sync_loop auth-aware retry
# ---------------------------------------------------------------------------

class TestMatrixSyncAuthRetry:
    """gateway/platforms/matrix.py — _sync_loop()"""

    def test_unknown_token_sync_error_stops_loop(self):
        """A SyncError with M_UNKNOWN_TOKEN should stop syncing."""
        import types
        nio_mock = types.ModuleType("nio")

        class SyncError:
            def __init__(self, message):
                self.message = message

        nio_mock.SyncError = SyncError

        from plugins.platforms.matrix.adapter import MatrixAdapter
        adapter = MatrixAdapter.__new__(MatrixAdapter)
        adapter._closing = False

        sync_count = 0

        async def fake_sync(timeout=30000, since=None):
            nonlocal sync_count
            sync_count += 1
            return SyncError("M_UNKNOWN_TOKEN: Invalid access token")

        adapter._client = MagicMock()
        adapter._client.sync = fake_sync
        adapter._client.sync_store = MagicMock()
        adapter._client.sync_store.get_next_batch = AsyncMock(return_value=None)
        adapter._pending_megolm = []
        adapter._joined_rooms = set()

        async def run():
            import sys
            sys.modules["nio"] = nio_mock
            try:
                await adapter._sync_loop()
            finally:
                del sys.modules["nio"]

        asyncio.run(run())
        assert sync_count == 1


