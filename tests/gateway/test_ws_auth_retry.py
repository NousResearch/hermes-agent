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
        """A WSServerHandshakeError with status 401 should stop the loop —
        AND escalate through the fatal-error hook (a bare return used to
        leave _running True: dead listener, healthy-looking adapter)."""
        import aiohttp
        from gateway.config import Platform

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
        adapter._running = True
        adapter.platform = Platform.MATTERMOST
        notified = []

        async def fatal_handler(a):
            notified.append(a)

        adapter._fatal_error_handler = fatal_handler

        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            raise exc

        adapter._ws_connect_and_listen = fake_connect

        async def run():
            await adapter._ws_loop()
            # Let the detached fatal-handler task complete.
            await asyncio.sleep(0)

        asyncio.run(run())

        # Should have attempted once and stopped, not retried
        assert call_count == 1
        # Escalated: fatal error recorded, _running cleared, handler notified.
        assert adapter._fatal_error_code == "mattermost_auth_error"
        assert adapter._fatal_error_retryable is False
        assert adapter._running is False
        assert notified == [adapter]

    def test_transient_401_substring_does_not_stop_reconnect(self):
        """A transient exception whose stringified message merely contains
        "401" (e.g. a proxy error body with digits) must not be mistaken
        for a genuine auth rejection. The loop should log a warning and
        retry, not return."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        adapter = MattermostAdapter.__new__(MattermostAdapter)
        adapter._closing = False

        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Stop the loop once we've proven a retry happened.
                adapter._closing = True
            raise RuntimeError(
                "proxy returned HTTP/1.1 401 in body but connection reset"
            )

        adapter._ws_connect_and_listen = fake_connect

        with patch("asyncio.sleep", new=AsyncMock()):
            asyncio.run(adapter._ws_loop())

        # The substring fallback is gone, so this must retry past the
        # first attempt instead of returning immediately.
        assert call_count == 2

    def test_closing_flag_prevents_further_connect_attempts(self):
        """Existing self._closing early-return behavior is unaffected by
        the substring-fallback removal: once _closing is set, the loop
        must not attempt to connect at all."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        adapter = MattermostAdapter.__new__(MattermostAdapter)
        adapter._closing = True

        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("should never be called")

        adapter._ws_connect_and_listen = fake_connect

        asyncio.run(adapter._ws_loop())

        assert call_count == 0
