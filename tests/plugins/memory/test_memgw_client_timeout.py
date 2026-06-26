"""Tests for MemGatewayClient timeout cancellation (Codex PR #30 review #13).

When an MCP call exceeds the timeout, `_run_sync` must cancel the underlying
coroutine so a stalled endpoint doesn't leave a pending HTTP session running on
the shared background loop after the caller has already given up.
"""

import asyncio
import threading

import pytest

from plugins.memory.memgw.client import MemGatewayClient


class TestRunSyncCancelsOnTimeout:
    def test_timeout_cancels_the_coroutine(self):
        """The cancelled coroutine must observe CancelledError, not keep pending."""
        cancelled = threading.Event()

        async def _tracks_cancellation():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        client = MemGatewayClient('http://127.0.0.1:9/mcp', api_key='k', timeout=0.05)
        with pytest.raises(TimeoutError):
            client._run_sync(_tracks_cancellation())

        # Give the loop a moment to deliver the cancellation to the task.
        cancelled.wait(timeout=2.0)
        assert cancelled.is_set(), (
            "coroutine was not cancelled after timeout — the timed-out MCP call "
            "is still pending on the shared loop (Codex PR #30 review #13)"
        )
        client.close()

    def test_normal_call_returns_result(self):
        client = MemGatewayClient('http://127.0.0.1:9/mcp', api_key='k', timeout=5)

        async def _quick():
            return {'value': 42}

        assert client._run_sync(_quick()) == {'value': 42}
        client.close()
