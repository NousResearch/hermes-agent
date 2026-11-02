"""Tests for MatrixAdapter.create_handoff_thread.

Matrix has no channel-level "create thread" API, so the adapter seeds a root
message and returns its event_id as the thread handle (Slack-style — mirrors
``tests/gateway/test_slack_sdk_response.py::TestHandoffThread``). These tests
cover: the seed event id becomes the thread id (and the root is registered as a
participated thread), and ``None`` is returned when the seed send fails (#112918).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _make_adapter():
    """Create a MatrixAdapter with mocked config (no live client)."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._startup_ts = time.time() - 10
    return adapter


class TestHandoffThread:
    """``create_handoff_thread`` anchors a session on the seed event's id."""

    def test_seed_event_becomes_the_thread_id(self):
        adapter = _make_adapter()
        adapter._client = MagicMock()  # truthy: a client is connected
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$root_evt"))

        thread_id = asyncio.run(adapter.create_handoff_thread("!room1:example.org", "Hermes — daily brief"))

        assert thread_id == "$root_evt"
        adapter.send.assert_awaited_once_with("!room1:example.org", "Hermes — daily brief")
        # Root registered so inbound replies in this thread are recognised.
        assert "$root_evt" in adapter._threads

    def test_failed_seed_send_yields_no_thread(self):
        adapter = _make_adapter()
        adapter._client = MagicMock()
        adapter.send = AsyncMock(return_value=SendResult(success=False, error="boom"))

        assert asyncio.run(adapter.create_handoff_thread("!room1:example.org", "x")) is None
