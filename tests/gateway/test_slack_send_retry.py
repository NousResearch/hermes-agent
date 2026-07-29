"""Regression: Slack send() must surface retryable + retry_after on 429.

The Telegram adapter (PR #46762) extracts retry_after from FloodWait errors
and sets retryable=True so the base _send_with_retry() layer can honor the
server-requested backoff.  Slack's send() caught all exceptions and returned
a bare SendResult(success=False) with retryable=False — so 429 rate-limit
errors were never retried and remaining message chunks were silently dropped.

These tests verify the Slack adapter now:
  1. Sets retryable=True for 429 / 500+ / connection errors
  2. Extracts the Retry-After header when present
  3. Leaves non-retryable errors (403, 404, etc.) unchanged
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Ensure slack mocks are in place before importing the adapter
# ---------------------------------------------------------------------------

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
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    return a


def _slack_api_error(status_code: int, retry_after: str = None):
    """Simulate a SlackApiError with response status and optional Retry-After."""
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    resp = SimpleNamespace(
        status_code=status_code,
        headers=headers,
    )
    exc = Exception(f"The request to the Slack API failed. (status: {status_code})")
    exc.response = resp
    return exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSlackSendRetryable:
    @pytest.mark.asyncio
    async def test_429_returns_retryable_with_retry_after(self):
        adapter = _make_adapter()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(
            side_effect=_slack_api_error(429, retry_after="30")
        )
        adapter._get_client = lambda cid, team_id="": client

        result = await adapter.send("C123", "hello")
        assert not result.success
        assert result.retryable is True
        assert result.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_429_without_retry_after_header(self):
        adapter = _make_adapter()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(
            side_effect=_slack_api_error(429)
        )
        adapter._get_client = lambda cid, team_id="": client

        result = await adapter.send("C123", "hello")
        assert not result.success
        assert result.retryable is True
        assert result.retry_after is None

    @pytest.mark.asyncio
    async def test_500_is_retryable_no_retry_after(self):
        adapter = _make_adapter()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(
            side_effect=_slack_api_error(500)
        )
        adapter._get_client = lambda cid, team_id="": client

        result = await adapter.send("C123", "hello")
        assert not result.success
        assert result.retryable is True
        assert result.retry_after is None

    @pytest.mark.asyncio
    async def test_403_is_not_retryable(self):
        adapter = _make_adapter()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(
            side_effect=_slack_api_error(403)
        )
        adapter._get_client = lambda cid, team_id="": client

        result = await adapter.send("C123", "hello")
        assert not result.success
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_connection_error_is_retryable(self):
        adapter = _make_adapter()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(
            side_effect=ConnectionError("Connection reset by peer")
        )
        adapter._get_client = lambda cid, team_id="": client

        result = await adapter.send("C123", "hello")
        assert not result.success
        assert result.retryable is True
