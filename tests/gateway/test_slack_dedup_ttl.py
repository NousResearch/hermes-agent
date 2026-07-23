"""
Tests for Slack Socket Mode dedup TTL (#4777).

Slack replays un-acked Socket Mode events when the websocket reconnects.
The replay can land several minutes after the original; the dedup window
must outlast that gap so the redelivered event is suppressed instead of
producing a second bot reply. Regression for the 300s-default bug where
replays >5 min later slipped through.

Follows the slack-bolt mocking pattern from test_slack_mention.py.
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch


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

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.helpers import MessageDeduplicator  # noqa: E402
from plugins.platforms.slack.adapter import _slack_dedup_ttl_seconds  # noqa: E402


def test_default_ttl_outlasts_slack_reconnect_redelivery_window():
    # The whole point of the fix: the window must be much longer than the
    # ~6 min reconnect-redelivery gap that caused the duplicate reply.
    with patch.dict(os.environ, {}, clear=True):
        assert _slack_dedup_ttl_seconds() >= 1800.0


def test_env_override_is_respected():
    with patch.dict(os.environ, {"SLACK_DEDUP_TTL_SECONDS": "120"}, clear=True):
        assert _slack_dedup_ttl_seconds() == 120.0


def test_invalid_env_falls_back_to_default():
    with patch.dict(os.environ, {"SLACK_DEDUP_TTL_SECONDS": "not-a-number"}, clear=True):
        assert _slack_dedup_ttl_seconds() >= 1800.0
    with patch.dict(os.environ, {"SLACK_DEDUP_TTL_SECONDS": "0"}, clear=True):
        assert _slack_dedup_ttl_seconds() >= 1800.0


def test_redelivery_six_minutes_later_is_suppressed():
    """A replay 6 min after first processing must be treated as duplicate."""
    dedup = MessageDeduplicator(ttl_seconds=_slack_dedup_ttl_seconds())
    event_ts = "1733382960.001500"

    # First delivery — recorded now.
    assert dedup.is_duplicate(event_ts) is False
    # Simulate the entry being stamped 6 minutes ago (reconnect redelivery gap).
    dedup._seen[event_ts] = time.time() - 360
    # Redelivery of the SAME event must still be caught.
    assert dedup.is_duplicate(event_ts) is True


def test_old_default_300s_would_have_missed_it():
    """Pins the regression: the prior 300s window let the replay through."""
    dedup = MessageDeduplicator(ttl_seconds=300)
    event_ts = "1733382960.001500"
    assert dedup.is_duplicate(event_ts) is False
    dedup._seen[event_ts] = time.time() - 360  # 6 min ago, past 300s TTL
    # Demonstrates the bug: replay treated as new → second reply.
    assert dedup.is_duplicate(event_ts) is False
