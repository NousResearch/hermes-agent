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
from pathlib import Path
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
from hermes_constants import get_hermes_home  # noqa: E402
from plugins.platforms.slack.adapter import (  # noqa: E402
    _slack_dedup_state_path,
    _slack_dedup_ttl_seconds,
)


def test_default_ttl_outlasts_slack_reconnect_redelivery_window():
    # The whole point of the fix: the window must be much longer than the
    # ~6 min reconnect-redelivery gap that caused the duplicate reply.
    with patch.dict(os.environ, {}, clear=True):
        assert _slack_dedup_ttl_seconds() >= 1800.0


def test_env_override_is_respected():
    with patch.dict(os.environ, {"SLACK_DEDUP_TTL_SECONDS": "120"}, clear=True):
        assert _slack_dedup_ttl_seconds() == 120.0


def test_dedup_state_lives_in_the_profile_home():
    """The seen-set is per-profile: each profile's bot has its own Slack event stream."""
    path = _slack_dedup_state_path()
    assert path.parent == Path(get_hermes_home())
    assert path.name == "slack_seen_event_ids.json"


def test_adapter_dedup_survives_a_restart_via_its_state_path(tmp_path):
    """A redelivered event must be suppressed by a process that did not receive it."""
    state = tmp_path / "slack_seen_event_ids.json"
    before = MessageDeduplicator(ttl_seconds=1800, persist_path=state)
    assert before.is_duplicate("T0BURRXNHQB:1789134549.911449") is False
    # The restart: same path, new process.
    after = MessageDeduplicator(ttl_seconds=1800, persist_path=state)
    assert after.is_duplicate("T0BURRXNHQB:1789134549.911449") is True


