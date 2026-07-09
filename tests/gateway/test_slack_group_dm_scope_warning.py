"""
Tests for the connect-time group-DM scope nudge.

When a Slack app handles 1:1 DMs (``im:history`` granted) but is missing
``mpim:history``, group DMs are silently dropped by Slack before the adapter
ever sees them. ``_warn_if_missing_group_dm_scopes`` inspects the
``x-oauth-scopes`` header from ``auth.test`` at connect time and logs an
actionable reinstall nudge — the only point where a stale install is
detectable, since a missing event produces no runtime API error.
"""

import logging
import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
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
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


class _FakeAuthResponse:
    """Mimics slack_sdk's AsyncSlackResponse — a .headers dict carrying scopes."""

    def __init__(self, scopes_csv):
        self.headers = {"x-oauth-scopes": scopes_csv}


def _make_adapter():
    # object.__new__ skips __init__ (heavy setup) — established slack-test pattern.
    return object.__new__(SlackAdapter)


def test_warns_when_mpim_history_missing(caplog):
    adapter = _make_adapter()
    resp = _FakeAuthResponse("chat:write,im:history,im:read,channels:history")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
    assert any("Group DMs" in r.message and "mpim:history" in r.message
               for r in caplog.records)


def test_no_warning_when_mpim_history_present(caplog):
    adapter = _make_adapter()
    resp = _FakeAuthResponse("chat:write,im:history,mpim:history,mpim:read")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
    assert not any("Group DMs" in r.message for r in caplog.records)


def test_no_warning_when_no_dm_scopes_at_all(caplog):
    # A channel-only app (no im:history) shouldn't be nudged about group DMs.
    adapter = _make_adapter()
    resp = _FakeAuthResponse("chat:write,channels:history")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
    assert not any("Group DMs" in r.message for r in caplog.records)


def test_warns_only_once_per_workspace(caplog):
    adapter = _make_adapter()
    resp = _FakeAuthResponse("im:history")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
    warnings = [r for r in caplog.records if "Group DMs" in r.message]
    assert len(warnings) == 1


def test_missing_header_does_not_warn(caplog):
    # Header absent (e.g. some proxies strip it) — don't guess, stay silent.
    adapter = _make_adapter()
    resp = _FakeAuthResponse("")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_missing_group_dm_scopes(resp, "Acme")
    assert not any("Group DMs" in r.message for r in caplog.records)
