"""
Tests for the connect-time user-token (vs bot-token) nudge.

``auth.test`` returns the ``user_id`` of whatever principal owns the configured
token. A real bot token (``xoxb-…``) resolves to the app's bot user and the
response carries a ``bot_id``; a user/legacy token (``xoxp-…``) resolves to the
installing *human's* member ID with **no** ``bot_id``. In the latter case the
adapter binds its identity to a human's member ID, so that person's ``<@…>``
mentions are misrouted as mentions of the bot. ``_warn_if_not_bot_token``
detects the missing ``bot_id`` at connect time — the only point where this is
observable, since a user token still sends/receives without any runtime error —
and logs an actionable, warning-only nudge.
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


class _DictAuthResponse(dict):
    """Mimics slack_sdk's AsyncSlackResponse — dict-like with .get(), like the
    real object the adapter already calls ``.get()`` on in ``connect``."""


class _AttrAuthResponse:
    """A response shape that is NOT dict-like; values live on ``.data``."""

    def __init__(self, data):
        self.data = data


def _make_adapter():
    # object.__new__ skips __init__ (heavy setup) — established slack-test pattern.
    return object.__new__(SlackAdapter)


def test_warns_when_bot_id_absent(caplog):
    # User token: auth.test resolves a human member but carries no bot_id.
    adapter = _make_adapter()
    resp = _DictAuthResponse(team_id="T1", user_id="U_HUMAN", user="trevor")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_not_bot_token(resp, "Acme")
    matched = [r for r in caplog.records
               if "authenticated as a USER" in r.message and "U_HUMAN" in r.message]
    assert matched


def test_no_warning_when_bot_id_present(caplog):
    # Real bot token: auth.test carries a bot_id.
    adapter = _make_adapter()
    resp = _DictAuthResponse(team_id="T1", user_id="U_BOT", bot_id="B123", user="hermes")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_not_bot_token(resp, "Acme")
    assert not any("authenticated as a USER" in r.message for r in caplog.records)


def test_no_warning_when_user_id_unresolved(caplog):
    # Nothing resolved (e.g. odd/empty response) — don't guess, stay silent.
    adapter = _make_adapter()
    resp = _DictAuthResponse(team_id="T1")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_not_bot_token(resp, "Acme")
    assert not any("authenticated as a USER" in r.message for r in caplog.records)


def test_warns_only_once_per_workspace(caplog):
    adapter = _make_adapter()
    resp = _DictAuthResponse(user_id="U_HUMAN")
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_not_bot_token(resp, "Acme")
        adapter._warn_if_not_bot_token(resp, "Acme")
    warnings = [r for r in caplog.records if "authenticated as a USER" in r.message]
    assert len(warnings) == 1


def test_handles_attribute_only_response_shape(caplog):
    # Response without dict .get(): values must be read off .data.
    adapter = _make_adapter()
    resp = _AttrAuthResponse({"user_id": "U_HUMAN", "user": "trevor"})
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_not_bot_token(resp, "Acme")
    assert any("authenticated as a USER" in r.message and "U_HUMAN" in r.message
               for r in caplog.records)
