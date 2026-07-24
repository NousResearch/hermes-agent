"""
Tests for the connect-time user-token (vs bot-token) nudge.

``auth.test`` returns the ``user_id`` of whatever principal owns the configured
token. A real bot token resolves to the app's bot user and the response carries
a ``bot_id``; a user/legacy token resolves to the installing human's member ID
with no ``bot_id``. The adapter must not trust that human ID as the bot identity.
``_warn_if_not_bot_token`` detects the missing ``bot_id`` at connect time and
logs an actionable warning while shared-channel mention gating stays closed.
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

from plugins.platforms.slack.adapter import (  # noqa: E402
    SlackAdapter,
    _trusted_bot_user_id,
)


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


def test_user_token_identity_is_not_bound_as_bot():
    resp = _DictAuthResponse(team_id="T1", user_id="U_HUMAN", user="trevor")
    assert _trusted_bot_user_id(resp) == ""


def test_bot_token_identity_is_trusted():
    resp = _DictAuthResponse(
        team_id="T1",
        user_id="U_BOT",
        bot_id="B123",
        user="hermes",
    )
    assert _trusted_bot_user_id(resp) == "U_BOT"


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
