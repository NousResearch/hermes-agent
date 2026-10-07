"""Tests for the Slack ``api_human_users`` allowlist.

A message posted through the Web API with a *user* token (``xoxp-``) is
authored by a real person, but it arrives with the posting ``app_id`` and no
``client_msg_id`` — the #35777 app/bot signature — so
``_event_declares_bot_sender`` drops it. ``platforms.slack.extra.api_human_users``
allowlists those *users* (never apps: an app's own ``xoxb`` bot posts carry
the same user+app_id shape).
"""

import sys
from unittest.mock import MagicMock

import pytest


# Mock slack-bolt / slack-sdk the same way test_slack_mention.py does.
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
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

from gateway.config import Platform, PlatformConfig  # noqa: E402


HUMAN_ID = "U_human"


def _make_adapter(extra=None):
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra or {}))
    return adapter


def _api_post(**overrides):
    """A user-token chat.postMessage as delivered over Socket Mode:
    real ``user``, app_id stamp, no ``client_msg_id``."""
    event = {"type": "message", "user": HUMAN_ID, "app_id": "A_frontend", "text": "hi"}
    event.update(overrides)
    return event


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SLACK_API_HUMAN_USERS", raising=False)


def test_api_post_is_bot_by_default():
    assert _make_adapter()._event_declares_bot_sender(_api_post()) is True


def test_allowlisted_user_api_post_is_human():
    adapter = _make_adapter({"api_human_users": ["U_other", HUMAN_ID]})
    assert adapter._event_declares_bot_sender(_api_post()) is False
    # Same predicate everywhere: no other user, and no user-less app post, rides it.
    assert adapter._event_declares_bot_sender(_api_post(user="U_stranger")) is True
    assert adapter._event_declares_bot_sender({"app_id": "A_frontend", "text": "hi"}) is True


def test_bot_markers_win_over_allowlist():
    """Allowlisting a user never admits genuine bot posts, so the app's own
    ``xoxb`` traffic (bot_id / subtype=bot_message) cannot loop back in."""
    adapter = _make_adapter({"api_human_users": HUMAN_ID})
    assert adapter._event_declares_bot_sender(_api_post(subtype="bot_message")) is True
    assert adapter._event_declares_bot_sender(_api_post(bot_id="B_stamp")) is True
