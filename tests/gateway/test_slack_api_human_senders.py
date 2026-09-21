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


def test_bot_message_subtype_wins_over_allowlist():
    """``subtype: bot_message`` is bot-authored no matter what — allowlisting a user
    never admits classic bot posts, so ``xoxb`` traffic cannot loop back in."""
    adapter = _make_adapter({"api_human_users": HUMAN_ID})
    assert adapter._event_declares_bot_sender(_api_post(subtype="bot_message")) is True


def test_bot_profile_stamped_user_token_post_is_human():
    """Slack also stamps ``bot_id``/``bot_profile`` (not just ``app_id``) on some
    user-token posts — observed live: ``chat.postMessage`` with ``xoxp-`` delivered as
    ``{user: U_human, bot_id: B..., bot_profile: {...}, app_id: A..., no client_msg_id}``.
    The allowlist must be consulted before the bot short-circuit or those human posts
    are silently dropped. Loop safety holds: an app's own ``xoxb`` posts carry the
    *bot's* user id, which is never in a human allowlist."""
    adapter = _make_adapter({"api_human_users": HUMAN_ID})
    stamped = _api_post(bot_id="B_stamp", bot_profile={"id": "B_stamp", "name": "Front-end"})
    assert adapter._event_declares_bot_sender(stamped) is False
    # A non-allowlisted user with the same stamps stays bot-classified.
    assert adapter._event_declares_bot_sender(
        _api_post(user="U_stranger", bot_id="B_stamp")) is True
    # The bot's own posts (bot user id, or no user at all) stay bot-classified.
    assert adapter._event_declares_bot_sender(
        {"type": "message", "bot_id": "B_stamp", "app_id": "A_frontend", "text": "hi"}) is True
