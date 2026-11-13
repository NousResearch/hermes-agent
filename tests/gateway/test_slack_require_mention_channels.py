"""Tests for the Slack ``require_mention_channels`` per-channel override.

Channels listed here ALWAYS require an explicit bot @mention, even when
``require_mention`` is disabled globally or the channel would otherwise be
free-response — the opposite direction of ``free_response_channels`` (#13855).
Wake checks (bot-authored thread, previously mentioned thread, active session)
still apply, so ongoing conversations are not cut off.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


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


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config

BOT_USER_ID = "U_BOT"
CHANNEL_ID = "C_FORCED"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    for var in (
        "SLACK_REQUIRE_MENTION",
        "SLACK_REQUIRE_MENTION_CHANNELS",
        "SLACK_FREE_RESPONSE_CHANNELS",
        "SLACK_STRICT_MENTION",
        "SLACK_THREAD_REQUIRE_MENTION",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    a._bot_user_id = BOT_USER_ID
    a._running = True
    a.handle_message = AsyncMock()
    a._fetch_thread_context = AsyncMock(return_value="")
    a._fetch_thread_parent_text = AsyncMock(return_value="")
    a._has_active_session_for_thread = MagicMock(return_value=False)
    return a


def _event(text, ts="100.000", thread_ts=None, channel=CHANNEL_ID):
    event = {
        "type": "message",
        "channel": channel,
        "channel_type": "channel",
        "user": "U_HUMAN",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


# ---------------------------------------------------------------------------
# _slack_require_mention_channels() parsing
# ---------------------------------------------------------------------------


def _make(extra=None):
    a = object.__new__(SlackAdapter)
    a.config = PlatformConfig(enabled=True, extra=dict(extra or {}))
    return a


def test_require_mention_channels_csv_and_list():
    assert _make({"require_mention_channels": "C1, C2"})._slack_require_mention_channels() == {
        "C1",
        "C2",
    }
    assert _make({"require_mention_channels": ["C1", "C2"]})._slack_require_mention_channels() == {
        "C1",
        "C2",
    }


def test_yaml_bridge_sets_env(monkeypatch):
    monkeypatch.delenv("SLACK_REQUIRE_MENTION_CHANNELS", raising=False)
    _apply_yaml_config({}, {"require_mention_channels": ["C1", "C2"]})
    import os

    assert os.environ["SLACK_REQUIRE_MENTION_CHANNELS"] == "C1,C2"
    monkeypatch.delenv("SLACK_REQUIRE_MENTION_CHANNELS", raising=False)


# ---------------------------------------------------------------------------
# Routing behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forced_channel_wake_checks_still_apply(adapter):
    """A previously mentioned thread still auto-follows in a forced channel."""
    adapter.config.extra["require_mention"] = False
    adapter.config.extra["require_mention_channels"] = CHANNEL_ID
    adapter._mentioned_threads.add("100.000")

    await adapter._handle_slack_message(
        _event("follow-up", ts="101.000", thread_ts="100.000")
    )

    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_authentication_records_bot_id_alias(adapter):
    client = AsyncMock()
    client.auth_test.return_value = {
        "team_id": "T1",
        "team": "Example",
        "user_id": BOT_USER_ID,
        "bot_id": "B_BOT",
        "user": "testbot",
    }
    adapter._new_web_client = MagicMock(return_value=client)
    adapter._team_clients = {}
    adapter._team_bot_user_ids = {}
    adapter._team_bot_names = {}
    adapter._bot_user_id = None
    adapter._bot_display_name = None

    await adapter._authenticate_workspace("xoxb-test", None)

    assert adapter._team_bot_ids == {"T1": "B_BOT"}
    assert adapter._bot_id == "B_BOT"


@pytest.mark.asyncio
async def test_bot_id_mentioned_parent_wakes_followup_after_restart(adapter):
    adapter._team_bot_user_ids = {"T1": BOT_USER_ID}
    adapter._team_bot_ids = {"T1": "B_BOT"}
    adapter._channel_team[CHANNEL_ID] = "T1"
    adapter._fetch_thread_parent_text = AsyncMock(return_value="<@B_BOT> original ask")

    await adapter._handle_slack_message(
        _event("follow-up", ts="103.000", thread_ts="100.000")
    )

    adapter.handle_message.assert_called_once()


@pytest.mark.asyncio
async def test_other_workspace_bot_id_does_not_wake_channel_message(adapter):
    adapter._bot_id = "B_PRIMARY"
    adapter._team_bot_user_ids = {"T1": BOT_USER_ID}
    adapter._team_bot_ids = {"T1": "B_TEAM"}
    adapter._channel_team[CHANNEL_ID] = "T1"

    await adapter._handle_slack_message(_event("<@B_PRIMARY> not you", ts="102.500"))

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_bot_id_mention_wakes_channel_message(adapter):
    """Slack can encode an app mention with auth.test's bot_id instead of user_id."""
    adapter._team_bot_user_ids = {"T1": BOT_USER_ID}
    adapter._team_bot_ids = {"T1": "B_BOT"}
    adapter._channel_team[CHANNEL_ID] = "T1"

    await adapter._handle_slack_message(_event("<@B_BOT> help", ts="102.000"))

    adapter.handle_message.assert_called_once()
    assert adapter.handle_message.call_args.args[0].text == "help"


