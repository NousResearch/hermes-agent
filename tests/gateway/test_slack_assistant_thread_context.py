"""Regression tests for Slack Assistant thread context hydration."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_slack_mock():
    """Wire up the minimal mocks required to import SlackAdapter."""
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter


def _make_adapter() -> SlackAdapter:
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    adapter._resolve_user_name = AsyncMock(side_effect=lambda user_id, chat_id="": user_id)
    adapter._fetch_thread_parent_text = AsyncMock(return_value="thread parent")
    return adapter


@pytest.mark.asyncio
async def test_assistant_thread_metadata_hydrates_context_without_top_level_thread_ts():
    """Assistant message events may omit event.thread_ts.

    The adapter still receives assistant_thread.thread_ts from Slack. That
    resolved thread id must be treated as the reply thread; otherwise the first
    real user follow-up in a visible Assistant thread is sent to the agent with
    no prior thread context, making it forget the request and files discussed in
    the thread.
    """
    adapter = _make_adapter()
    adapter.config.extra["require_mention"] = False
    adapter._fetch_thread_context = AsyncMock(return_value="[Thread context]\nGaurav: previous request\n")
    adapter._has_active_session_for_thread = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    event = {
        "type": "message",
        "channel": "C1",
        "channel_type": "channel",
        "team": "T1",
        "ts": "1001.000000",
        "user": "U123",
        "text": "Now create the video as requested",
        # Regression shape: Slack Assistant metadata has thread_ts, but the
        # message event itself does not.
        "assistant_thread": {
            "channel_id": "C1",
            "thread_ts": "1000.000000",
            "user_id": "U123",
            "team_id": "T1",
        },
    }

    await adapter._handle_slack_message(event)

    adapter._fetch_thread_context.assert_awaited_once_with(
        channel_id="C1",
        thread_ts="1000.000000",
        current_ts="1001.000000",
        team_id="T1",
    )
    adapter.handle_message.assert_awaited_once()
    msg_event = adapter.handle_message.await_args.args[0]
    assert msg_event.source.thread_id == "1000.000000"
    assert msg_event.reply_to_message_id == "1000.000000"
    assert msg_event.text.startswith("[Thread context]\nGaurav: previous request\n")


@pytest.mark.asyncio
async def test_assistant_thread_metadata_registers_mentioned_thread_without_event_thread_ts():
    """Mention tracking should use assistant_thread.thread_ts too."""
    adapter = _make_adapter()
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter.handle_message = AsyncMock()

    event = {
        "type": "message",
        "channel": "C1",
        "channel_type": "channel",
        "team": "T1",
        "ts": "1001.000000",
        "user": "U123",
        "text": "<@U_BOT> please continue",
        "assistant_thread": {
            "channel_id": "C1",
            "thread_ts": "1000.000000",
            "user_id": "U123",
            "team_id": "T1",
        },
    }

    await adapter._handle_slack_message(event)

    assert "1000.000000" in adapter._mentioned_threads



