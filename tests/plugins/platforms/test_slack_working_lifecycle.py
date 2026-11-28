"""Offline contracts for Slack lifecycle reactions on accepted human messages.

The adapter is real. Slack transports are mocked and no connection is opened.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import ProcessingOutcome
from plugins.platforms.slack import adapter as slack

# slack_sdk is an opt-in platform extra, outside the default [all] test environment.
AsyncWebClient = pytest.importorskip("slack_sdk.web.async_client").AsyncWebClient


CHANNEL = "C_TEST"
TEAM = "T_TEST"
ROOT = "1710000000.000000"
MESSAGE = "1710000001.000001"
METADATA = {"thread_id": ROOT, "team_id": TEAM, "message_id": MESSAGE}


@pytest.fixture
def adapter():
    config = PlatformConfig(
        enabled=True,
        extra={"free_response_channels": [CHANNEL]},
    )
    a = slack.SlackAdapter(config)
    client = AsyncMock(spec=AsyncWebClient)
    client.users_info.return_value = {
        "ok": True,
        "user": {"is_bot": False, "profile": {"display_name": "Human"}},
    }
    client.conversations_replies.return_value = {"ok": True, "messages": []}
    client.conversations_info.return_value = {
        "ok": True, "channel": {"name": "test-channel"},
    }
    a._app = SimpleNamespace(client=client)
    a._get_client = MagicMock(return_value=client)
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    a._fetch_thread_context = AsyncMock(return_value="")
    a._fetch_thread_parent_text = AsyncMock(return_value="")
    return a


def _message(**extra):
    return {
        "type": "message", "channel": CHANNEL, "channel_type": "channel",
        "team": TEAM, "user": "U_HUMAN", "client_msg_id": "human-message",
        "text": "Continue the task", "ts": MESSAGE, **extra,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["free_channel", "thread_followup", "free_mpim"])
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_accepted_human_gets_outcome_reactions(adapter, route, outcome):
    event = _message()
    if route == "thread_followup":
        adapter.config.extra.pop("free_response_channels")
        adapter._register_mentioned_thread(ROOT, team_id=TEAM)
        event["thread_ts"] = ROOT
    elif route == "free_mpim":
        event["channel_type"] = "mpim"

    await adapter._handle_slack_message(event)
    adapter.handle_message.assert_awaited_once()
    accepted = adapter.handle_message.await_args.args[0]
    client = adapter._get_client.return_value
    await adapter.on_processing_start(accepted)
    client.reactions_add.assert_awaited_once_with(
        channel=CHANNEL, timestamp=MESSAGE, name="eyes",
    )
    await adapter.on_processing_complete(accepted, outcome)
    client.reactions_remove.assert_awaited_once_with(
        channel=CHANNEL, timestamp=MESSAGE, name="eyes",
    )
    expected = [call(channel=CHANNEL, timestamp=MESSAGE, name="eyes")]
    final = {
        ProcessingOutcome.SUCCESS: "white_check_mark",
        ProcessingOutcome.FAILURE: "x",
    }.get(outcome)
    if final:
        expected.append(call(channel=CHANNEL, timestamp=MESSAGE, name=final))
    assert client.reactions_add.await_args_list == expected
    # A completion replay must not add a second final reaction.
    await adapter.on_processing_complete(accepted, outcome)
    assert client.reactions_add.await_args_list == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_extra,event_extra,sender_is_bot,authorized,accepted",
    [
        pytest.param({"ignored_channels": [CHANNEL]}, {}, False, True, False,
                     id="ignored_channel"),
        pytest.param({}, {}, False, False, False, id="unauthorized"),
        pytest.param({"allowed_channels": ["C_OTHER"]}, {}, False, True, False,
                     id="disallowed_channel"),
        pytest.param({"disable_dms": True}, {"channel_type": "im"}, False, True, False,
                     id="disabled_dm"),
        pytest.param({"reactions": False}, {}, False, True, True,
                     id="reactions_disabled"),
        pytest.param({"allow_bots": "all"},
                     {"user": "U_PEER_BOT", "bot_id": "B_PEER",
                      "text": "<@U_BOT> Continue the task"}, True, True, True,
                     id="allowed_declared_bot"),
        pytest.param({"allow_bots": "all"},
                     {"user": "U_PEER_BOT", "text": "<@U_BOT> Continue the task"},
                     True, True, True, id="allowed_bot_user"),
    ],
)
async def test_reaction_registration_preserves_guards(
    adapter, config_extra, event_extra, sender_is_bot, authorized, accepted,
):
    adapter.config.extra.update(config_extra)
    adapter.set_authorization_check(lambda *_, **__: authorized)  # the runner forwards thread_id/is_bot keywords
    client = adapter._get_client.return_value
    client.users_info.return_value["user"]["is_bot"] = sender_is_bot

    await adapter._handle_slack_message(_message(**event_extra))
    if accepted:
        adapter.handle_message.assert_awaited_once()
        processed = adapter.handle_message.await_args.args[0]
        await adapter.on_processing_start(processed)
        await adapter.on_processing_complete(processed, ProcessingOutcome.SUCCESS)
    else:
        adapter.handle_message.assert_not_awaited()
    client.reactions_add.assert_not_awaited()
    client.reactions_remove.assert_not_awaited()
