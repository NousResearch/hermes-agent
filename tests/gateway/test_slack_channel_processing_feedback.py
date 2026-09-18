"""Slack channel feedback follows every accepted message through its lifecycle."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import ProcessingOutcome
from gateway.platforms.event import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


CHANNEL = "C0C24CKC6PL"
TEAM = "T_WORKSPACE"
MESSAGE_TS = "1700000000.000001"


def _adapter(*, reply_in_thread: bool, typing_indicator: bool = True) -> tuple[SlackAdapter, AsyncMock]:
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        typing_indicator=typing_indicator,
        extra={
            "free_response_channels": [CHANNEL],
            "working_message_channels": [CHANNEL],
            "reply_in_thread": reply_in_thread,
        },
    )
    adapter = SlackAdapter(config)
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "working.1"})
    client.chat_delete = AsyncMock(return_value={"ok": True})
    adapter._app = MagicMock()
    adapter._app.client = client
    adapter._team_clients[TEAM] = client
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids[TEAM] = "U_BOT"
    adapter._running = True
    return adapter, client


def _event(channel_type="channel") -> dict:
    return {
        "channel": CHANNEL,
        "channel_type": channel_type,
        "team": TEAM,
        "user": "U_USER",
        "text": "please reconcile the ledger",
        "ts": MESSAGE_TS,
    }


async def _deliver(adapter: SlackAdapter, outcome: ProcessingOutcome, event=None):
    captured = []

    async def process(event):
        captured.append(event)
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, outcome)

    adapter.handle_message = process
    with (
        patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="user")),
        patch.object(adapter, "_resolve_channel_name", new=AsyncMock(return_value="money")),
    ):
        await adapter._handle_slack_message(event or _event(), payload={"team_id": TEAM})
    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_free_response_channel_message_gets_eyes_then_success_reaction():
    adapter, client = _adapter(reply_in_thread=True)

    await _deliver(adapter, ProcessingOutcome.SUCCESS)

    assert [call.kwargs for call in client.reactions_add.await_args_list] == [
        {"channel": CHANNEL, "timestamp": MESSAGE_TS, "name": "eyes"},
        {"channel": CHANNEL, "timestamp": MESSAGE_TS, "name": "white_check_mark"},
    ]
    client.reactions_remove.assert_awaited_once_with(
        channel=CHANNEL, timestamp=MESSAGE_TS, name="eyes"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply_in_thread", "outcome"),
    [
        (True, ProcessingOutcome.SUCCESS),
        (False, ProcessingOutcome.FAILURE),
    ],
)
async def test_channel_working_message_is_single_scoped_and_deleted(
    reply_in_thread, outcome
):
    adapter, client = _adapter(reply_in_thread=reply_in_thread)

    await _deliver(adapter, outcome)

    client.chat_postMessage.assert_awaited_once()
    posted = client.chat_postMessage.await_args.kwargs
    assert posted["channel"] == CHANNEL
    assert posted["text"] == "Working…"
    if reply_in_thread:
        assert posted["thread_ts"] == MESSAGE_TS
    else:
        assert "thread_ts" not in posted
    client.chat_delete.assert_awaited_once_with(channel=CHANNEL, ts="working.1")


@pytest.mark.asyncio
async def test_unmentioned_free_response_mpim_remains_reaction_quiet():
    adapter, client = _adapter(reply_in_thread=True)

    await _deliver(adapter, ProcessingOutcome.SUCCESS, _event("mpim"))

    client.reactions_add.assert_not_awaited()
    client.reactions_remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_slash_command_does_not_post_public_working_message():
    adapter, client = _adapter(reply_in_thread=True)
    source = MagicMock(
        scope_id=TEAM,
        chat_id=CHANNEL,
        thread_id=MESSAGE_TS,
        chat_type="group",
    )
    event = MagicMock(
        source=source,
        raw_message={"channel": CHANNEL},
        message_type=MessageType.COMMAND,
        message_id=MESSAGE_TS,
    )

    await adapter._start_channel_working(event)

    client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_typing_indicator_opt_out_suppresses_working_message():
    adapter, client = _adapter(reply_in_thread=True, typing_indicator=False)
    source = MagicMock(
        scope_id=TEAM,
        chat_id=CHANNEL,
        thread_id=MESSAGE_TS,
        chat_type="group",
    )
    event = MagicMock(
        source=source,
        raw_message={"channel_type": "channel"},
        message_type=MessageType.TEXT,
        message_id=MESSAGE_TS,
    )

    await adapter._start_channel_working(event)

    client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthetic_reaction_turn_does_not_add_message_reactions():
    adapter, client = _adapter(reply_in_thread=True)
    event = _event()
    event["_hermes_reaction"] = {"reacted_to_ts": MESSAGE_TS}

    await _deliver(adapter, ProcessingOutcome.SUCCESS, event)

    client.reactions_add.assert_not_awaited()
    client.reactions_remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_working_cleanup_retries_before_forgetting_message():
    adapter, client = _adapter(reply_in_thread=True)
    client.chat_delete.side_effect = [ConnectionError("temporary"), {"ok": True}]

    event = await _deliver(adapter, ProcessingOutcome.SUCCESS)

    assert client.chat_delete.await_count == 2
    assert adapter._channel_working_key(event) not in adapter._channel_working_messages


@pytest.mark.asyncio
async def test_working_message_is_excluded_from_model_thread_context():
    adapter, _client = _adapter(reply_in_thread=True)
    adapter._channel_working_message_ts.add(
        adapter._workspace_message_marker(TEAM, "working.1")
    )
    messages = [
        {
            "ts": MESSAGE_TS,
            "text": "please reconcile the ledger",
            "user": "U_USER",
        },
        {
            "ts": "working.1",
            "text": "Working…",
            "user": "U_BOT",
            "bot_id": "B_BOT",
        },
    ]

    context, _parent = await adapter._format_thread_context(
        messages,
        thread_ts=MESSAGE_TS,
        current_ts="1700000000.000002",
        team_id=TEAM,
        channel_id=CHANNEL,
    )

    assert "Working" not in context


@pytest.mark.asyncio
async def test_cancelled_start_still_records_and_deletes_accepted_working_post():
    adapter, client = _adapter(reply_in_thread=True)
    accepted = asyncio.Event()
    release = asyncio.Event()

    async def accepted_then_return(**_kwargs):
        accepted.set()
        await release.wait()
        return {"ok": True, "ts": "working.cancelled"}

    client.chat_postMessage.side_effect = accepted_then_return
    source = MagicMock(
        scope_id=TEAM,
        chat_id=CHANNEL,
        thread_id=MESSAGE_TS,
        chat_type="group",
    )
    event = MagicMock(
        source=source,
        raw_message={"channel_type": "channel"},
        message_type=MessageType.TEXT,
        message_id=MESSAGE_TS,
    )

    start = asyncio.create_task(adapter._start_channel_working(event))
    await accepted.wait()
    start.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await start
    await adapter._stop_channel_working(event)

    client.chat_delete.assert_awaited_once_with(
        channel=CHANNEL, ts="working.cancelled")
