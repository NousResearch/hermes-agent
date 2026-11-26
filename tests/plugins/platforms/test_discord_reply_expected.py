"""Free-response admission is permission to listen, not a demand to answer.

Exercise real Discord event construction and the gateway's silence policy; direct
addressing, commands and unknown reply authors retain the visible fallback.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.response_filters import silence_allowed
from tests.gateway.test_discord_free_response import (
    FakeDMChannel, FakeTextChannel, FakeThread, make_message,
)

pytest_plugins = ["tests.gateway.test_discord_free_response"]


@pytest.mark.asyncio
@pytest.mark.parametrize("options, expected", [
    pytest.param({}, False, id="free-channel"),
    pytest.param({"config": {"require_mention": False, "free_response_channels": []}},
                 False, id="global-free"),
    pytest.param({"config": {"free_response_channels": ["*"]}}, False, id="wildcard-free"),
    pytest.param({"thread": True}, False, id="parent-free-thread"),
    pytest.param({"voice": True, "config": {"free_response_channels": []}}, False, id="voice-linked"),
    pytest.param({"content": "<@999> are you awake?", "mentions": True}, True, id="mention"),
    pytest.param({"content": "<@999> are you awake?"}, True, id="raw-mention"),
    pytest.param({"content": "<@!999> are you awake?"}, True, id="legacy-mention"),
    pytest.param({"content": "/status"}, True, id="command"),
    pytest.param({"dm": True}, True, id="dm"),
    pytest.param({"reply_author": 999}, True, id="reply-to-bot"),
    pytest.param({"reply_author": 999, "cached": True}, True, id="cached-reply-to-bot"),
    pytest.param({"reply_author": 42}, False, id="reply-to-peer"),
    pytest.param({"reply_author": None}, None, id="unknown-reply"),
    pytest.param({"role_ping": 777, "bot_roles": [777]}, True, id="own-role-ping"),
    pytest.param({"role_ping": 777, "bot_roles": [777], "raw_role": True},
                 True, id="raw-role-ping"),
    pytest.param({"role_ping": 888, "bot_roles": [777]}, False, id="other-role-ping"),
    pytest.param({"role_ping": 777}, None, id="uncached-bot-member-role-ping"),
    pytest.param({"role_ping": 777, "bot_roles": [777], "member_id": 123},
                 None, id="wrong-member-role-ping"),
    pytest.param({"thread": True, "config": {"free_response_channels": []}}, None, id="joined-thread"),
])
async def test_admitted_message_carries_reply_expectation(adapter, options, expected):
    adapter.config.extra.update(auto_thread=False, history_backfill=False,
                                free_response_channels=["123"])
    adapter.config.extra.update(options.get("config", {}))
    channel = FakeTextChannel(channel_id=123)
    if options.get("thread"):
        channel = FakeThread(channel_id=456, parent=channel)
        await adapter._threads.mark_async("456")
    if options.get("dm"):
        channel = FakeDMChannel(channel_id=123)
    if options.get("voice"):
        adapter._voice_text_channels["guild"] = 123
    message = make_message(channel=channel, content=options.get("content", "clawd awake?"),
                           mentions=[adapter._client.user] if options.get("mentions") else [])
    if "role_ping" in options:
        message.content = f"<@&{options['role_ping']}> are you awake?"
        message.raw_role_mentions = [options["role_ping"]]
        message.role_mentions = [] if options.get("raw_role") else [SimpleNamespace(id=options["role_ping"])]
        member = None if "bot_roles" not in options else SimpleNamespace(
            id=options.get("member_id", 999),
            roles=[SimpleNamespace(id=role_id) for role_id in options["bot_roles"]])
        message.guild = SimpleNamespace(id=111, me=member)
    if "reply_author" in options:
        target = None if options["reply_author"] is None else SimpleNamespace(
            author=SimpleNamespace(id=options["reply_author"]),
            content="previous message", attachments=[])
        message.reference = SimpleNamespace(message_id=100,
                                            resolved=None if options.get("cached") else target,
                                            cached_message=target)
    assert await adapter._handle_message(message)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_expected is expected
    assert silence_allowed(None, event.reply_expected) is (expected is False)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True], ids=["live-batched", "recovered"])
async def test_expectation_is_set_before_batching_or_recovery_dispatch(adapter, recovered):
    adapter.config.extra.update(auto_thread=False, history_backfill=False,
                                free_response_channels=["123"])
    adapter._text_batch_delay_seconds = 1
    adapter._enqueue_text_event = Mock()
    message = make_message(channel=FakeTextChannel(channel_id=123), content="clawd awake?")
    assert await adapter._handle_message(message, recovered=recovered)
    if recovered:
        event = adapter.handle_message.await_args.args[0]
        adapter._enqueue_text_event.assert_not_called()
    else:
        event = adapter._enqueue_text_event.call_args.args[0]
        adapter.handle_message.assert_not_called()
    assert event.reply_expected is False
