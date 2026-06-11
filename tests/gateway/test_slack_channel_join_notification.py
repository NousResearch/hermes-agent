"""Tests for channel_join_notification — a configurable notice posted when
the bot is added to a Slack channel (member_joined_channel event).

Behavior contract:
  - No config → handler is a silent no-op (stock behavior preserved).
  - Only the bot's OWN join triggers the notice; other members joining are
    ignored (Slack fires member_joined_channel for every member).
  - The notice is never posted about the target channel itself (avoids a
    self-referential ping when the bot is first invited to the status
    channel).
  - Message template supports {channel_id}/{channel_ref}/{inviter_id}/
    {inviter_ref}; an invalid template falls back to the default rather
    than raising.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.slack import SlackAdapter


def make_adapter(cjn=None):
    config = PlatformConfig(
        enabled=True,
        token="***",
        channel_join_notification=cjn,
    )
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.send = AsyncMock(return_value=SendResult(success=True))
    return a


def make_event(user="U_BOT", channel="C_NEW", inviter="U_HUMAN", team="T1"):
    return {
        "type": "member_joined_channel",
        "user": user,
        "channel": channel,
        "channel_type": "C",
        "team": team,
        "inviter": inviter,
    }


@pytest.mark.asyncio
async def test_noop_without_config():
    a = make_adapter(cjn=None)
    await a._handle_member_joined_channel(make_event())
    a.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_posts_default_message_on_bot_join():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    await a._handle_member_joined_channel(make_event())
    a.send.assert_awaited_once()
    target, text = a.send.await_args.args
    assert target == "C_STATUS"
    assert "<#C_NEW>" in text
    assert "<@U_HUMAN>" in text


@pytest.mark.asyncio
async def test_ignores_other_members_joining():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    await a._handle_member_joined_channel(make_event(user="U_SOMEONE"))
    a.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_ignores_join_of_target_channel_itself():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    await a._handle_member_joined_channel(make_event(channel="C_STATUS"))
    a.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_template_placeholders():
    a = make_adapter(
        cjn={
            "channel": "C_STATUS",
            "message": "Added to {channel_ref} ({channel_id}) by {inviter_ref}",
        }
    )
    await a._handle_member_joined_channel(make_event())
    _, text = a.send.await_args.args
    assert text == "Added to <#C_NEW> (C_NEW) by <@U_HUMAN>"


@pytest.mark.asyncio
async def test_invalid_template_falls_back_to_default():
    a = make_adapter(
        cjn={"channel": "C_STATUS", "message": "bad {nonexistent_placeholder}"}
    )
    await a._handle_member_joined_channel(make_event())
    a.send.assert_awaited_once()
    _, text = a.send.await_args.args
    assert "<#C_NEW>" in text  # default template used


@pytest.mark.asyncio
async def test_missing_inviter_renders_someone():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    event = make_event()
    del event["inviter"]
    await a._handle_member_joined_channel(event)
    _, text = a.send.await_args.args
    assert "someone" in text


@pytest.mark.asyncio
async def test_multiworkspace_uses_team_bot_id():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    a._team_bot_user_ids = {"T2": "U_BOT_T2"}
    # Bot id in workspace T2 differs from the primary _bot_user_id
    await a._handle_member_joined_channel(
        make_event(user="U_BOT_T2", team="T2")
    )
    a.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_failure_is_swallowed():
    a = make_adapter(cjn={"channel": "C_STATUS"})
    a.send = AsyncMock(return_value=SendResult(success=False, error="boom"))
    # Must not raise
    await a._handle_member_joined_channel(make_event())
