"""The Slack adapter stamps ``reply_expected`` on the event it hands the gateway.

A message admitted only because of a free-response channel, a thread follow-up, or a mention of
someone else carries ``reply_expected=False``; a 1:1 DM, an @mention of this bot, or a command
carries ``True``. The gateway lets a bare silence marker stand on ``False`` and keeps the visible
fallback on ``True`` (see tests/gateway/test_gateway_silence_tokens.py).
"""
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_slack_mention import _make_adapter


async def _event(adapter, **kw):
    # _make_adapter builds the object without __init__; stub the network-backed resolvers only.
    adapter._resolve_user_name = AsyncMock(return_value="alice")
    adapter._resolve_channel_name = AsyncMock(return_value="general")
    adapter._humanize_user_mentions = AsyncMock(side_effect=lambda text, **_: text)
    adapter._channel_prompt_with_identity = lambda *_a, **_k: None
    base = dict(
        event={"user": "U1", "ts": "1.0", "channel": "C1", "text": "hi"}, text="hi", original_text="hi",
        command_probe_text="hi", is_command_text=False, channel_id="C1", team_id="T1", ts="1.0",
        user_id="U1", thread_ts=None, is_dm=False, media_urls=[], media_types=[],
        media_text_inlined=[], channel_context=None,
    )
    base.update(kw)
    return await adapter._build_message_event(**base)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_expected", [True, False, None])
async def test_build_message_event_carries_reply_expected(reply_expected):
    adapter = _make_adapter()
    ev = await _event(adapter, reply_expected=reply_expected)
    assert ev.reply_expected is reply_expected


def test_reply_expected_rule_matches_addressing():
    """The value the inbound handler computes: DM, mention of this bot, or a command."""
    from plugins.platforms.slack.adapter import slack_reply_expected
    assert slack_reply_expected(is_one_to_one_dm=True, is_mentioned=False, is_command_text=False) is True
    assert slack_reply_expected(is_one_to_one_dm=False, is_mentioned=True, is_command_text=False) is True
    assert slack_reply_expected(is_one_to_one_dm=False, is_mentioned=False, is_command_text=True) is True
    # Admitted via free channel / thread follow-up / peer mention with ignore_other_user_mentions off.
    assert slack_reply_expected(is_one_to_one_dm=False, is_mentioned=False, is_command_text=False) is False
