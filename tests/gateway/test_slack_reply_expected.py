"""The Slack adapter tells the gateway whether an admitted message was addressed to the bot.

``MessageEvent.reply_expected`` decides whether a bare silence marker may stand
(tests/gateway/test_gateway_silence_tokens.py). Only a message that opens by @mentioning someone
else, or a top-level message a free-response channel admitted unaddressed, may stay silent; a plain
reply in a thread the bot is part of keeps the visible fallback (#110952). Driven through the real
``_handle_slack_message`` so the admission wiring, not just the rule, is covered.
"""
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_slack_ignore_other_user_mentions import (  # noqa: F401 - fixtures
    BOT_USER_ID, CHANNEL_ID, OTHER_USER_ID, _redirect_cache, adapter,
)

THREAD = "1700000000.000010"


def _message(text, ts, **extra):
    return {"channel": CHANNEL_ID, "channel_type": "channel", "user": "U_HUMAN", "text": text, "ts": ts, **extra}


@pytest.mark.asyncio
@pytest.mark.parametrize("extra, event, expected", [
    ({}, _message("hi", "1.1", channel="D0001", channel_type="im"), True),
    ({}, _message(f"<@{BOT_USER_ID}> hi", "1.2"), True),
    ({}, _message("reaction:added:eyes", "1.3", thread_ts=THREAD, _hermes_force_process=True), True),
    ({}, _message("done?", "1.4", thread_ts=THREAD), None),
    ({}, _message(f"<@{OTHER_USER_ID}> can you check?", "1.5", thread_ts=THREAD), False),
    ({"free_response_channels": CHANNEL_ID}, _message("side chatter", "1.6"), False),
    ({"free_response_channels": CHANNEL_ID, "reply_in_thread": False}, _message("any update?", "1.7"), None),
    ({}, _message(f"<@{OTHER_USER_ID}> and <@{BOT_USER_ID}|hermes> both look", "1.8", thread_ts=THREAD), True),
], ids=["dm", "mention", "reaction", "thread-followup", "peer-addressed", "free-channel-top-level",
        "flat-channel-followup", "peer-led-but-mentions-bot"])
async def test_admitted_message_carries_whether_it_was_addressed(adapter, extra, event, expected):
    adapter.config.extra.update(extra)
    adapter._mentioned_threads.add(THREAD)
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="human")), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=False):
        await adapter._handle_slack_message(event)
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].reply_expected is expected
