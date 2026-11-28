"""``reactions: progress`` shows :eyes: while a turn runs and leaves no :white_check_mark:.

The default (``true``) swaps :eyes: for a check mark on every answered message, which piles up
in a busy channel. ``progress`` keeps the "seen, working" signal and the :x: on failure, and
drops only the success mark.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(reactions: str) -> SlackAdapter:
    a = SlackAdapter(PlatformConfig(enabled=True, token="***", extra={"reactions": reactions}))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={"user": {"is_bot": False, "profile": {"display_name": "Tyler"}}})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize("reactions, outcome_name, final_adds", [
    ("progress", "SUCCESS", []),
    ("progress", "FAILURE", ["x"]),
    ("true", "SUCCESS", ["white_check_mark"]),
])
async def test_final_reaction_per_mode(reactions, outcome_name, final_adds):
    adapter = _adapter(reactions)
    # A DM goes through the real intake, which is what starts tracking the message for reactions.
    await adapter._handle_slack_message({
        "text": "hello", "user": "U_USER", "channel": "D123",
        "channel_type": "im", "ts": "1234567890.000002"})
    event = MessageEvent(
        text="hello", message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id="D123", chat_type="dm", user_id="U_USER"),
        message_id="1234567890.000002")

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome[outcome_name])

    added = [c.kwargs["name"] for c in adapter._app.client.reactions_add.call_args_list]
    removed = [c.kwargs["name"] for c in adapter._app.client.reactions_remove.call_args_list]
    assert added == ["eyes", *final_adds]
    assert removed == ["eyes"]
