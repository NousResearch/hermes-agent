"""The inherited completion hook preserves opt-in reaction ordering."""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.platforms.event import MessageEvent, ProcessingOutcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, enabled, expected",
    [
        (ProcessingOutcome.SUCCESS, True, "ok"),
        (ProcessingOutcome.FAILURE, True, "failed"),
        (ProcessingOutcome.CANCELLED, True, None),
        (ProcessingOutcome.SUCCESS, False, None),
    ],
)
async def test_inherited_completion_replaces_only_enabled_terminal_reactions(
    outcome, enabled, expected
):
    adapter = BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server_url": "http://localhost:1234",
                "password": "test-secret",
            },
        )
    )
    adapter._OK_EMOJI = "ok"
    adapter._FAIL_EMOJI = "failed"
    adapter._reactions_enabled = lambda: enabled
    calls = []

    async def add(chat_id, message_id, emoji):
        calls.append(("add", chat_id, message_id, emoji))

    async def remove(chat_id, message_id):
        calls.append(("remove", chat_id, message_id))

    adapter._add_reaction = add
    adapter._remove_reaction = remove
    event = MessageEvent(
        text="hello", message_id="message-1", source=adapter.build_source("chat-1")
    )
    await adapter.on_processing_complete(event, outcome)
    expected_calls = [("remove", "chat-1", "message-1")] if enabled else []
    if expected is not None:
        expected_calls.append(("add", "chat-1", "message-1", expected))
    assert calls == expected_calls
