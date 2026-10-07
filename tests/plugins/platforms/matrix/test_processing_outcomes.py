"""Regression for #81440: Matrix must retire progress reactions on cancelled/refused turns."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, verdict", [
    (ProcessingOutcome.CANCELLED, None),
    (ProcessingOutcome.SUCCESS, "✅"),
    (ProcessingOutcome.FAILURE, "❌"),
])
async def test_completion_retires_progress_without_inventing_a_verdict(outcome, verdict):
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={"reactions": True}))
    adapter._reaction_redaction_delay_seconds = 0
    client = SimpleNamespace(
        send_message_event=AsyncMock(return_value="$eyes"),
        redact=AsyncMock(),
    )
    adapter._client = client
    room, message = "!room:example.org", "$message"
    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.MATRIX, chat_id=room, user_id="@sender:example.org"),
        message_id=message,
    )

    await adapter.on_processing_start(event)
    assert adapter._pending_reactions[(room, message)] == "$eyes"
    await adapter.on_processing_complete(event, outcome)
    await asyncio.wait_for(asyncio.gather(*tuple(adapter._reaction_redaction_tasks)), timeout=5)

    assert (room, message) not in adapter._pending_reactions
    client.redact.assert_awaited_once_with(room, "$eyes", reason="processing complete")
    relations = [call.args[2]["m.relates_to"] for call in client.send_message_event.await_args_list]
    assert [r["key"] for r in relations] == ["👀"] + ([verdict] if verdict else [])
    assert all(r["event_id"] == message for r in relations)
    assert all(call.args[0] == room for call in client.send_message_event.await_args_list)
