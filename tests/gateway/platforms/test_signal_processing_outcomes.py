"""Regression for #81440: cancelled/refused turns must clear Signal's progress reaction."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.signal import SignalAdapter
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, verdict", [
    (ProcessingOutcome.CANCELLED, None),
    (ProcessingOutcome.SUCCESS, "✅"),
    (ProcessingOutcome.FAILURE, "❌"),
])
async def test_completion_clears_progress_and_only_adds_a_verdict_when_known(outcome, verdict):
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": "+15550000001"}))
    adapter._rpc = AsyncMock(return_value={})
    sender, timestamp = "+15550000002", 1700000000000
    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.SIGNAL, chat_id=sender, user_id=sender),
        raw_message={"sender": sender, "timestamp_ms": timestamp},
        message_id=str(timestamp),
    )

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, outcome)

    calls = adapter._rpc.await_args_list
    assert [call.args[0] for call in calls] == ["sendReaction"] * (3 if verdict else 2)
    params = [call.args[1] for call in calls]
    assert [p["emoji"] for p in params] == ["👀", ""] + ([verdict] if verdict else [])
    assert params[1]["remove"] is True
    assert all(p["targetAuthor"] == sender and p["targetTimestamp"] == timestamp for p in params)
    assert all(p["recipient"] == [sender] for p in params)
