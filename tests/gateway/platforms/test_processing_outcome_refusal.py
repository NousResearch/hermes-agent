"""Regression for #81440: a refused inbound message must not score as a successful turn.

The runner's admission gate returns ``None`` for an unauthorized sender. The base turn wrapper
computed ``processing_ok = not bool(response)`` for that, so reaction adapters swapped the 👀 for
✅ on a message nothing handled. A refused event now ends as CANCELLED (in-progress reaction
cleared, no verdict); a deliberately silent turn still ends as SUCCESS.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource, build_session_key


class _OutcomeAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.DISCORD)
        self.outcomes: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        self.outcomes.append(outcome)


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    await (stop_event.wait() if stop_event is not None else asyncio.Event().wait())


async def _run(handler) -> list:
    adapter = _OutcomeAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(handler)
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="111", chat_type="group", user_id="42"),
        message_id="m1",
    )
    await adapter._process_message_background(event, build_session_key(event.source))
    return adapter.outcomes


@pytest.mark.asyncio
async def test_refused_event_is_not_scored_success():
    async def refuse(event):
        event._hermes_refused = True
        return None

    assert await _run(refuse) == [ProcessingOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_unauthorized_sender_at_admission_gate_scores_cancelled():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = None
    runner._is_user_authorized_for_source = lambda source: False

    async def admit(event):
        assert await runner._hm_admit_event(event) is None
        return None

    assert await _run(admit) == [ProcessingOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_silent_turn_still_scores_success():
    async def silent(_event):
        return None

    assert await _run(silent) == [ProcessingOutcome.SUCCESS]
