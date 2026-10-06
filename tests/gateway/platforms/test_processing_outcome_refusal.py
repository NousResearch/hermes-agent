"""Regression for #81440: a refused inbound message must not score as a successful turn.

The runner's admission gate returns ``None`` for an unauthorized sender. The base turn wrapper
computed ``processing_ok = not bool(response)`` for that, so reaction adapters swapped the 👀 for
✅ on a message nothing handled. A refused event now ends as CANCELLED (in-progress reaction
cleared, no verdict); a deliberately silent turn still ends as SUCCESS.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
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


@pytest.fixture
def runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner(config=GatewayConfig())
    yield runner
    runner.session_store.close_all_db_handles()
    runner.close_all_session_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_unauthorized_sender_at_admission_gate_scores_cancelled(runner, monkeypatch, rewrite):
    seen = []

    async def plugin_hook(name, **kwargs):
        assert name == "pre_gateway_dispatch"
        seen.append(kwargs["event"])
        return [{"action": "rewrite", "text": "rewritten"}] if rewrite else []

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", plugin_hook)

    assert await _run(runner._handle_message) == [ProcessingOutcome.CANCELLED]
    assert len(seen) == 1
    assert seen[0].text == "hello"


@pytest.mark.asyncio
async def test_busy_session_refusal_is_marked_without_queuing_or_completing(runner):
    adapter = _OutcomeAdapter()
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.DISCORD, chat_id="111", chat_type="group", user_id="42"),
        message_id="m2",
    )
    session_key = build_session_key(event.source)
    active = asyncio.Event()
    adapter._active_sessions[session_key] = active

    await adapter._handle_message_while_active(event, session_key)

    assert getattr(event, "_hermes_refused", False) is True
    assert not adapter._pending_messages
    assert adapter._active_sessions[session_key] is active
    assert not active.is_set()
    assert adapter.outcomes == []


@pytest.mark.asyncio
async def test_silent_turn_still_scores_success():
    async def silent(_event):
        return None

    assert await _run(silent) == [ProcessingOutcome.SUCCESS]
