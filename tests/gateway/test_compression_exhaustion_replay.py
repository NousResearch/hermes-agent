"""Regression for #137565: safe compression exhaustion retries preserve the input."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource


def _event(*, internal=False):
    return MessageEvent(
        text="create the calendar event",
        internal=internal,
        message_id="in-137565",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user"),
    )


def _runner(adapter):
    runner = SimpleNamespace()
    state = SimpleNamespace(
        conversation=SimpleNamespace(queued_events=[]),
        persistent=SimpleNamespace(native_image_paths=[]),
    )
    runner._delivery_adapter_for = lambda _source: adapter
    runner._peek_session_state = lambda _key: state
    runner._session_state = lambda _key: state
    return runner



def _exhausted(**overrides):
    return {
        "compression_exhausted": True, "stream_visible_output": False,
        "compression_reset_session_id": "fresh", "messages": [], **overrides,
    }


def test_safe_exhaustion_requeues_original_event_before_existing_followup():
    adapter = SimpleNamespace(_pending_messages={"key": _event()})
    original = _event()
    runner = _runner(adapter)

    accepted = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, original, "key", _exhausted(), "prepared request", ["photo.jpg"],
    )

    assert accepted is True
    assert adapter._pending_messages["key"] is original
    assert runner._peek_session_state("key").conversation.queued_events[0].text == "create the calendar event"
    assert original._gateway_compression_replay_attempted is True
    assert original._gateway_compression_replay_message_text == "prepared request"
    assert original._gateway_compression_replay_native_image_paths == ["photo.jpg"]


def test_exhaustion_with_side_effect_evidence_or_internal_event_is_not_requeued():
    adapter = SimpleNamespace(_pending_messages={})
    runner = _runner(adapter)

    tool_result = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(), "key", _exhausted(messages=[{"role": "tool"}]), "request", [],
    )
    internal_result = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(internal=True), "key", _exhausted(), "request", [],
    )
    streamed_result = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(), "key", _exhausted(already_sent=True), "request", [],
    )

    assert tool_result is False
    assert internal_result is False
    assert streamed_result is False
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["none", "partial", "commentary", "draft"])
async def test_failed_turn_uses_actual_consumer_output_to_refuse_replay(delivery):
    from gateway.platforms.base import SendResult
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="partial"))
    adapter.send_draft = AsyncMock(return_value=SendResult(success=True))
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    if delivery == "partial":
        await consumer._send_or_edit("I have started checking the request")
        adapter.send.assert_awaited()
    elif delivery == "commentary":
        await consumer._send_commentary("I have started checking the request")
        adapter.send.assert_awaited()
    elif delivery == "draft":
        consumer._draft_id = 1
        await consumer._send_draft_frame("I have started checking the request")
        adapter.send_draft.assert_awaited()
    result = _exhausted(failed=True)
    ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=_event().source, session_key="key")
    runner = _runner(adapter)
    await GatewayTurnMixin._run_agent_mark_streamed_delivery(runner, result, ctx)
    assert not result.get("already_sent")  # failure still needs its error notice
    accepted = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(), "key", result, "request", [],
    )
    assert accepted is (delivery == "none")
    assert bool(adapter._pending_messages) is (delivery == "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_kind", ["fresh", "missing", "same", "deferred"])
async def test_handler_only_enqueues_after_confirmed_new_session(monkeypatch, tmp_path, reset_kind):
    from dataclasses import replace
    from gateway.platforms.base import SendResult
    from tests.gateway.test_duplicate_user_message import _bootstrap, _event as inbound, _source

    runner = _bootstrap(monkeypatch, tmp_path)
    event = inbound()
    adapter = SimpleNamespace(
        _pending_messages={}, send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
    )
    runner._delivery_adapter_for = lambda _source: adapter
    entry = runner.session_store.get_or_create_session.return_value
    runner._hmwa_defer_exhaustion_reset = lambda *_args: reset_kind == "deferred"
    runner.session_store.reset_session.return_value = (
        None if reset_kind == "missing" else
        replace(entry, session_id="fresh-session" if reset_kind == "fresh" else entry.session_id)
    )
    # Exercise the actual reset and handler wiring, isolating transport-specific bookkeeping.
    runner._evict_cached_agent = lambda _key: None
    runner._clear_conversation_scope = lambda *_a, **_kw: None
    runner._sync_telegram_topic_binding = lambda *_a, **_kw: None
    runner._run_agent = AsyncMock(return_value={
        "failed": True, "compression_exhausted": True, "messages": [],
        "stream_visible_output": False, "final_response": "context exhausted",
    })
    response = await runner._handle_message_with_agent(event, _source(), entry.session_key, 1)
    assert bool(adapter._pending_messages) is (reset_kind == "fresh")
    if reset_kind == "fresh":
        assert adapter._pending_messages[entry.session_key] is event
        assert response is None
    else:
        assert response is not None


@pytest.mark.parametrize("evidence", [None, "missing"])
def test_unknown_stream_delivery_evidence_fails_closed(evidence):
    adapter = SimpleNamespace(_pending_messages={})
    result = _exhausted(stream_visible_output=evidence)
    if evidence == "missing":
        del result["stream_visible_output"]
    assert GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        _runner(adapter), _event(), "key", result, "request", [],
    ) is False
    assert adapter._pending_messages == {}
