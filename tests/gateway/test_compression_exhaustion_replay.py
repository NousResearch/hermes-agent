"""Regression for #137565: safe compression exhaustion retries preserve the input."""

from types import SimpleNamespace

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


def test_safe_exhaustion_requeues_original_event_before_existing_followup():
    adapter = SimpleNamespace(_pending_messages={"key": _event()})
    original = _event()
    runner = _runner(adapter)

    accepted = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, original, "key", {"compression_exhausted": True, "messages": []}, "prepared request", ["photo.jpg"],
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
        runner, _event(), "key", {"compression_exhausted": True, "messages": [{"role": "tool"}]}, "request", [],
    )
    internal_result = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(internal=True), "key", {"compression_exhausted": True, "messages": []}, "request", [],
    )
    streamed_result = GatewayTurnMixin._hmwa_stage_compression_exhaustion_replay(
        runner, _event(), "key", {"compression_exhausted": True, "already_sent": True, "messages": []}, "request", [],
    )

    assert tool_result is False
    assert internal_result is False
    assert streamed_result is False
    assert adapter._pending_messages == {}
