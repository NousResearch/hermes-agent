"""Invariant tests for content-free interactive gateway timing."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.interactive_timing import InteractiveTurnTiming, ensure_interactive_timing
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource, build_session_key


class _Clock:
    def __init__(self, *values: int):
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {}


class _BusyQueueHarness(GatewayBusySessionMixin):
    def __init__(self):
        self.adapter = SimpleNamespace(_pending_messages={})

    def _adapter_for_source(self, _source):
        return self.adapter

    def _peek_session_state(self, _session_key):
        return None


def test_interactive_timing_is_monotonic_machine_readable_and_content_free():
    timing = InteractiveTurnTiming(
        platform="slack",
        turn_id="turn-1",
        _wall_ns=_Clock(1_700_000_000_000_000_000),
        _monotonic_ns=_Clock(
            1_000_000_000, 1_120_000_000, 1_250_000_000, 1_900_000_000, 2_000_000_000),
    )
    timing.update_runtime(
        model="openai/gpt-test", provider="openai", reasoning={"effort": "low"},
        harness_revision="rev-1",
    )
    assert timing.mark("working_state", basis="typing_send_completed_proxy") is True
    assert timing.mark("working_state", basis="duplicate") is False
    timing.mark("first_meaningful_response", basis="server_send_completed_proxy")
    timing.complete("success")

    record = timing.to_record()
    encoded = json.dumps(record)

    assert record["milestones_ms"] == {
        "server_receipt": 0,
        "working_state": 120,
        "first_meaningful_response": 250,
        "completed": 900,
    }
    assert record["milestone_basis"]["first_meaningful_response"] == "server_send_completed_proxy"
    assert record["runtime"] == {
        "platform": "slack",
        "model": "openai/gpt-test",
        "provider": "openai",
        "reasoning": '{"effort":"low"}',
        "harness_revision": "rev-1",
        "lane": "unknown",
    }
    assert record["outcome"] == "success"
    assert "secret message" not in encoded
    assert "user_id" not in encoded and "chat_id" not in encoded and "session_id" not in encoded

    deferred = timing.fork_deferred_turn()
    deferred.mark("accepted", basis="queued")
    deferred_record = deferred.to_record()
    assert deferred_record["turn_id"] != record["turn_id"]
    assert deferred_record["received_at"] == record["received_at"]
    assert deferred_record["milestones_ms"] == {"server_receipt": 0, "accepted": 1000}
    assert deferred_record["outcome"] == "unknown"


@pytest.mark.asyncio
async def test_adapter_records_receipt_working_response_and_completion(caplog):
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=True), Platform.SLACK,
    )
    adapter.send_typing = AsyncMock(return_value=None)
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="sent"))

    async def _handler(_event):
        await asyncio.sleep(0.02)
        return "a useful response"

    adapter._message_handler = _handler
    source = SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="dm")
    event = MessageEvent(text="private request body", source=source, message_id="m1")
    session_key = build_session_key(source)

    with caplog.at_level("INFO", logger="gateway.interactive_timing"):
        await adapter.handle_message(event)
        task = adapter._session_tasks[session_key]
        await task

    record = event._interactive_timing.to_record()
    offsets = record["milestones_ms"]
    assert event._gateway_accepted is True
    assert offsets["server_receipt"] == 0
    assert offsets["accepted"] <= offsets["working_state"] <= offsets["first_meaningful_response"]
    assert offsets["first_meaningful_response"] <= offsets["completed"]
    assert record["outcome"] == "success"
    assert record["runtime"]["platform"] == "slack"
    emitted = next(message for message in caplog.messages if message.startswith("interactive_turn_timing "))
    payload = json.loads(emitted.removeprefix("interactive_turn_timing "))
    assert payload == record
    assert "private request body" not in emitted

    # A platform inheriting the no-op typing method must not claim a visible working state.
    unsupported = _StubAdapter(
        PlatformConfig(enabled=True, token="t", typing_indicator=True), Platform.WEBHOOK,
    )
    unsupported_event = MessageEvent(text="hidden", source=SessionSource(
        platform=Platform.WEBHOOK, chat_id="hook", chat_type="dm"))
    unsupported_timing = ensure_interactive_timing(unsupported_event, platform=Platform.WEBHOOK)
    stop_event = asyncio.Event()
    typing_task = unsupported._start_typing_refresh(unsupported_event, stop_event, None)
    await asyncio.sleep(0)
    stop_event.set()
    await typing_task
    assert "working_state" not in unsupported_timing.to_record()["milestones_ms"]

    # A successfully delivered failure notice is still the first meaningful response.
    error_event = MessageEvent(text="hidden", source=source, message_id="m2")
    ensure_interactive_timing(error_event, platform=Platform.SLACK)
    await adapter._notify_turn_error(error_event, RuntimeError("boom"))
    error_record = error_event._interactive_timing.to_record()
    assert error_record["milestone_basis"]["first_meaningful_response"] == (
        "error_send_completed_proxy"
    )

    # Synthetic /queue and failed-/steer turns preserve receipt time and mark admission in FIFO.
    for command, handler_name in (
        ("/queue finish this", "_busy_queue_command"),
        ("/steer finish this", "_busy_steer_command"),
    ):
        harness = _BusyQueueHarness()
        queued_source = SessionSource(platform=Platform.SLACK, chat_id="C456", chat_type="dm")
        command_event = MessageEvent(text=command, source=queued_source, message_id="queued")
        command_event._interactive_timing = InteractiveTurnTiming(
            platform="slack",
            _wall_ns=_Clock(1_700_000_000_000_000_000),
            _monotonic_ns=_Clock(1_000_000_000, 1_250_000_000),
        )
        await getattr(harness, handler_name)(command_event, "session-key", queued_source)
        queued_event = harness.adapter._pending_messages["session-key"]
        queued_record = queued_event._interactive_timing.to_record()
        assert queued_record["received_at"] == command_event._interactive_timing.to_record()["received_at"]
        assert queued_record["milestones_ms"]["accepted"] == 250
        assert queued_record["milestone_basis"]["accepted"] == "busy_fifo_enqueue"
