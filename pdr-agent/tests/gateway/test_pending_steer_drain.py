"""Regression for #131644: late steering retains FIFO messages and channel inputs."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext
from tests.gateway.test_internal_event_pin_wiring import (
    KEY,
    _capture,
    _human_source,
    _make_runner,
)


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["alone", "internal", "queued", "interrupt"])
async def test_accepted_steer_and_events_drain_once_in_order(kind):
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="test", user_id="human", chat_type="dm"
    )
    key = build_session_key(source)
    events = []
    if kind in ("internal", "queued"):
        events = [
            MessageEvent(
                text=f"event {i}",
                source=source,
                message_id=f"id-{i}",
                internal=kind == "internal",
                channel_prompt="pinned",
                metadata={"sentinel": i},
            )
            for i in range(3)
        ]
        for event in events:
            runner._enqueue_fifo(key, event, adapter)
    result = {"final_response": "done", "pending_steer": "accepted correction"}
    expected = ["accepted correction", *[event.text for event in events]]
    if kind == "interrupt":
        result.update(interrupted=True, interrupt_message="replacement request")
        expected.insert(0, "replacement request")
    delivered, delivered_events = [], []
    for _ in range(6):
        event, text = await runner._run_agent_drain_pending(
            result, adapter, source, key
        )
        if not event and not text:
            break
        delivered.append(text)
        if event is not None:
            delivered_events.append(event)
        result = {"final_response": "done"}
    assert delivered == expected
    if events:
        assert delivered_events == events
        for original, actual in zip(events, delivered_events):
            assert original is actual
            assert actual.channel_prompt == "pinned"
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
async def test_interrupt_then_steer_preserves_channel_inputs(monkeypatch):
    runner = _make_runner(monkeypatch)
    calls = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="correction"
    )
    runner._persist_prompt_pins = AsyncMock()
    runner._session_key_for_source = lambda source: KEY
    source = _human_source()
    source.parent_chat_id = "parent-channel"
    prompt = "Keep this channel instruction."
    runner._pinned_channel_inputs(KEY, prompt, source, internal=False)
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    ctx = TurnContext(
        source=source,
        context_prompt="context",
        channel_prompt=prompt,
        session_key=KEY,
        session_id="sess-wiring",
        run_generation=1,
        history=[],
    )
    result = dict(
        final_response="interrupted",
        messages=[],
        interrupted=True,
        interrupt_message="replacement",
        pending_steer="correction",
    )
    event, text = await runner._run_agent_drain_pending(result, adapter, source, KEY)
    assert text == "replacement" and event is None
    await runner._run_agent_queued_followup(
        ctx, adapter, text, event, result, result, None
    )
    assert calls[-1]["channel_prompt"] == prompt

    result = dict(final_response="done", messages=[])
    event, text = await runner._run_agent_drain_pending(result, adapter, source, KEY)
    assert text == "correction" and event is not None
    assert (
        not event.internal
    )  # A user correction must not become a background notification.
    await runner._run_agent_queued_followup(
        ctx, adapter, text, event, result, result, None
    )
    assert calls[-1]["channel_prompt"] == prompt
    assert calls[-1]["source"].parent_chat_id == source.parent_chat_id
    assert runner._peek_session_state(KEY).conversation.channel_pin == (
        prompt,
        source.parent_chat_id,
    )
    assert runner._queue_depth(KEY, adapter=adapter) == 0
