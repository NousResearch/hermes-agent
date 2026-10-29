"""A silence marker that ENDS a real reply is not delivered as text.

The model answers and then closes the same message with a marker line
(``"answer\\n\\nNO_REPLY"``). The whole-response silence filters only match a bare
marker, so the marker used to reach the chat on both delivery paths.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source
from tests.gateway.test_stream_consumer_silence import _make_adapter, _sent_and_edited
from tests.waiting import async_wait_until

ANSWER = "The capital of France is Paris."


@pytest.mark.asyncio
async def test_final_send_drops_a_marker_that_closes_the_reply(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    runner = _runner(monkeypatch, tmp_path)
    raw = f"{ANSWER}\n\nNO_REPLY"
    runner._run_agent = AsyncMock(return_value={
        "final_response": raw,
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": raw}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1)

    assert response == ANSWER


@pytest.mark.asyncio
async def test_stream_never_shows_a_trailing_marker_and_finalizes_on_the_reply():
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1))
    seen = []  # the buffer each interim edit decision was made on
    decide = consumer._should_edit

    def recording(tick):
        seen.append(consumer._accumulated)
        return decide(tick)

    consumer._should_edit = recording
    task = asyncio.create_task(consumer.run())
    # Each delta gets two edit decisions of its own (the edit interval has passed by the
    # second), so every interim frame is a chance to leak.
    for delta in (ANSWER, "\n\n", "NO", "_RE", "PLY"):
        consumer.on_delta(delta)
        await async_wait_until(lambda d=delta: sum(b.endswith(d) for b in seen) >= 2, timeout=5)
    consumer.finish()
    await asyncio.wait_for(task, 5)

    frames = _sent_and_edited(adapter)
    assert frames, "the reply was never shown"
    assert not [f for f in frames if "NO" in f], frames
    assert frames[-1] == ANSWER
    assert consumer.final_content_delivered is True
