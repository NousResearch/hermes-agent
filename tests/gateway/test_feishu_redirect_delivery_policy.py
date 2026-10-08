"""A fresh Feishu redirect anchor gets its own delivery-policy decision."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.platforms.base_thread_metadata import _thread_metadata_for_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from gateway.turn_context import TurnContext
from tests.gateway.test_busy_redirect_anchor import Receiver
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [True, False])
async def test_redirect_replaces_event_progress_status_and_stream_policy_state(accept):
    runner = GatewayRunner(config=GatewayConfig())
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner.adapters[Platform.FEISHU] = adapter
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group", thread_id="omt_topic")
    opening = MessageEvent(text="first question", source=source, message_id="om_first")
    incoming = MessageEvent(text="new question", source=source, message_id="om_new")
    receiver = Receiver(accept=accept)
    context = TurnContext(source=source, session_key="key", event_message_id="om_first", inbound_message_id="om_first")
    runner._run_agent_bind_turn_wiring(context, TurnRunner(runner, context), source, "om_first", False,
                                       runner._event_thread_metadata(opening, source))
    turn = runner._session_state("key").turn
    turn.agent, turn.event, turn.ctx = receiver, opening, context
    old_state = context._status_thread_metadata["_feishu_topic_delivery"]
    terminal = SendResult(success=False, error="old topic anchor exhausted", retry_suppressed=True)
    old_state["terminal"] = terminal
    opening._delivery_retry_suppressed_result = terminal
    consumer = GatewayStreamConsumer(adapter, source.chat_id, metadata=context._status_thread_metadata)
    consumer._capture_retry_suppressed(terminal)
    context.stream_consumer_holder[0] = consumer

    redirected = runner._redirect_active_turn(receiver, incoming.text, "key", incoming)
    assert redirected is accept
    if not accept:
        assert context._status_thread_metadata["_feishu_topic_delivery"] is old_state
        assert opening._delivery_retry_suppressed_result is terminal
        assert consumer.retry_suppressed_result is terminal
        return

    new_state = _thread_metadata_for_event(incoming)["_feishu_topic_delivery"]
    assert new_state is not old_state
    assert _thread_metadata_for_event(opening)["_feishu_topic_delivery"] is new_state
    assert context._progress_metadata["_feishu_topic_delivery"] is new_state
    assert context._status_thread_metadata["_feishu_topic_delivery"] is new_state
    assert context._progress_metadata["reply_to_message_id"] == "om_new"
    assert context._status_thread_metadata["reply_to_message_id"] == "om_new"
    assert context._progress_reply_to == "om_new"
    assert getattr(opening, "_delivery_retry_suppressed_result", None) is None
    assert context.stream_consumer_holder[0] is None
    assert not consumer._run_still_current()
    assert consumer.retry_suppressed_result is terminal
    assert consumer.metadata["_feishu_topic_delivery"] is old_state
    assert old_state["terminal"] is terminal
    response = {"final_response": "new answer"}
    await runner._run_agent_mark_streamed_delivery(response, context)
    assert not response.get("delivery_retry_suppressed")
    assert not response.get("already_sent")
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    assert await runner._hmwa_deliver_turn_response(
        opening, source, SimpleNamespace(session_id="s1"), "key", 1,
        response, [], "new answer", None, False,
    ) == "new answer"


@pytest.mark.asyncio
async def test_inflight_old_stream_failure_cannot_poison_redirected_final():
    runner = GatewayRunner(config=GatewayConfig())
    adapter = ProgressCaptureAdapter(platform=Platform.FEISHU)
    runner.adapters[Platform.FEISHU] = adapter
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group", thread_id="omt_topic")
    opening = MessageEvent(text="first", source=source, message_id="om_first")
    incoming = MessageEvent(text="next", source=source, message_id="om_next")
    receiver = Receiver()
    ctx = TurnContext(source=source, session_key="key", event_message_id="om_first")
    runner._run_agent_bind_turn_wiring(ctx, TurnRunner(runner, ctx), source, "om_first", False,
                                       runner._event_thread_metadata(opening, source))
    turn = runner._session_state("key").turn
    turn.agent, turn.event, turn.ctx = receiver, opening, ctx
    consumer = GatewayStreamConsumer(adapter, source.chat_id, metadata=ctx._status_thread_metadata)
    ctx.stream_consumer_holder[0] = consumer
    entered, release = asyncio.Event(), asyncio.Event()
    terminal = SendResult(success=False, retry_suppressed=True, error="old anchor failed")

    async def late_send(*args, metadata=None, **kwargs):
        old_state = metadata["_feishu_topic_delivery"]
        entered.set()
        await release.wait()
        old_state["terminal"] = terminal
        return terminal

    adapter.send = late_send
    pending_send = asyncio.create_task(consumer._send_message(source.chat_id, "old preview", metadata=consumer.metadata))
    await entered.wait()
    assert runner._redirect_active_turn(receiver, incoming.text, "key", incoming)
    release.set()
    await pending_send

    assert consumer.retry_suppressed_result is terminal
    assert ctx.stream_consumer_holder[0] is None
    assert "terminal" not in ctx._status_thread_metadata["_feishu_topic_delivery"]
    response = {"final_response": "new final"}
    await runner._run_agent_mark_streamed_delivery(response, ctx)
    assert not response.get("delivery_retry_suppressed")
    assert not response.get("already_sent")
