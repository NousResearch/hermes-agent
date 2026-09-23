"""Accepted busy follow-ups re-anchor both final replies and streaming answers.

Covers steering and redirect siblings; rejected or displaced follow-ups must
leave the running turn alone. Existing preview messages cannot be re-threaded
by editing them, so the answer must start a new message after earlier progress.
"""

import asyncio
import re

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_stream_consumer_draft import _make_draft_capable_adapter

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class Receiver:
    _supports_active_turn_redirect = True

    def __init__(self, accept=True):
        self.accept = accept

    def redirect(self, text):
        return self.accept

    def steer(self, text):
        return self.accept

    def interrupt(self, text):
        self.interrupted = text


def _running_turn(runner, key, receiver):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    opening = MessageEvent(text="What is the weather in Shanghai?", source=source, message_id="A")
    ctx = TurnContext(session_key=key, event_message_id="A", inbound_message_id="A")
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.ctx = receiver, opening, ctx
    redirecting = MessageEvent(text="What day is tomorrow?", source=source, message_id="B", reply_to_message_id="quoted-task")
    return opening, ctx, redirecting, source


async def _follow_up(runner, route, event, source, receiver, key):
    if route == "priority":
        await runner._hm_busy_interrupt(event, source, receiver, key)
    elif route == "priority_steer":
        runner._hm_busy_steer(event, receiver, key)
    elif route == "command":
        event.text = "/steer " + event.text
        await runner._busy_steer_command(event, key, source)
    else:
        mode = "steer" if route == "busy_steer" else "interrupt"
        await runner._resolve_busy_steer_or_redirect(event, key, mode, receiver)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy_interrupt", "priority", "busy_steer", "priority_steer", "command"])
@pytest.mark.parametrize("transport", [None, "edit", "draft"])
@pytest.mark.parametrize("progress", ["", "Earlier progress", "Earlier progress " * 350], ids=["no-preview", "visible-preview", "oversized-buffer"])
async def test_accepted_followup_owns_final_reply_and_stream_attachment(route, transport, progress):
    runner = GatewayRunner(config=GatewayConfig())
    receiver = Receiver()
    opening, ctx, redirecting, source = _running_turn(runner, "key", receiver)
    consumer = task = None
    if transport:
        adapter = _make_draft_capable_adapter()
        consumer = GatewayStreamConsumer(
            adapter, source.chat_id,
            StreamConsumerConfig(transport=transport, chat_type="dm", cursor="", buffer_threshold=1),
            initial_reply_to_id=opening.message_id,
            metadata={"reply_to_message_id": opening.message_id},
        )
        ctx.stream_consumer_holder[0] = consumer
        # An existing preview cannot have its reply attachment changed by an edit.
        visible = asyncio.Event()
        sender = adapter.send if transport == "edit" else adapter.send_draft
        original = sender
        async def observed_send(**kwargs):
            visible.set()
            if transport == "draft":
                return await original(**kwargs)
            return sender.return_value
        if transport == "edit":
            sender.side_effect = observed_send
        else:
            adapter.send_draft = observed_send
        if progress:
            consumer.on_delta(progress)
        if progress and len(progress) < adapter.MAX_MESSAGE_LENGTH:
            task = asyncio.create_task(consumer.run())
            await asyncio.wait_for(visible.wait(), 5)
    try:
        await _follow_up(runner, route, redirecting, source, receiver, "key")
        assert _reply_anchor_for_event(opening) == redirecting.message_id
        assert opening.ledger_message_id == redirecting.message_id
        assert (ctx.event_message_id, ctx.inbound_message_id) == (redirecting.message_id,) * 2
        assert opening.message_id == "A"
        if consumer:
            consumer.on_delta("BRAVO-892")
            consumer.finish("BRAVO-892")
            if task is None:
                task = asyncio.create_task(consumer.run())
            await asyncio.wait_for(task, 5)
            final = adapter.send.call_args.kwargs
            assert final["content"] == "BRAVO-892"
            earlier = [call.kwargs for call in adapter.send.call_args_list[:-1]]
            # Oversized sends add chunk labels, but must preserve the earlier progress.
            delivered = " ".join(re.sub(r"\(\d+/\d+\)", "", item["content"]) for item in earlier)
            assert delivered.split() == progress.split()
            if earlier:
                first = earlier[0]
                assert (first.get("reply_to") or first["metadata"]["reply_to_message_id"]) == opening.message_id
            assert (final.get("reply_to") or final["metadata"]["reply_to_message_id"]) == redirecting.message_id
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy_interrupt", "priority", "busy_steer", "priority_steer", "command"])
async def test_refused_or_foreign_followup_leaves_the_opening_anchor(route):
    runner = GatewayRunner(config=GatewayConfig())
    refusing = Receiver(accept=False)
    opening, ctx, redirecting, source = _running_turn(runner, "key", refusing)
    await _follow_up(runner, route, redirecting, source, refusing, "key")
    assert _reply_anchor_for_event(opening) == "A" and opening.ledger_message_id is None
    assert ctx.event_message_id == "A"

    # A redirect that lands on an agent which no longer owns the slot (a newer turn claimed it)
    # must not re-anchor the newer turn.
    displaced = Receiver()
    opening2, ctx2, redirecting2, source2 = _running_turn(runner, "key2", Receiver())
    if route == "command":
        return  # command resolves the current owner itself
    await _follow_up(runner, route, redirecting2, source2, displaced, "key2")
    assert _reply_anchor_for_event(opening2) == "A" and ctx2.event_message_id == "A"
