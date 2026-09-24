"""A resolved clarify reply shares its original turn's processing lifecycle (#121653)."""

import asyncio


import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource
from tools import clarify_gateway


@pytest.fixture(autouse=True)
def _isolate_clarify_entries():
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()
    yield
    with clarify_gateway._lock:
        clarify_gateway._entries.clear()
        clarify_gateway._session_index.clear()


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.reactions = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}

    async def on_processing_start(self, event):
        self.reactions.append((event.message_id, "eyes"))

    async def on_processing_complete(self, event, outcome):
        self.reactions.append((event.message_id, outcome))


class _Inbound(GatewayInboundMixin):
    def __init__(self, adapter):
        self.adapter = adapter

    def _pending_event_audio_paths(self, event):
        return []

    async def _prepare_clarify_reply_text(self, event):
        return event.text

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _intake_adapter_for(self, source):
        return self.adapter


def _event(message_id, text):
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="private", user_id="user"),
        message_id=message_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_resolved_reply_reacts_until_original_turn_finishes(failure):
    adapter = _Adapter()
    inbound = _Inbound(adapter)
    original = _event("original", "question")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def process(event):
        entered.set()
        await finish.wait()
        if failure:
            raise RuntimeError("turn failed")
        return "done"

    adapter._message_handler = process
    key = adapter._source_session_key(original.source)
    task = asyncio.create_task(adapter._process_message_background(original, key))
    adapter._session_tasks[key] = task
    try:
        await entered.wait()
        clarify_gateway.register("clarify-reaction", key, "Your answer?", None)
        reply = _event("reply", "yes")
        assert await inbound._hm_clarify_reply(reply, reply.source, key) == ""
        assert ("reply", "eyes") in adapter.reactions
        assert not any(mid == "reply" and isinstance(value, ProcessingOutcome)
                       for mid, value in adapter.reactions)
        finish.set()
        await task
        outcome = ProcessingOutcome.FAILURE if failure else ProcessingOutcome.SUCCESS
        assert ("reply", outcome) in adapter.reactions
    finally:
        clarify_gateway.resolve_gateway_clarify("clarify-reaction", "")
        finish.set()
        if not task.done():
            await task


@pytest.mark.asyncio
async def test_invalid_clarify_reply_does_not_acknowledge():
    adapter = _Adapter()
    inbound = _Inbound(adapter)
    key = adapter._source_session_key(_event("original", "question").source)
    clarify_gateway.register("clarify-invalid", key, "Pick one", ["one", "two"])
    try:
        reply = _event("invalid", "99")
        assert await inbound._hm_clarify_reply(reply, reply.source, key) == ""
        assert adapter.reactions == []
    finally:
        clarify_gateway.resolve_gateway_clarify("clarify-invalid", "")


@pytest.mark.asyncio
async def test_fast_turn_cannot_finish_reply_before_start_reaction():
    adapter = _Adapter()
    inbound = _Inbound(adapter)
    original = _event("original", "question")
    key = adapter._source_session_key(original.source)
    entered = asyncio.Event()
    finish = asyncio.Event()
    started = asyncio.Event()
    release_start = asyncio.Event()

    async def process(event):
        entered.set()
        await finish.wait()
        return "done"

    original_start = adapter.on_processing_start

    async def slow_start(event):
        if event.message_id == "reply":
            started.set()
            await release_start.wait()
        await original_start(event)

    adapter.on_processing_start = slow_start
    adapter._message_handler = process
    task = asyncio.create_task(adapter._process_message_background(original, key))
    adapter._session_tasks[key] = task
    reply_task = None
    try:
        await entered.wait()
        clarify_gateway.register("clarify-fast", key, "Answer?", None)
        reply = _event("reply", "yes")
        reply_task = asyncio.create_task(inbound._hm_clarify_reply(reply, reply.source, key))
        await started.wait()
        finish.set()
        await asyncio.sleep(0)
        assert ("reply", ProcessingOutcome.SUCCESS) not in adapter.reactions
        release_start.set()
        await reply_task
        await task
        assert adapter.reactions.index(("reply", "eyes")) < adapter.reactions.index(
            ("reply", ProcessingOutcome.SUCCESS))
    finally:
        clarify_gateway.resolve_gateway_clarify("clarify-fast", "")
        release_start.set()
        finish.set()
        if reply_task is not None and not reply_task.done():
            await reply_task
        if not task.done():
            await task
