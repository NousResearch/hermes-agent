"""Lifecycle and timeout-race coverage for Yuanbao bounded inbound work."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.yuanbao.adapter import (
    AccessPolicy,
    DispatchMiddleware,
    InboundContext,
    YuanbaoAdapter,
)
from plugins.platforms.yuanbao.inbound_capacity import (
    GROUP_GLOBAL_MAX_BYTES,
    GROUP_GLOBAL_MAX_MESSAGES,
    GROUP_QUEUE_MAX_MESSAGES,
)
from tests.waiting import async_wait_until


def _adapter() -> YuanbaoAdapter:
    return YuanbaoAdapter(
        PlatformConfig(
            enabled=True,
            extra={"app_id": "test", "app_secret": "test", "group_policy": "open"},
        )
    )


def _group_context(adapter: YuanbaoAdapter, source, index: int) -> InboundContext:
    return InboundContext(
        adapter=adapter,
        msg_id=f"message-{index}",
        chat_id="group:restart",
        chat_type="group",
        raw_text=f"message {index}",
        source=source,
        msg_type=MessageType.TEXT,
    )


def _valid_dm_frame(index: int) -> bytes:
    return json.dumps(
        {
            "CallbackCommand": "C2C.CallbackAfterSendMsg",
            "From_Account": "alice",
            "To_Account": "bot_123",
            "MsgBody": [
                {"MsgType": "TIMTextElem", "MsgContent": {"text": f"message {index}"}}
            ],
            "MsgKey": f"capacity-{index}",
        }
    ).encode()


def _authorized_adapter(monkeypatch) -> YuanbaoAdapter:
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _adapter()
    adapter._bot_id = "bot_123"
    adapter._access_policy = AccessPolicy(
        dm_policy="open", dm_allow_from=[], group_policy="open", group_allow_from=[]
    )
    adapter.handle_message = AsyncMock()
    adapter._resolve_inbound_media_urls = AsyncMock(return_value=([], []))
    adapter._connection._DEBOUNCE_WINDOW = 3600
    return adapter


async def _flush_frame(connection, frame: bytes) -> None:
    key = connection._extract_sender_key(frame)
    await async_wait_until(
        lambda: key in connection._inbound_admission.buffer,
        what="the admitted frame to reach its debounce buffer",
    )
    connection._flush_inbound_buffer(key)


@pytest.mark.asyncio
async def test_raw_admission_releases_capacity_after_real_dispatch(monkeypatch):
    adapter = _authorized_adapter(monkeypatch)
    connection = adapter._connection
    admission = connection._inbound_admission
    frame = _valid_dm_frame(1)

    try:
        assert await connection._push_to_inbound(frame)
        await _flush_frame(connection, frame)
        await async_wait_until(
            lambda: adapter.handle_message.await_count == 1,
            what="the valid authorized message to dispatch",
        )
        await async_wait_until(
            lambda: not admission.active_keys,
            what="the inbound pipeline batch to finish",
        )

        assert admission.retained_frames == 0
        assert admission.retained_bytes == 0
    finally:
        await admission.close()


@pytest.mark.asyncio
async def test_more_than_frame_capacity_processes_without_wedging(monkeypatch):
    adapter = _authorized_adapter(monkeypatch)
    connection = adapter._connection
    admission = connection._inbound_admission

    async def produce() -> None:
        for index in range(80):
            frame = _valid_dm_frame(index)
            assert await connection._push_to_inbound(frame)
            await _flush_frame(connection, frame)

    producer = asyncio.create_task(produce())
    try:
        await async_wait_until(
            producer.done,
            timeout=3,
            what="more than one full raw-admission window to process",
        )
        await producer
        await async_wait_until(
            lambda: adapter.handle_message.await_count == 80,
            what="all repeated valid messages to dispatch",
        )
        await async_wait_until(
            lambda: not admission.active_keys,
            what="all repeated pipeline batches to finish",
        )
        assert admission.retained_frames == 0
        assert admission.retained_bytes == 0
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        await admission.close()


@pytest.mark.asyncio
async def test_mutated_batch_releases_capacity_after_pipeline_exception():
    adapter = _adapter()
    connection = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    frame = b"exception-payload"

    class MutatingFailure:
        async def execute(self, ctx):
            ctx.raw_frames.clear()
            raise RuntimeError("expected test failure")

    adapter._inbound_pipeline = MutatingFailure()
    try:
        assert await connection._push_to_inbound(frame)
        await _flush_frame(connection, frame)
        await async_wait_until(
            lambda: not admission.active_keys,
            what="the failing pipeline batch to finish",
        )
        assert admission.retained_frames == 0
        assert admission.retained_bytes == 0
    finally:
        await admission.close()


@pytest.mark.asyncio
async def test_mutated_batch_releases_capacity_on_pipeline_cancellation():
    adapter = _adapter()
    connection = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    started = asyncio.Event()
    frame = b"cancellation-payload"

    class MutatingBlockedPipeline:
        async def execute(self, ctx):
            ctx.raw_frames.clear()
            started.set()
            await asyncio.Event().wait()

    adapter._inbound_pipeline = MutatingBlockedPipeline()
    try:
        assert await connection._push_to_inbound(frame)
        await _flush_frame(connection, frame)
        await started.wait()
        workers = list(admission.pipeline_tasks)
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        assert admission.retained_frames == 0
        assert admission.retained_bytes == 0
    finally:
        await admission.close()


@pytest.mark.asyncio
async def test_disconnect_clears_buffers_timers_and_fixed_workers():
    adapter = _adapter()
    connection = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    release = asyncio.Event()

    class BlockedPipeline:
        async def execute(self, _ctx):
            await release.wait()

    adapter._inbound_pipeline = BlockedPipeline()
    payload = json.dumps(
        {
            "From_Account": "owner",
            "GroupId": "cleanup",
            "MsgKey": "message",
            "MsgBody": [{"MsgType": "TIMTextElem", "MsgContent": {"text": "hello"}}],
        }
    ).encode()
    for index in range(12):
        assert await connection._push_to_inbound(payload + b" " * index)
    await async_wait_until(
        lambda: admission.retained_frames == 12,
        what="all cleanup frames to be retained",
    )

    await adapter.disconnect()

    assert admission.retained_frames == 0
    assert admission.retained_bytes == 0
    assert not admission.buffer
    assert not admission.timers
    assert admission.ingest_task is None
    assert not admission.pipeline_tasks
    assert not adapter._group_queues


@pytest.mark.asyncio
async def test_idle_group_consumer_exits_and_restarts_without_stranding():
    adapter = _adapter()
    seen = []
    adapter.handle_message = lambda event: _append_async(seen, event.message_id)
    source = adapter.build_source(
        chat_id="group:restart", chat_type="group", user_id="owner"
    )
    session_key = adapter._source_session_key(source)
    middleware = DispatchMiddleware()

    await middleware.handle(_group_context(adapter, source, 1), _noop)
    await async_wait_until(
        lambda: seen == ["message-1"], what="the first consumer dispatch"
    )
    await async_wait_until(
        lambda: session_key not in adapter._group_queues,
        timeout=4,
        what="the idle group consumer to exit",
    )

    await middleware.handle(_group_context(adapter, source, 2), _noop)
    await async_wait_until(
        lambda: seen == ["message-1", "message-2"],
        what="the restarted consumer dispatch",
    )
    for task in list(adapter._inbound_tasks):
        task.cancel()
    await asyncio.gather(*list(adapter._inbound_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_timeout_does_not_discard_a_boundary_arrival():
    adapter = _adapter()
    source = adapter.build_source(
        chat_id="group:boundary", chat_type="group", user_id="owner"
    )
    session_key = adapter._source_session_key(source)
    queue = asyncio.Queue(maxsize=GROUP_QUEUE_MAX_MESSAGES)
    adapter._group_dispatch.queues[session_key] = queue
    seen = asyncio.Event()

    async def handle_message(_event):
        seen.set()

    adapter.handle_message = handle_message
    event = MessageEvent(
        text="boundary", message_type=MessageType.TEXT, source=source, message_id="boundary"
    )
    real_wait_for = asyncio.wait_for
    raced = False

    async def racing_wait_for(awaitable, timeout):
        nonlocal raced
        if not raced:
            raced = True
            awaitable.close()
            queue.put_nowait((event, 8))
            adapter._group_dispatch.retained_messages = 1
            adapter._group_dispatch.retained_bytes = 8
            raise TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    with patch(
        "plugins.platforms.yuanbao.inbound_capacity.asyncio.wait_for",
        new=racing_wait_for,
    ):
        consumer = asyncio.create_task(
            DispatchMiddleware._consume_group_queue(adapter, session_key)
        )
        await seen.wait()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)

    assert adapter._group_dispatch.retained_messages == 0
    assert adapter._group_dispatch.retained_bytes == 0


@pytest.mark.asyncio
async def test_aggregate_group_message_and_byte_budgets():
    adapter = _adapter()
    adapter.send = _async_noop
    source = adapter.build_source(
        chat_id="group:aggregate", chat_type="group", user_id="owner"
    )
    event = MessageEvent(
        text="bounded", message_type=MessageType.TEXT, source=source, message_id="aggregate"
    )

    state = adapter._group_dispatch
    for index in range(GROUP_GLOBAL_MAX_MESSAGES):
        queue, _ = await state.admit(
            adapter, f"message-group-{index // GROUP_QUEUE_MAX_MESSAGES}", event, 1, "group:aggregate"
        )
        assert queue is not None
    rejected, _ = await state.admit(
        adapter, "message-overflow", event, 1, "group:aggregate"
    )
    assert rejected is None
    assert state.retained_messages == GROUP_GLOBAL_MAX_MESSAGES

    state.clear()
    payload_bytes = 700_000
    accepted = GROUP_GLOBAL_MAX_BYTES // payload_bytes
    for index in range(accepted):
        queue, _ = await state.admit(
            adapter, f"byte-group-{index}", event, payload_bytes, "group:aggregate"
        )
        assert queue is not None
    rejected, _ = await state.admit(
        adapter, "byte-overflow", event, payload_bytes, "group:aggregate"
    )
    assert rejected is None
    assert state.retained_bytes <= GROUP_GLOBAL_MAX_BYTES


async def _append_async(items, value) -> None:
    items.append(value)


async def _noop() -> None:
    return None


async def _async_noop(*_args, **_kwargs):
    return None
