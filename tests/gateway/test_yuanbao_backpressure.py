"""Regression coverage for bounded Yuanbao inbound admission (#145079)."""

from __future__ import annotations

import asyncio
import gc
import json
import tracemalloc
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.yuanbao.adapter import (
    ConnectionManager,
    CMD_TYPE,
    DispatchMiddleware,
    InboundContext,
    OwnerCommandMiddleware,
    YuanbaoAdapter,
)
from plugins.platforms.yuanbao.inbound_capacity import (
    DEBOUNCE_MAX_FRAMES_PER_BATCH,
    GROUP_GLOBAL_MAX_BYTES,
    GROUP_GLOBAL_MAX_MESSAGES,
    GROUP_MAX_ACTIVE_QUEUES,
    GROUP_OVERLOAD_NOTICE,
    INBOUND_MAX_RETAINED_BYTES,
    INBOUND_MAX_RETAINED_FRAMES,
)
from tests.waiting import async_wait_until


def _adapter() -> YuanbaoAdapter:
    return YuanbaoAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "app_id": "test",
                "app_secret": "test",
                "group_policy": "open",
            },
        )
    )


def _group_context(adapter: YuanbaoAdapter, source, index: int) -> InboundContext:
    return InboundContext(
        adapter=adapter,
        msg_id=f"message-{index}",
        chat_id="group:bounded",
        chat_type="group",
        raw_text=("Q" * 4000) + str(index),
        source=source,
        msg_type=MessageType.TEXT,
    )


@pytest.mark.asyncio
async def test_blocked_group_backlog_has_finite_capacity():
    adapter = _adapter()
    adapter.send = AsyncMock()
    release = asyncio.Event()
    started = asyncio.Event()
    processed = 0

    async def handler(_event):
        nonlocal processed
        processed += 1
        started.set()
        if processed == 1:
            await release.wait()

    adapter.set_message_handler(handler)
    source = adapter.build_source(
        chat_id="group:bounded",
        chat_name="bounded",
        chat_type="group",
        user_id="owner",
        user_name="Owner",
    )
    middleware = DispatchMiddleware()

    tracemalloc.start()
    gc.collect()
    baseline, _ = tracemalloc.get_traced_memory()
    for index in range(10_000):
        await middleware.handle(_group_context(adapter, source, index), _noop)
        if index % 64 == 0:
            await asyncio.sleep(0)
    await started.wait()
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()

    session_key = adapter._source_session_key(source)
    queue = adapter._group_queues[session_key]
    measurement = {
        "messages": 10_000,
        "processed": processed,
        "queue_maxsize": queue.maxsize,
        "queue_size": queue.qsize(),
        "active_consumers": sum(
            not task.done() and task.get_name().startswith("yuanbao-group-consumer-")
            for task in adapter._inbound_tasks
        ),
        "accepted": adapter._group_dispatch.accepted,
        "rejected": adapter._group_dispatch.rejected,
        "python_alloc_delta_mb": round((current - baseline) / 1024 / 1024, 2),
    }
    print("GROUP_FIXED=" + json.dumps(measurement, sort_keys=True))

    release.set()
    await async_wait_until(
        lambda: processed == measurement["accepted"],
        timeout=8,
        what="all accepted group messages to drain",
    )
    await async_wait_until(
        lambda: adapter._group_dispatch.retained_messages == 0,
        what="the retained group budget to be released",
    )
    assert queue.empty()
    for task in list(adapter._inbound_tasks):
        task.cancel()
    await asyncio.gather(*list(adapter._inbound_tasks), return_exceptions=True)
    for task in list(adapter._session_tasks.values()):
        task.cancel()
    await asyncio.gather(*list(adapter._session_tasks.values()), return_exceptions=True)
    tracemalloc.stop()

    assert queue.maxsize > 0
    assert queue.qsize() <= 32
    assert measurement["accepted"] == 33
    assert measurement["rejected"] == 10_000 - measurement["accepted"]
    adapter.send.assert_awaited_once_with("group:bounded", GROUP_OVERLOAD_NOTICE)


@pytest.mark.asyncio
async def test_continuous_debounce_has_finite_capacity():
    adapter = _adapter()
    connection: ConnectionManager = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    release = asyncio.Event()
    pipeline_started = asyncio.Event()

    class BlockedPipeline:
        async def execute(self, _ctx):
            pipeline_started.set()
            await release.wait()

    adapter._inbound_pipeline = BlockedPipeline()
    payload = json.dumps(
        {
            "From_Account": "owner",
            "GroupId": "bounded",
            "MsgKey": "message",
            "MsgBody": [{"MsgType": "TIMTextElem", "MsgContent": {"text": "Q" * 4000}}],
        }
    ).encode()

    tracemalloc.start()
    gc.collect()
    baseline, _ = tracemalloc.get_traced_memory()

    async def produce():
        for index in range(10_000):
            frame = payload.replace(b'"message"', f'"message-{index}"'.encode())
            assert await connection._push_to_inbound(frame)

    producer = asyncio.create_task(produce())
    await pipeline_started.wait()
    await async_wait_until(
        lambda: admission.retained_frames == INBOUND_MAX_RETAINED_FRAMES,
        what="the bounded raw admission budget to fill",
    )
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()

    measurement = {
        "frames": 10_000,
        "retained_frames": admission.retained_frames,
        "retained_payload_bytes": admission.retained_bytes,
        "buffered_frames": sum(map(len, admission.buffer.values())),
        "raw_queue_frames": admission.raw_queue.qsize(),
        "ready_batches": admission.ready.qsize(),
        "pending_batches": sum(map(len, admission.pending.values())),
        "pipeline_workers": len(admission.pipeline_tasks),
        "active_timers": len(admission.timers),
        "python_alloc_delta_mb": round((current - baseline) / 1024 / 1024, 2),
    }
    print("DEBOUNCE_FIXED=" + json.dumps(measurement, sort_keys=True))

    assert not producer.done()
    assert admission.retained_frames <= INBOUND_MAX_RETAINED_FRAMES
    assert admission.retained_bytes <= INBOUND_MAX_RETAINED_BYTES
    assert all(len(frames) <= DEBOUNCE_MAX_FRAMES_PER_BATCH for frames in admission.buffer.values())

    release.set()
    await async_wait_until(producer.done, what="the backpressured producer to drain")
    await producer
    await admission.close()
    tracemalloc.stop()

    assert admission.retained_frames == 0
    assert admission.retained_bytes == 0
    assert not admission.buffer
    assert not admission.timers


@pytest.mark.asyncio
async def test_raw_admission_enforces_byte_budget():
    adapter = _adapter()
    admission = adapter._connection._inbound_admission
    release = asyncio.Event()
    started = asyncio.Event()

    class BlockedPipeline:
        async def execute(self, _ctx):
            started.set()
            await release.wait()

    adapter._inbound_pipeline = BlockedPipeline()
    first = b"a" * (INBOUND_MAX_RETAINED_BYTES // 2)
    second = b"b" * (INBOUND_MAX_RETAINED_BYTES // 2)
    third = b"c"

    assert await admission.admit(first)
    assert await admission.admit(second)
    await started.wait()
    blocked = asyncio.create_task(admission.admit(third))
    await asyncio.sleep(0)

    assert admission.retained_bytes == INBOUND_MAX_RETAINED_BYTES
    assert not blocked.done()

    release.set()
    await async_wait_until(blocked.done, what="the byte-budget admission to resume")
    assert await blocked
    await admission.close()


@pytest.mark.asyncio
async def test_accepted_group_messages_preserve_fifo_and_drain():
    adapter = _adapter()
    adapter.send = AsyncMock()
    seen = []
    drained = asyncio.Event()

    async def handle_message(event):
        seen.append(event.message_id)
        if len(seen) == 32:
            drained.set()

    adapter.handle_message = handle_message
    source = adapter.build_source(
        chat_id="group:fifo",
        chat_type="group",
        user_id="owner",
        user_name="Owner",
    )
    middleware = DispatchMiddleware()
    for index in range(40):
        ctx = _group_context(adapter, source, index)
        ctx.chat_id = "group:fifo"
        await middleware.handle(ctx, _noop)

    await drained.wait()
    assert seen == [f"message-{index}" for index in range(32)]
    assert adapter._group_dispatch.accepted == 32
    assert adapter._group_dispatch.rejected == 8
    adapter.send.assert_awaited_once_with("group:fifo", GROUP_OVERLOAD_NOTICE)

    for task in list(adapter._inbound_tasks):
        task.cancel()
    await asyncio.gather(*list(adapter._inbound_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_group_payload_budget_rejects_before_message_capacity():
    adapter = _adapter()
    adapter.send = AsyncMock()
    source = adapter.build_source(
        chat_id="group:bytes", chat_type="group", user_id="owner"
    )
    middleware = DispatchMiddleware()
    first = _group_context(adapter, source, 1)
    first.chat_id = "group:bytes"
    first.raw_text = "a" * 700_000
    second = _group_context(adapter, source, 2)
    second.chat_id = "group:bytes"
    second.raw_text = "b" * 700_000

    await middleware.handle(first, _noop)
    await middleware.handle(second, _noop)

    assert adapter._group_dispatch.accepted == 1
    assert adapter._group_dispatch.rejected == 1
    adapter.send.assert_awaited_once_with("group:bytes", GROUP_OVERLOAD_NOTICE)
    for task in list(adapter._inbound_tasks):
        task.cancel()
    await asyncio.gather(*list(adapter._inbound_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_many_groups_have_aggregate_work_and_consumer_caps():
    adapter = _adapter()
    adapter.send = AsyncMock()
    release = asyncio.Event()

    async def handle_message(_event):
        await release.wait()

    adapter.handle_message = handle_message
    middleware = DispatchMiddleware()
    for index in range(GROUP_MAX_ACTIVE_QUEUES + 20):
        source = adapter.build_source(
            chat_id=f"group:{index}", chat_type="group", user_id="owner"
        )
        ctx = _group_context(adapter, source, index)
        ctx.chat_id = f"group:{index}"
        await middleware.handle(ctx, _noop)
        await asyncio.sleep(0)

    consumers = [
        task
        for task in adapter._inbound_tasks
        if not task.done() and task.get_name().startswith("yuanbao-group-consumer-")
    ]
    assert len(adapter._group_queues) == GROUP_MAX_ACTIVE_QUEUES
    assert len(consumers) == GROUP_MAX_ACTIVE_QUEUES
    assert adapter._group_dispatch.retained_messages <= GROUP_GLOBAL_MAX_MESSAGES
    assert adapter._group_dispatch.retained_bytes <= GROUP_GLOBAL_MAX_BYTES
    assert adapter._group_dispatch.rejected == 20

    release.set()
    for task in consumers:
        task.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)


@pytest.mark.asyncio
async def test_independent_group_consumers_do_not_block_each_other():
    adapter = _adapter()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_seen = asyncio.Event()

    async def handle_message(event):
        if event.source.chat_id == "group:first":
            first_started.set()
            await release_first.wait()
        else:
            second_seen.set()

    adapter.handle_message = handle_message
    middleware = DispatchMiddleware()
    first_source = adapter.build_source(
        chat_id="group:first", chat_type="group", user_id="owner"
    )
    second_source = adapter.build_source(
        chat_id="group:second", chat_type="group", user_id="owner"
    )

    first = _group_context(adapter, first_source, 1)
    first.chat_id = "group:first"
    second = _group_context(adapter, second_source, 2)
    second.chat_id = "group:second"
    await middleware.handle(first, _noop)
    await first_started.wait()
    await middleware.handle(second, _noop)
    await second_seen.wait()

    consumers = [
        task
        for task in adapter._inbound_tasks
        if not task.done() and task.get_name().startswith("yuanbao-group-consumer-")
    ]
    assert len(consumers) == 2
    release_first.set()
    for task in consumers:
        task.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)


@pytest.mark.asyncio
async def test_pipeline_serializes_senders_per_group_but_not_other_groups():
    adapter = _adapter()
    connection = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    other_started = asyncio.Event()
    seen = []

    class CapturePipeline:
        async def execute(self, ctx):
            message_id = json.loads(ctx.raw_frames[0])["MsgKey"]
            seen.append(message_id)
            if message_id == "first":
                first_started.set()
                await release_first.wait()
            elif message_id == "second":
                second_started.set()
            else:
                other_started.set()

    def frame(sender, group, message_id):
        return json.dumps(
            {
                "From_Account": sender,
                "GroupId": group,
                "MsgKey": message_id,
                "MsgBody": [
                    {"MsgType": "TIMTextElem", "MsgContent": {"text": message_id}}
                ],
            }
        ).encode()

    adapter._inbound_pipeline = CapturePipeline()
    first = frame("one", "shared", "first")
    second = frame("two", "shared", "second")
    other = frame("three", "independent", "other")
    for payload in (first, second, other):
        assert await connection._push_to_inbound(payload)
        await async_wait_until(
            lambda payload=payload: connection._extract_sender_key(payload)
            in admission.buffer,
            what="the frame to enter its sender debounce buffer",
        )
        connection._flush_inbound_buffer(connection._extract_sender_key(payload))

    await first_started.wait()
    await other_started.wait()
    assert not second_started.is_set()

    release_first.set()
    await second_started.wait()
    assert seen == ["first", "other", "second"]
    await admission.close()


@pytest.mark.parametrize("command", ["/stop", "/new", "/reset", "/approve", "/deny"])
@pytest.mark.asyncio
async def test_authenticated_owner_controls_bypass_group_backlog(command):
    adapter = _adapter()
    source = adapter.build_source(
        chat_id="group:controls", chat_type="group", user_id="owner"
    )
    session_key = adapter._source_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    responses = []

    async def handler(event):
        return f"handled:{event.get_command()}"

    async def send_response(chat_id, content, **_kwargs):
        assert chat_id == "group:controls"
        responses.append(content)

    adapter.set_message_handler(handler)
    adapter._send_with_retry = send_response
    ctx = _group_context(adapter, source, 1)
    ctx.chat_id = "group:controls"
    ctx.raw_text = command
    ctx.owner_command = command

    await DispatchMiddleware().handle(ctx, _noop)

    assert session_key not in adapter._group_queues
    assert any(
        f"handled:{command.removeprefix('/')}" in response for response in responses
    )
    assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_non_owner_cannot_manufacture_control_with_a_mention():
    adapter = _adapter()
    adapter.send = AsyncMock()
    mention = {
        "msg_type": "TIMCustomElem",
        "msg_content": {
            "data": json.dumps(
                {"elem_type": 1002, "text": "@bot", "user_id": "bot_123"}
            ),
        },
    }
    ctx = InboundContext(
        adapter=adapter,
        push={"bot_owner_id": "real-owner"},
        msg_body=[
            {"msg_type": "TIMTextElem", "msg_content": {"text": "/approve"}},
            mention,
        ],
        chat_type="group",
        chat_id="group:controls",
        from_account="attacker",
    )
    next_fn = AsyncMock()

    await OwnerCommandMiddleware().handle(ctx, next_fn)

    next_fn.assert_not_awaited()
    assert ctx.owner_command is None
    await async_wait_until(
        lambda: adapter.send.await_count == 1, what="the denial notice"
    )
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_multipart_debounce_keeps_image_text_frame_order():
    adapter = _adapter()
    connection = adapter._connection
    admission = connection._inbound_admission
    connection._DEBOUNCE_WINDOW = 3600
    received = []
    processed = asyncio.Event()

    class CapturePipeline:
        async def execute(self, ctx):
            received.append(list(ctx.raw_frames))
            processed.set()

    adapter._inbound_pipeline = CapturePipeline()
    image = json.dumps(
        {
            "From_Account": "owner",
            "GroupId": "multipart",
            "MsgKey": "image",
            "MsgBody": [
                {
                    "MsgType": "TIMImageElem",
                    "MsgContent": {"url": "https://example/image"},
                }
            ],
        }
    ).encode()
    text_frame = json.dumps(
        {
            "From_Account": "owner",
            "GroupId": "multipart",
            "MsgKey": "text",
            "MsgBody": [{"MsgType": "TIMTextElem", "MsgContent": {"text": "caption"}}],
        }
    ).encode()

    assert await connection._push_to_inbound(image)
    assert await connection._push_to_inbound(text_frame)
    await async_wait_until(
        lambda: sum(map(len, admission.buffer.values())) == 2,
        what="both companion frames to enter the debounce buffer",
    )
    connection._flush_inbound_buffer(connection._extract_sender_key(image))
    await processed.wait()

    assert received == [[image, text_frame]]
    await admission.close()


@pytest.mark.asyncio
async def test_push_ack_waits_for_bounded_admission():
    adapter = _adapter()
    connection = adapter._connection
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def blocked_admission(_data):
        admitted.set()
        await release.wait()
        return True

    connection._push_to_inbound = blocked_admission
    connection._ws = AsyncMock()
    decoded = {
        "head": {
            "cmd_type": CMD_TYPE["Push"],
            "cmd": "push",
            "msg_id": "push-1",
            "need_ack": True,
        },
        "data": b"payload",
    }
    with patch(
        "plugins.platforms.yuanbao.adapter.decode_conn_msg", return_value=decoded
    ):
        task = asyncio.create_task(connection._handle_frame(b"frame"))
        await admitted.wait()
        connection._ws.send.assert_not_awaited()
        release.set()
        await task
    connection._ws.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_push_ack_is_not_sent_on_a_replacement_socket():
    adapter = _adapter()
    connection = adapter._connection
    old_ws = AsyncMock()
    new_ws = AsyncMock()
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def blocked_admission(_data):
        admitted.set()
        await release.wait()
        return True

    connection._push_to_inbound = blocked_admission
    connection._ws = old_ws
    decoded = {
        "head": {
            "cmd_type": CMD_TYPE["Push"],
            "cmd": "push",
            "msg_id": "push-old-socket",
            "need_ack": True,
        },
        "data": b"payload",
    }
    with patch(
        "plugins.platforms.yuanbao.adapter.decode_conn_msg", return_value=decoded
    ):
        task = asyncio.create_task(
            connection._handle_frame(b"frame", source_ws=old_ws)
        )
        await admitted.wait()
        connection._ws = new_ws
        release.set()
        await task

    old_ws.send.assert_not_awaited()
    new_ws.send.assert_not_awaited()


async def _noop() -> None:
    return None
