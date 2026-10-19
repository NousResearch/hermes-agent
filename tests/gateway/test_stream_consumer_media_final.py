"""Regression for #141584: a completed MEDIA directive retracts its partial preview."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_stream_consumer_draft import _make_draft_capable_adapter
from tests.gateway.test_stream_consumer_fresh_final import _make_adapter
from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter


@pytest_asyncio.fixture
async def display():
    adapter = _make_adapter()
    adapter.MAX_MESSAGE_LENGTH = 600
    adapter.draft_stream_is_message = False
    config = StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=" ▉", transport="edit")
    state = SimpleNamespace(
        screen={}, calls=[], media_preview=asyncio.Event(), prior_tail=asyncio.Event(),
        prior_finalized=asyncio.Event(), edit_failed=asyncio.Event(), fail_edits=False,
    )

    async def send(*, chat_id, content, **kwargs):
        message_id = f"preview-{adapter.send.await_count}"
        state.screen[message_id] = content
        state.calls.append(("send", message_id, content))
        if content.startswith("MEDIA:"):
            state.media_preview.set()
        elif content.endswith(config.cursor):
            state.prior_tail.set()
        return SimpleNamespace(success=True, message_id=message_id)

    async def edit(*, chat_id, message_id, content, finalize=False, **kwargs):
        state.calls.append(("edit", message_id, content))
        if state.fail_edits:
            state.edit_failed.set()
            return SimpleNamespace(success=False, error="editing unsupported", raw_response=None)
        state.screen[message_id] = content
        if finalize and not content.startswith("MEDIA:"):
            state.prior_finalized.set()
        return SimpleNamespace(success=True, message_id=message_id)

    async def delete(chat_id, message_id):
        state.calls.append(("delete", message_id, None))
        state.screen.pop(message_id)
        return True

    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(side_effect=edit)
    adapter.delete_message = AsyncMock(side_effect=delete)
    state.adapter = adapter
    state.consumer = GatewayStreamConsumer(adapter, "fixture-chat", config)
    state.task = asyncio.create_task(state.consumer.run())
    try:
        yield state
    finally:
        if not state.task.done():
            state.consumer.finish()
            await asyncio.wait_for(state.task, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", [
    "delta_then_finish", "authoritative_finish", "bare_finish", "empty_finish", "whitespace_finish",
])
@pytest.mark.parametrize("heads", ["earlier_segment", "current_segment"])
async def test_finish_retracts_only_completed_media_preview(tmp_path, display, completion, heads):
    consumer = display.consumer
    partial = "MEDIA:" + str(tmp_path / "image-partial")
    head = "Finalized head. " * 30 + "\n"
    if heads == "earlier_segment":
        consumer.on_delta(head * 2)
        await asyncio.wait_for(display.prior_tail.wait(), 5)
        consumer.on_segment_break()
        await asyncio.wait_for(display.prior_finalized.wait(), 5)
        consumer.on_delta(partial)
    else:
        consumer.on_delta(head + partial)
    await asyncio.wait_for(display.media_preview.wait(), 5)
    preview_id = consumer._message_id
    preserved = {key: value for key, value in display.screen.items() if key != preview_id}
    assert preserved
    assert all(not text.endswith(consumer.cfg.cursor) for text in preserved.values())
    assert consumer._turn_split_delivery is (heads == "current_segment")

    final = (head if heads == "current_segment" else "") + partial + ".png"
    if completion == "delta_then_finish":
        consumer.on_delta(".png")
        consumer.finish()
    elif completion == "authoritative_finish":
        consumer.finish(final_text=final)
    elif completion == "bare_finish":
        consumer.finish()
    else:
        consumer.finish(final_text="" if completion == "empty_finish" else " \t ")
    await asyncio.wait_for(display.task, 5)

    if completion not in {"delta_then_finish", "authoritative_finish"}:
        assert display.screen.keys() == preserved.keys() | {preview_id}
        assert all(display.screen[key] == value for key, value in preserved.items())
        display.adapter.delete_message.assert_not_awaited()
        assert consumer._message_id == preview_id
        return
    assert display.screen == preserved
    display.adapter.delete_message.assert_awaited_once_with("fixture-chat", preview_id)
    assert consumer._message_id is None
    assert preview_id not in consumer._preview_message_ids
    assert preview_id not in consumer._segment_preview_message_ids
    if heads == "current_segment":
        assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("deletion", ["missing", "false", "raises", "acknowledged"])
async def test_media_cleanup_requires_acknowledgment(tmp_path, display, fallback, deletion):
    consumer = display.consumer
    consumer.on_delta("MEDIA:" + str(tmp_path / "image-partial"))
    await asyncio.wait_for(display.media_preview.wait(), 5)
    if fallback:
        display.fail_edits = True
        consumer.on_delta("-continued")
        await asyncio.wait_for(display.edit_failed.wait(), 5)
        assert consumer._fallback_final_send

    if deletion == "missing":
        del display.adapter.delete_message
    elif deletion == "false":
        display.adapter.delete_message.side_effect = None
        display.adapter.delete_message.return_value = False
    elif deletion == "raises":
        display.adapter.delete_message.side_effect = RuntimeError("delete unavailable")
    preview_id, last_visible = consumer._message_id, consumer._last_sent_text
    screen, call_count = dict(display.screen), len(display.calls)
    consumer.finish(final_text=consumer._accumulated + ".png")
    await asyncio.wait_for(display.task, 5)

    if deletion == "acknowledged":
        assert display.screen == {}
        assert display.calls[call_count:] == [("delete", preview_id, None)]
        assert consumer._message_id is None
        assert consumer._last_sent_text == ""
        assert preview_id not in consumer._preview_message_ids
        assert preview_id not in consumer._segment_preview_message_ids
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True
    else:
        assert display.screen == screen
        assert display.calls[call_count:] == []
        assert consumer._message_id == preview_id
        assert consumer._last_sent_text == last_visible
        assert preview_id in consumer._preview_message_ids
        assert preview_id in consumer._segment_preview_message_ids
        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is False
    if deletion != "missing":
        display.adapter.delete_message.assert_awaited_once_with("fixture-chat", preview_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "draft", "cumulative_draft"])
async def test_authoritative_media_keeps_other_transports_finalizing(tmp_path, transport):
    partial = "MEDIA:" + str(tmp_path / "image-partial")
    preview_sent = asyncio.Event()
    adapter = (_make_native_streaming_adapter() if transport == "native"
               else _make_draft_capable_adapter())
    adapter.draft_stream_is_message = transport == "cumulative_draft"
    adapter.delete_message = AsyncMock(return_value=True)
    push_name = "send_stream_frame" if transport == "native" else "send_draft"
    original_push = getattr(adapter, push_name)

    async def push(*args, **kwargs):
        result = await original_push(*args, **kwargs)
        if kwargs.get("content", args[0] if args else "") == partial:
            preview_sent.set()
        return result

    setattr(adapter, push_name, push)
    consumer = GatewayStreamConsumer(adapter, "fixture-chat", StreamConsumerConfig(
        edit_interval=0, buffer_threshold=1, cursor="", transport="auto", chat_type="dm",
    ))
    task = asyncio.create_task(consumer.run())
    try:
        consumer.on_delta(partial)
        await asyncio.wait_for(preview_sent.wait(), 5)
        consumer.finish(final_text=partial + ".png")
        await asyncio.wait_for(task, 5)
        # These transports retain their existing final payload and close/seal path.
        if transport == "native":
            assert adapter.frames[-1]["finalize"] is True
            assert adapter.frames[-1]["text"] == partial
            adapter.send.assert_not_awaited()
        else:
            adapter.send.assert_awaited_once()
            assert adapter.send.await_args.kwargs["content"] == partial
        adapter.delete_message.assert_not_awaited()
    finally:
        if not task.done():
            consumer.finish()
            await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_cancelled_media_delete_keeps_cleanup_unconfirmed(tmp_path, display):
    consumer = display.consumer
    consumer.on_delta("MEDIA:" + str(tmp_path / "image-partial"))
    await asyncio.wait_for(display.media_preview.wait(), 5)
    delete_started, delete_cancelled, release_delete = (
        asyncio.Event(), asyncio.Event(), asyncio.Event(),
    )

    async def blocked_delete(chat_id, message_id):
        delete_started.set()
        try:
            await release_delete.wait()
        except asyncio.CancelledError:
            delete_cancelled.set()
            raise
        return True

    display.adapter.delete_message.side_effect = blocked_delete
    preview_id, last_visible = consumer._message_id, consumer._last_sent_text
    screen, call_count = dict(display.screen), len(display.calls)
    consumer.finish(final_text=consumer._accumulated + ".png")
    await asyncio.wait_for(delete_started.wait(), 5)
    display.task.cancel()
    await asyncio.wait_for(display.task, 5)

    assert delete_cancelled.is_set()
    assert display.screen == screen
    assert display.calls[call_count:] == []
    assert consumer._message_id == preview_id
    assert consumer._last_sent_text == last_visible
    assert preview_id in consumer._preview_message_ids
    assert preview_id in consumer._segment_preview_message_ids
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False
    assert consumer._delivered_final_text is None
    display.adapter.delete_message.assert_awaited_once_with("fixture-chat", preview_id)
