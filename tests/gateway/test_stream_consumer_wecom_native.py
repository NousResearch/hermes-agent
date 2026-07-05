"""Tests for WeCom native reply streaming in GatewayStreamConsumer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _WeComNativeAdapter:
    MAX_MESSAGE_LENGTH = 4096
    SUPPORTS_NATIVE_STREAMING_REPLIES = True

    def __init__(self, *, native_results=None, thinking_result=None):
        self.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_final")
        )
        self.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="msg_edit")
        )
        self.draft_calls = []
        self.send_draft = AsyncMock(return_value=SimpleNamespace(success=True))
        self.draft_supported = False
        self._native_results = list(native_results or [])
        self._thinking_result = thinking_result
        self.native_calls = []

    async def send_stream_chunk(self, **kwargs):
        self.native_calls.append(kwargs)
        if kwargs.get("content") == "THINKING_MESSAGE":
            if self._thinking_result is not None:
                return self._thinking_result
            return SimpleNamespace(success=True)
        if self._native_results:
            return self._native_results.pop(0)
        return SimpleNamespace(success=True)

    def supports_draft_streaming(self, chat_type=None, metadata=None) -> bool:
        return bool(self.draft_supported)


def _reply_metadata():
    return {
        "reply": {
            "enterprise_id": "ent_1",
            "chat_type": "dm",
            "receive_id": "user_1",
            "msgid": "wecom_msg_1",
        }
    }


@pytest.mark.asyncio
async def test_native_reply_transport_requires_adapter_capability_and_reply_context():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls, "expected native send_stream_chunk calls"
    adapter.edit_message.assert_not_called()
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_reply_transport_honors_edit_opt_out():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
            transport="edit",
        ),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls == []
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "Hello"


@pytest.mark.asyncio
async def test_native_reply_transport_not_used_without_reply_context():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=False, error="missing_reply_context")]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
        metadata={},
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls == []
    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "Hello ▉"


@pytest.mark.asyncio
async def test_native_final_success_sets_final_content_delivered():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=True), SimpleNamespace(success=True)]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    final_calls = [call for call in adapter.native_calls if call["finalize"] is True]
    assert len(final_calls) == 1


@pytest.mark.asyncio
async def test_native_final_success_sends_single_finish_frame_when_done_arrives_with_first_delta():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=True), SimpleNamespace(success=True)]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=100, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    consumer.finish()
    await consumer.run()

    final_calls = [call for call in adapter.native_calls if call["finalize"] is True]
    assert len(final_calls) == 1


@pytest.mark.asyncio
async def test_native_partial_success_falls_back_tail_only():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Hello "},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_partial_success_falls_back_tail_only_after_more_deltas():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Hello "},
            ),
            SimpleNamespace(
                success=False,
                error="native_partial_again",
                raw_response={"delivered_prefix": "Hello world"},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.on_delta(" again")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    # After the first native partial success, the consumer must stop sending
    # native chunks. Otherwise a later partially-visible native frame can make
    # the tail-only fallback duplicate client-visible text.
    assert [call["content"] for call in adapter.native_calls] == [
        "THINKING_MESSAGE",
        "Hello ",
        "Hello world",
    ]
    assert adapter._native_results, "later native results should remain unused"
    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world again"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_partial_success_confirmed_prefix_len_falls_back_tail_only():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"confirmed_prefix_len": len("Hello ")},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_partial_success_confirmed_prefix_len_larger_than_content_clamps_visible_prefix():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"confirmed_prefix_len": len("Hello world") + 100},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_reply_transport_takes_precedence_over_draft_capability():
    adapter = _WeComNativeAdapter()
    adapter.draft_supported = True
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        ),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert adapter.native_calls
    adapter.send_draft.assert_not_awaited()
    adapter.edit_message.assert_not_called()


@pytest.mark.asyncio
async def test_native_failure_before_visible_content_sends_full_final_once():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=False, error="native_unavailable")]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello world")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert len(adapter.native_calls) == 2
    assert [call["content"] for call in adapter.native_calls] == ["THINKING_MESSAGE", "Hello world"]
    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "Hello world"


@pytest.mark.asyncio
async def test_segment_break_native_failure_flushes_tail_without_duplicate():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Hello "},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.on_segment_break()
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter.send.assert_awaited_once()
    send_await_args = adapter.send.await_args
    assert send_await_args is not None
    assert send_await_args.kwargs["content"] == "world"


@pytest.mark.asyncio
async def test_ambiguous_native_delivered_prefix_mismatch_suppresses_blind_full_resend():
    adapter = _WeComNativeAdapter(
        native_results=[
            SimpleNamespace(success=True),
            SimpleNamespace(
                success=False,
                error="native_partial",
                raw_response={"delivered_prefix": "Mismatch"},
            ),
        ]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello ")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("world")
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert [call["content"] for call in adapter.native_calls] == [
        "THINKING_MESSAGE",
        "Hello ",
        "Hello world",
    ]
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_native_reply_stream_sends_empty_start_frame_before_text_delta():
    adapter = _WeComNativeAdapter(
        native_results=[SimpleNamespace(success=True), SimpleNamespace(success=True)]
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=100, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    assert adapter.native_calls, "expected an empty native start frame before visible text"
    assert adapter.native_calls[0]["content"] == "THINKING_MESSAGE"
    assert adapter.native_calls[0]["finalize"] is False

    consumer.on_delta("Hello")
    consumer.finish()
    await task

    assert [call["finalize"] for call in adapter.native_calls] == [False, True]
    assert adapter.native_calls[1]["content"] == "Hello"
    assert adapter.native_calls[0]["stream_key"] == adapter.native_calls[1]["stream_key"]


@pytest.mark.asyncio
async def test_native_thinking_frame_failure_does_not_leak_sentinel_as_visible_text():
    adapter = _WeComNativeAdapter(
        thinking_result=SimpleNamespace(success=False, error="reply context expired")
    )
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.on_delta("Hello")
    consumer.finish()
    await task

    sent_contents = [call.kwargs.get("content") for call in adapter.send.await_args_list]
    assert "THINKING_MESSAGE" not in sent_contents
    assert sent_contents == ["Hello"]


@pytest.mark.asyncio
async def test_native_streaming_keeps_full_accumulated_text_until_final_split():
    adapter = _WeComNativeAdapter()
    adapter.MAX_MESSAGE_LENGTH = 700
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=100, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )
    text = ("A" * 590) + "   \n" + ("B" * 700) + "\n  " + ("C" * 300)

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(text[:900])
    await asyncio.sleep(0.08)
    consumer.on_delta(text[900:])
    await asyncio.sleep(0.08)
    consumer.finish()
    await task

    final_calls = [call for call in adapter.native_calls if call.get("finalize") is True]
    assert len(final_calls) == 1
    assert final_calls[-1]["content"] == text
    assert {call["stream_key"] for call in adapter.native_calls} == {
        adapter.native_calls[0]["stream_key"]
    }
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_native_streaming_tool_boundary_reuses_same_stream_key():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta("before tools. ")
    await asyncio.sleep(0.06)
    consumer.on_delta(None)
    await asyncio.sleep(0.06)
    consumer.on_delta("after tools.")
    consumer.finish()
    await task

    final_calls = [call for call in adapter.native_calls if call.get("finalize") is True]
    assert len(final_calls) == 1
    assert final_calls[-1]["content"] == "before tools. after tools."
    assert {call["stream_key"] for call in adapter.native_calls} == {
        adapter.native_calls[0]["stream_key"]
    }
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True

@pytest.mark.asyncio
async def test_native_reply_thinking_only_finish_clears_stream_state():
    from gateway.config import PlatformConfig
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._remember_reply_req_id("origin-msg-1", "reply-req-1")
    sent_bodies = []

    async def _fake_send_reply_request(reply_req_id, body):
        sent_bodies.append((reply_req_id, body))
        return {
            "errcode": 0,
            "headers": {"req_id": reply_req_id},
            "body": {"stream": {"stream_id": "stream-1"}},
        }

    adapter._send_reply_request = _fake_send_reply_request
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        initial_reply_to_id="origin-msg-1",
        metadata=_reply_metadata(),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert [body["stream"]["event"] for _reply_req_id, body in sent_bodies] == [
        "start",
        "finish",
    ]
    assert adapter._stream_states == {}
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_native_reply_stale_generation_finalizes_started_stream():
    adapter = _WeComNativeAdapter()
    current_checks = iter([True, False])
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
        run_still_current=lambda: next(current_checks, False),
    )

    await consumer.run()

    assert [call["content"] for call in adapter.native_calls] == [
        "THINKING_MESSAGE",
        "",
    ]
    assert [call["finalize"] for call in adapter.native_calls] == [False, True]
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_native_reply_cancelled_run_finalizes_started_stream():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    assert [call["content"] for call in adapter.native_calls] == [
        "THINKING_MESSAGE",
        "",
    ]
    assert [call["finalize"] for call in adapter.native_calls] == [False, True]
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_native_reply_intentional_silence_finalizes_started_stream():
    adapter = _WeComNativeAdapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        metadata=_reply_metadata(),
    )

    consumer.on_delta("NO_REPLY")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert [call["content"] for call in adapter.native_calls] == [
        "THINKING_MESSAGE",
        "",
    ]
    assert [call["finalize"] for call in adapter.native_calls] == [False, True]
    adapter.send.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_real_wecom_adapter_resolves_consumer_reply_msgid_to_cached_req_id():
    from gateway.config import PlatformConfig
    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._remember_reply_req_id("origin-msg-1", "reply-req-1")
    sent_bodies = []

    async def _fake_send_reply_request(reply_req_id, body):
        sent_bodies.append((reply_req_id, body))
        return {
            "errcode": 0,
            "headers": {"req_id": reply_req_id},
            "body": {"stream": {"stream_id": "stream-1"}},
        }

    adapter._send_reply_request = _fake_send_reply_request
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor="", transport="auto"),
        initial_reply_to_id="origin-msg-1",
        metadata=_reply_metadata(),
    )

    consumer.on_delta("Hello")
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    assert [reply_req_id for reply_req_id, _body in sent_bodies] == ["reply-req-1", "reply-req-1", "reply-req-1"]
    assert [body["stream"]["event"] for _reply_req_id, body in sent_bodies] == ["start", "continue", "finish"]
    assert all(body["msgtype"] == "stream" for _reply_req_id, body in sent_bodies)
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
