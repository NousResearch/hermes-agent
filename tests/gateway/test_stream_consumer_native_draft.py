"""Animated draft previews coalesce deltas without rewriting already shown text."""

from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class AnimatedDraftAdapter(BasePlatformAdapter):
    DRAFT_STREAM_PREFIX_STABLE = True

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    def supports_draft_streaming(self, chat_type=None, metadata=None, **kwargs):
        return chat_type == "dm"

    async def send_draft(self, chat_id, draft_id, content, metadata=None):
        self.drafts.append((draft_id, content))
        return self.draft_results.pop(0) if self.draft_results else SendResult(success=True)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.messages.append(content)
        return SendResult(success=True, message_id=str(len(self.messages)))

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.messages.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {"type": "dm"}


def make_consumer(monkeypatch, *, cursor="", edit_interval=0.8):
    adapter = AnimatedDraftAdapter.__new__(AnimatedDraftAdapter)
    adapter.drafts = []
    adapter.draft_results = []
    adapter.messages = []
    clock = SimpleNamespace(now=100.0)
    fake_time = SimpleNamespace(monotonic=lambda: clock.now)
    monkeypatch.setattr("gateway.stream_consumer.time", fake_time)
    monkeypatch.setattr("gateway.stream_consumer_transport.time", fake_time)
    consumer = GatewayStreamConsumer(
        adapter, "123", StreamConsumerConfig(
            transport="draft", chat_type="dm", edit_interval=edit_interval,
            buffer_threshold=4, cursor=cursor,
        ),
    )
    return consumer, adapter, clock


async def push_delta(consumer, text):
    consumer.on_delta(text)
    tick = consumer._drain_queue()
    if consumer._should_edit(tick):
        await consumer._push_update(tick)


@pytest.mark.asyncio
async def test_native_drafts_coalesce_fast_deltas_and_publish_latest_snapshot(monkeypatch):
    consumer, adapter, clock = make_consumer(monkeypatch)
    await consumer._start_transports()

    await push_delta(consumer, "首个回复")
    for delta in ("，", "接", "着", "输", "出"):
        clock.now += 0.05
        await push_delta(consumer, delta)
    assert [text for _, text in adapter.drafts] == ["首个回复"]

    clock.now = 100.81
    await push_delta(consumer, "。")
    assert [text for _, text in adapter.drafts] == ["首个回复", "首个回复，接着输出。"]
    assert len({draft_id for draft_id, _ in adapter.drafts}) == 1


@pytest.mark.asyncio
async def test_native_draft_code_prefix_has_no_synthetic_fence_or_cursor(monkeypatch):
    consumer, adapter, clock = make_consumer(monkeypatch, cursor="▉")
    await consumer._start_transports()

    await push_delta(consumer, "代码：\n```python\nprint(")
    clock.now += 0.81
    await push_delta(consumer, "'hello')\n```")

    frames = [text for _, text in adapter.drafts]
    assert frames == ["代码：\n```python\nprint(", "代码：\n```python\nprint('hello')\n```"]
    assert frames[1].startswith(frames[0])


@pytest.mark.asyncio
async def test_skipped_draft_retries_latest_snapshot_without_claiming_delivery(monkeypatch):
    consumer, adapter, clock = make_consumer(monkeypatch)
    await consumer._start_transports()
    adapter.draft_results = [SendResult(success=True, raw_response={"skipped": True})]

    await push_delta(consumer, "首个回复")
    assert consumer._last_sent_text == ""
    assert consumer.already_sent is False
    assert consumer._use_draft_streaming is True
    assert adapter.messages == []

    clock.now += 0.81
    await push_delta(consumer, "，后续内容")
    assert consumer._last_sent_text == "首个回复，后续内容"
    assert adapter.drafts[-1][1] == "首个回复，后续内容"
    assert len({draft_id for draft_id, _ in adapter.drafts}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [False, True])
async def test_draft_cooldown_keeps_preview_transport_and_full_final_is_immediate(
    monkeypatch, success,
):
    consumer, adapter, clock = make_consumer(monkeypatch)
    await consumer._start_transports()
    adapter.draft_results = [SendResult(
        success=success, error=None if success else "flood_control:9", retry_after=9,
        raw_response={"skipped": True} if success else None,
    )]

    await push_delta(consumer, "首个回复")
    clock.now += 1.0
    await push_delta(consumer, "，新内容")
    assert len(adapter.drafts) == 1
    assert consumer._use_draft_streaming is True
    assert consumer._last_sent_text == ""
    assert adapter.messages == []

    # Completion bypasses preview cooldown and carries post-stream augmentation.
    final_text = "首个回复，新内容。\n\n完整的最终结论。"
    consumer.finish(final_text)
    tick = consumer._drain_queue()
    await consumer._push_update(tick)
    await consumer._finalize_turn(tick)
    assert adapter.messages == [final_text]
    assert consumer.delivered_final_matches(final_text) is True


@pytest.mark.asyncio
async def test_draft_resumes_after_cooldown_with_latest_accumulated_text(monkeypatch):
    consumer, adapter, clock = make_consumer(monkeypatch)
    await consumer._start_transports()
    adapter.draft_results = [SendResult(
        success=True, retry_after=9, raw_response={"skipped": True},
    )]

    await push_delta(consumer, "首个回复")
    clock.now += 1.0
    await push_delta(consumer, "，新内容")
    clock.now += 8.1
    await push_delta(consumer, "。")

    assert [text for _, text in adapter.drafts] == ["首个回复", "首个回复，新内容。"]
    assert consumer._last_sent_text == "首个回复，新内容。"
    assert adapter.messages == []
