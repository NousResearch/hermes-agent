"""Physical Slack continuations preserve text and earn their own useful preview."""

import asyncio
import re
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.test_slack_initial_streaming import (
    _displayed,
    _ObservedConsumer,
    _SlackClient,
)


class _FailingSlackClient(_SlackClient):
    """Inject uncertain SDK acknowledgements without changing successful display rules."""

    def __init__(self, downgrade=False):
        super().__init__(downgrade)
        self.stop_failures = 0
        self.update_failures = 0
        self.stop_after_effect = False
        self.update_after_effect = False
        self.failures = []

    async def chat_stopStream(self, **kwargs):
        if self.stop_failures:
            self.stop_failures -= 1
            self.failures.append("stop")
            if self.stop_after_effect:
                await super().chat_stopStream(**kwargs)
            else:
                self.calls.append(("stop", kwargs))
            raise TimeoutError("stopStream acknowledgement lost")
        return await super().chat_stopStream(**kwargs)

    async def chat_update(self, **kwargs):
        if self.update_failures:
            self.update_failures -= 1
            self.failures.append("update")
            if self.update_after_effect:
                await super().chat_update(**kwargs)
            else:
                self.calls.append(("update", kwargs))
            raise TimeoutError("chat.update temporarily unavailable")
        return await super().chat_update(**kwargs)


@asynccontextmanager
async def _delivery(transport, cursor, client=None):
    client = client or _SlackClient(downgrade=transport == "downgrade")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    adapter._get_client = MagicMock(return_value=client)
    adapter._running = True
    adapter.MAX_MESSAGE_LENGTH = 700
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm")
    config, _ = GatewayTurnMixin._build_stream_consumer_config(
        object(), source,
        StreamingConfig(edit_interval=0, buffer_threshold=1, cursor=cursor,
                        transport="edit" if transport == "edit" else "draft"),
        adapter, on_missing_cursor="raise")
    consumer = _ObservedConsumer(adapter, "D1", config=config,
                                 metadata={"thread_id": "111.000", "user_id": "U123"})
    task = asyncio.create_task(consumer.run())
    try:
        yield consumer, client, adapter, task
    finally:
        if not task.done():
            task.cancel()
            await task


def _parts(client, cursor=""):
    # Numbering is transport decoration; do not normalize whitespace or answer text.
    return [re.sub(r" \(\d+/\d+\)$", "", text) for text in _displayed(client, cursor)]


# Baseline fixtures avoid prose spaces; dedicated cases below add spaces/newlines
# at split seams and require the actual payload to survive without normalization.
_OPENING = (
    "开场说明提供足够完整的上下文，确保第一条通知有清晰的信息，并且能够独立表达当前答复的主要内容，"
    "读者不必依赖后续消息才能理解这项处理的目的和结果，也不会收到过短或尚未完成的片段。"
)
_CONTINUATION = (
    "后续说明同样需要包含充分的信息，让每个新的物理消息都能独立构成清晰且有用的通知，"
    "而不是把尚未完成的短小片段直接显示出来，只有完整的内容达到要求之后才开始正常更新并继续传递剩余结果"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["draft", "edit", "downgrade"])
@pytest.mark.parametrize("cursor", ["", " ▉"])
@pytest.mark.parametrize("ending", ["sentence", "short", "unpunctuated", "multiple", "newlines", "spaces"])
async def test_visible_slack_overflow_rearms_each_physical_preview(transport, cursor, ending):
    async with _delivery(transport, cursor) as (consumer, client, adapter, task):
        await consumer.settled_delta(_OPENING)
        assert _parts(client, cursor) == [_OPENING]
        sealed = ""
        active = _OPENING
        for index in range(3 if ending == "multiple" else 1):
            padding = str(index) * (consumer._safe_limit - len(active))
            if ending in ("newlines", "spaces"):
                padding = padding[:-1] + ("\n" if ending == "newlines" else " ")
            pending = sealed + active + padding
            tail = {"newlines": "\nNext", "spaces": " Next"}.get(ending, "Next")
            await consumer.settled_delta(padding + tail)
            displayed = "".join(_parts(client, cursor))
            assert pending.startswith(displayed)
            assert len(client.messages) == index + 1
            assert not consumer.final_response_sent
            active = (pending + tail)[len(displayed):]
            sealed = displayed

            if ending in ("short", "unpunctuated"):
                suffix = "。" if ending == "short" else "WithoutPunctuation"
                active += suffix
                await consumer.settled_delta(suffix)
                assert "".join(_parts(client, cursor)) == sealed
                break

            await consumer.settled_delta("。")
            assert len(client.messages) == index + 1  # Short complete sentences still wait.
            await consumer.settled_delta(_CONTINUATION)
            assert len(client.messages) == index + 1  # Length alone is insufficient.
            await consumer.settled_delta("。")
            active += "。" + _CONTINUATION + "。"
            assert len(client.messages) == index + 2
            assert "".join(_parts(client, cursor)) == sealed + active

            await consumer.settled_delta("NormalPartialUpdate")
            active += "NormalPartialUpdate"
            assert "".join(_parts(client, cursor)) == sealed + active

        final = sealed + active
        consumer.finish(final)
        await asyncio.wait_for(task, timeout=10)
        assert "".join(_parts(client)) == final
        assert sum(_OPENING in part for part in _parts(client)) == 1
        assert consumer.final_response_sent
        assert consumer.final_content_delivered
        assert consumer.delivered_final_matches(final) is True
        assert not adapter._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["", " ▉"])
@pytest.mark.parametrize("transport,failure", [
    ("draft", "stop"), ("draft", "stop-after-effect"), ("draft", "stop-twice"),
    ("edit", "update"), ("downgrade", "update"),
    ("draft", "stop-update"), ("draft", "stop-after-effect-update"),
    ("draft", "stop-update-after-effect"),
])
async def test_slack_overflow_recovers_sdk_failures_without_reposting_prefix(transport, failure, cursor):
    client = _FailingSlackClient(downgrade=transport == "downgrade")
    async with _delivery(transport, cursor, client) as (consumer, client, adapter, task):
        await consumer.settled_delta(_OPENING)
        assert _parts(client, cursor) == [_OPENING]
        head = _OPENING + "x" * (consumer._safe_limit - len(_OPENING))
        # Arm only after a preview is visible; sealing must add the unseen head
        # before opening the continuation, even if its acknowledgement is lost.
        client.stop_failures = 2 if failure == "stop-twice" else int(failure.startswith("stop"))
        client.stop_after_effect = "stop-after-effect" in failure
        client.update_failures = int("update" in failure)
        client.update_after_effect = failure.endswith("update-after-effect")
        tail = "AShortFinalTail"
        await consumer.settled_delta(head[len(_OPENING):] + tail)
        assert not consumer.final_response_sent
        final = head + tail
        consumer.finish(final)
        await asyncio.wait_for(task, timeout=10)
        if failure.startswith("stop"):
            assert "stop" in client.failures
        if "update" in failure:
            assert "update" in client.failures
        assert "".join(_parts(client)) == final
        assert sum(_OPENING in part for part in _parts(client)) == 1
        assert consumer.final_response_sent
        assert consumer.final_content_delivered
        assert consumer.delivered_final_matches(final) is True
        assert not adapter._active_streams
