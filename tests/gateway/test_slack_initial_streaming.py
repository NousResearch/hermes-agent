"""Slack's first answer notification waits for useful text, not its activity status."""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer
from plugins.platforms.slack.adapter import SlackAdapter
from tests.waiting import async_wait_until


class _SlackClient:
    """Fake only the Slack SDK: retain exactly what its successful calls display."""

    def __init__(self, downgrade):
        self.messages = {}
        self.calls = []
        self.downgrade = downgrade
        self.assistant_threads_setStatus = AsyncMock(return_value={"ok": True})
        self.agents_sessions_setStatus = self.assistant_threads_setStatus

    async def chat_startStream(self, **kwargs):
        self.calls.append(("start", kwargs))
        if self.downgrade:
            raise RuntimeError("not_allowed")
        ts = str(len(self.messages) + 1)
        self.messages[ts] = kwargs["markdown_text"]
        return {"ok": True, "ts": ts}

    async def chat_appendStream(self, **kwargs):
        self.calls.append(("append", kwargs))
        self.messages[kwargs["ts"]] += kwargs.get("markdown_text", "")
        return {"ok": True}

    async def chat_stopStream(self, **kwargs):
        self.calls.append(("stop", kwargs))
        self.messages[kwargs["ts"]] += kwargs.get("markdown_text", "")
        return {"ok": True}

    async def chat_postMessage(self, **kwargs):
        self.calls.append(("post", kwargs))
        ts = str(len(self.messages) + 1)
        self.messages[ts] = kwargs["text"]
        return {"ok": True, "ts": ts}

    async def chat_update(self, **kwargs):
        self.calls.append(("update", kwargs))
        self.messages[kwargs["ts"]] = kwargs["text"]
        return {"ok": True, "ts": kwargs["ts"]}


class _ObservedConsumer(GatewayStreamConsumer):
    """Observe completed loop iterations without turning a queue barrier into a flush."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drains = 0

    def _drain_queue(self):
        tick = super()._drain_queue()
        self.drains += 1
        return tick

    async def settled_delta(self, text):
        before = self.drains
        self.on_delta(text)
        await async_wait_until(lambda: self.drains >= before + 2,
                               what="consumer completed a delivery tick")


def _displayed(client, cursor):
    return [text.removesuffix(cursor) if cursor else text for text in client.messages.values()]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["draft", "edit", "downgrade"])
@pytest.mark.parametrize("cursor", ["", " ▉"])
@pytest.mark.parametrize("ending", [
    "sentence", "short", "bare-finish", "overflow", "tool", "commentary", "flush", "stale", "cancel",
])
async def test_slack_initial_answer_visibility(transport, cursor, ending):
    client = _SlackClient(downgrade=transport == "downgrade")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    adapter._get_client = MagicMock(return_value=client)
    adapter._running = True
    if ending == "overflow":
        adapter.MAX_MESSAGE_LENGTH = 700
    metadata = {"thread_id": "111.000", "user_id": "U123"}
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm")
    config, _ = GatewayTurnMixin._build_stream_consumer_config(
        object(), source,
        StreamingConfig(edit_interval=0, buffer_threshold=1, cursor=cursor, transport=(
            "edit" if transport == "edit" else "draft")),
        adapter, on_missing_cursor="raise")
    current = True
    consumer = _ObservedConsumer(adapter, "D1", config=config, metadata=metadata,
                                 run_still_current=lambda: current)
    task = asyncio.create_task(consumer.run())
    try:
        await adapter.send_typing("D1", metadata=metadata)
        assert client.assistant_threads_setStatus.await_count == 1
        await consumer.settled_delta("OK")
        assert client.calls == []
        assert not consumer.final_response_sent

        if ending in ("stale", "cancel"):
            if ending == "stale":
                current = False
            else:
                task.cancel()
            await asyncio.wait_for(task, timeout=10)
            assert client.calls == []
            assert not consumer.final_content_delivered
            return

        if ending in ("short", "bare-finish"):
            consumer.finish("OK" if ending == "short" else None)
            await asyncio.wait_for(task, timeout=10)
            assert list(client.messages.values()) == ["OK"]
            assert consumer.final_content_delivered
            return

        if ending == "overflow":
            final = "OK" + "x" * 1500
            await consumer.settled_delta(final[2:])
            assert client.calls == []  # Overflow must not bypass the sentence gate.
            consumer.finish(final)
            await asyncio.wait_for(task, timeout=10)
            displayed = list(client.messages.values())
            assert len(displayed) > 1
            assert "".join(re.sub(r" \(\d+/\d+\)$", "", part) for part in displayed) == final
            assert consumer.final_content_delivered
            return

        if ending == "sentence":
            await consumer.settled_delta(". ")
            assert client.calls == []  # The first completed sentence is too short.
            body = "This second sentence provides the useful answer and enough context for a readable notification"
            await consumer.settled_delta(body)
            assert client.calls == []  # The character threshold alone must not release it.
            await consumer.settled_delta(". ")
            first = "OK. " + body + ". "
            assert _displayed(client, cursor) == [first]
            initial = client.calls[0][1].get("markdown_text", client.calls[0][1].get("text"))
            assert initial.startswith(first)
            assert not consumer.final_response_sent
            await consumer.settled_delta("Later incomplete")
            assert _displayed(client, cursor) == [first + "Later incomplete"]
            final = first + "Later incomplete sentence. Verified."
            consumer.finish(final)
            await asyncio.wait_for(task, timeout=10)
            assert list(client.messages.values()) == [final]
            assert consumer.delivered_final_matches(final) is True
        else:
            if ending == "tool":
                consumer.on_delta(None)
            elif ending == "commentary":
                consumer.on_commentary("Checking the result")
            else:
                assert await asyncio.to_thread(consumer.flush_pending_sync)
            await async_wait_until(lambda: "OK" in client.messages.values(),
                                   what="short completed preamble delivered at boundary")
            assert not consumer.final_response_sent
            if ending == "commentary":
                await async_wait_until(lambda: len(client.messages) == 2,
                                       what="commentary delivered after preamble")
            await consumer.settled_delta("Next")
            assert _displayed(client, cursor)[-1] == "Next"
            consumer.finish("Next answer")
            await asyncio.wait_for(task, timeout=10)
            expected = ["OK", "Next answer"]
            if ending == "commentary":
                expected.insert(1, "Checking the result")
            assert list(client.messages.values()) == expected
        assert consumer.final_response_sent
        assert consumer.final_content_delivered
        assert not adapter._active_streams
    finally:
        if not task.done():
            task.cancel()
            await task
