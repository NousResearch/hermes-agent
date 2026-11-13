"""A streamed reply whose final edit AND fallback send were flood-refused is finished in place.

Telegram refuses the finalize edit and the continuation send with a RetryAfter longer than the
adapter's inline cap. Before: the consumer dropped the preview id, every delivery flag stayed unset
and the gateway's plain final send posted the whole answer again beside the frozen preview. Now the
consumer keeps the preview id and the gateway edits that preview up to the complete answer once the
flood window passes: one message, complete.
"""

import asyncio
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.platforms.base import SendResult
from hermes_agent.gateway.run_turn import GatewayTurnMixin
from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until


class _Harness(GatewayTurnMixin):
    """Only the delivery decision under test; the real consumer and predicate run."""


class _FloodedChat:
    """One chat whose edits and sends are flood-refused until ``open_at`` (loop time)."""

    MAX_MESSAGE_LENGTH = 4096
    FALLBACK_ON_FINAL_EDIT_FLOOD = True  # Telegram's: a flood-refused final edit falls back at once

    def __init__(self, flood_secs: float) -> None:
        self.flood_secs = flood_secs
        self.open_at: float = float("inf")
        self.armed = False
        self.messages: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    def _flooded(self) -> bool:
        return self.armed and asyncio.get_running_loop().time() < self.open_at

    def _refuse(self) -> SendResult:
        return SendResult(success=False, error=f"flood_control:{self.flood_secs}", retry_after=self.flood_secs)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", content))
        if self._flooded():
            return self._refuse()
        mid = str(len(self.messages) + 1)
        self.messages[mid] = content
        return SendResult(success=True, message_id=mid)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, **_kw):
        self.calls.append(("edit", content))
        if not self.armed:  # the first edit arms the flood window
            self.armed, self.open_at = True, asyncio.get_running_loop().time() + self.flood_secs
        if self._flooded():
            return self._refuse()
        self.messages[message_id] = content
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id):
        self.messages.pop(message_id, None)
        return True


async def _flood_frozen_turn(adapter: _FloodedChat, final: str):
    consumer = GatewayStreamConsumer(adapter, "chat-1", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=" ▉"))
    # The adapter fails a RetryAfter past its inline cap closed; the consumer's own bounded retry
    # (5 s) must not out-wait it, as with Telegram's real 18-35 s edit penalties.
    consumer._max_fallback_flood_retry_seconds = adapter.flood_secs / 2
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(final[:20])
    await async_wait_until(lambda: adapter.messages, timeout=5, what="first preview sent")
    consumer.on_delta(final[20:])
    await async_wait_until(lambda: adapter.armed, timeout=5, what="preview edit flood-refused")
    consumer.finish(final)
    await asyncio.wait_for(task, timeout=5)
    return consumer


@pytest.mark.asyncio
async def test_flood_frozen_preview_is_finished_in_place_not_sent_twice(monkeypatch):
    monkeypatch.setattr("hermes_agent.gateway.run_turn_final_reconcile._FLOOD_WAIT_SLACK_SECS", 0.05)
    adapter = _FloodedChat(flood_secs=0.6)  # long enough to outlast the consumer's own retries
    final = "The answer begins here and then " + "continues with more words " * 8 + "until the end."
    consumer = await _flood_frozen_turn(adapter, final)
    assert not consumer.final_response_sent and not consumer.final_content_delivered

    response = {"final_response": final}
    turn_ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="chat-1"),
                               session_key="agent:main:telegram:group:chat-1")
    await asyncio.wait_for(_Harness()._run_agent_mark_streamed_delivery(response, turn_ctx), timeout=5)

    assert response.get("already_sent") is True  # the gateway's normal final send stays off
    assert list(adapter.messages.values()) == [final]  # ONE message, complete, no cursor


@pytest.mark.asyncio
async def test_preview_still_flooded_past_the_retry_leaves_the_normal_send(monkeypatch):
    """A window that outlasts the bounded wait is not a delivery: ``already_sent`` stays unset so
    the normal ledgered final send (and its redelivery) still owns the answer."""
    monkeypatch.setattr("hermes_agent.gateway.run_turn_final_reconcile._FLOOD_WAIT_SLACK_SECS", 0.0)
    monkeypatch.setattr("hermes_agent.gateway.run_turn_final_reconcile._MAX_FLOOD_WAITS", 1)
    adapter = _FloodedChat(flood_secs=0.6)
    final = "Short answer that is long enough to stream in two pieces."
    consumer = await _flood_frozen_turn(adapter, final)
    adapter.open_at += 30  # Telegram extends the penalty

    response = {"final_response": final}
    turn_ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="chat-1"),
                               session_key="agent:main:telegram:group:chat-1")
    await asyncio.wait_for(_Harness()._run_agent_mark_streamed_delivery(response, turn_ctx), timeout=5)

    assert "already_sent" not in response


@pytest.mark.asyncio
async def test_consumer_cancelled_mid_final_leaves_one_complete_message():
    """The turn's 5 s flush bound cancels a consumer stuck in its final edit (#53449): the delivery
    flags stay unset while the preview is on screen. The gateway finishes that preview in place."""
    gate = asyncio.Event()

    class _StuckFinal(_FloodedChat):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False, **_kw):
            self.calls.append(("edit", content))
            if finalize and not gate.is_set():
                gate.set()
                await asyncio.Event().wait()  # the final edit never returns
            if finalize and len([c for c in self.calls if c[0] == "edit"]) == 2:
                return self._refuse()  # the cancel handler's best-effort edit fails too
            self.messages[message_id] = content
            return SendResult(success=True, message_id=message_id)

    adapter = _StuckFinal(flood_secs=0.0)
    final = "A medium answer that streams in two parts and is then completed."
    consumer = GatewayStreamConsumer(adapter, "chat-1", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=" ▉"))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(final[:25])
    await async_wait_until(lambda: adapter.messages, timeout=5, what="first preview sent")
    consumer.finish(final)
    await asyncio.wait_for(gate.wait(), timeout=5)
    task.cancel()  # what the turn's bounded flush (``_await_stream_task``) does on its timeout
    await asyncio.gather(task, return_exceptions=True)
    assert not consumer.final_response_sent and consumer.message_id == "1"

    response = {"final_response": final}
    turn_ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="chat-1"),
                               session_key="agent:main:telegram:dm:chat-1")
    await asyncio.wait_for(_Harness()._run_agent_mark_streamed_delivery(response, turn_ctx), timeout=5)

    assert response.get("already_sent") is True
    assert list(adapter.messages.values()) == [final]


@pytest.mark.asyncio
async def test_turn_replaced_during_the_flood_wait_publishes_nothing(monkeypatch):
    """``/new`` or ``/stop`` during the flood wait: the old answer is neither edited in nor sent."""
    monkeypatch.setattr("hermes_agent.gateway.run_turn_final_reconcile._FLOOD_WAIT_SLACK_SECS", 0.05)
    adapter = _FloodedChat(flood_secs=0.6)
    final = "The answer begins here and then " + "continues with more words " * 8 + "until the end."
    consumer = await _flood_frozen_turn(adapter, final)
    current = {"run": True}
    consumer._run_still_current = lambda: current["run"]
    frozen = dict(adapter.messages)

    async def reset_mid_wait():
        await async_wait_until(lambda: adapter.calls[-1][0] == "edit" and adapter._flooded(), timeout=5)
        current["run"] = False

    response = {"final_response": final}
    turn_ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="chat-1"),
                               session_key="agent:main:telegram:group:chat-1")
    await asyncio.wait_for(asyncio.gather(
        _Harness()._run_agent_mark_streamed_delivery(response, turn_ctx), reset_mid_wait()), timeout=5)

    assert adapter.messages == frozen  # nothing published after the reset
    assert "already_sent" not in response  # the outer stale-run check discards the result


@pytest.mark.asyncio
async def test_finishing_edit_carries_the_topic_routing_metadata():
    """A split's continuations are sent by the edit path: without the turn's routing metadata a
    retried continuation lands outside the topic."""
    seen = []

    class _Topic(_FloodedChat):
        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            seen.append(metadata)
            return await super().edit_message(chat_id, message_id, content, finalize=finalize)

    adapter = _Topic(flood_secs=0.0)
    from hermes_agent.gateway.run_turn_final_reconcile import edit_preview_to_final
    await edit_preview_to_final(adapter, "chat-1", "1", "answer", {"thread_id": "77"})
    assert seen == [{"thread_id": "77"}]


@pytest.mark.asyncio
async def test_an_edit_that_would_truncate_leaves_the_normal_send():
    """Google Chat, Slack and DingTalk cut an oversized edit short and still report success: a
    reply longer than one message is never "finished in place" there, the normal send owns it."""
    gate = asyncio.Event()

    class _TruncatingEdit(_FloodedChat):
        MAX_MESSAGE_LENGTH = 4000

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, **_kw):
            self.calls.append(("edit", content))
            if finalize and not gate.is_set():
                gate.set()
                await asyncio.Event().wait()  # stuck final edit: the turn's flush cancels the consumer
            if finalize and len([c for c in self.calls if c[0] == "edit"]) == 2:
                return self._refuse()  # the cancel handler's best-effort edit fails too
            self.messages[message_id] = content[:3999] + "…" if len(content) > 4000 else content
            return SendResult(success=True, message_id=message_id)

    adapter = _TruncatingEdit(flood_secs=0.0)
    final = "START " + "word " * 1500 + " END-OF-ANSWER"
    consumer = GatewayStreamConsumer(adapter, "chat-1", StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=" ▉"))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(final[:30])
    await async_wait_until(lambda: adapter.messages, timeout=5, what="first preview sent")
    consumer.finish(final)
    await asyncio.wait_for(gate.wait(), timeout=5)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    response = {"final_response": final}
    turn_ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="chat-1"),
                               session_key="agent:main:googlechat:dm:chat-1")
    await asyncio.wait_for(_Harness()._run_agent_mark_streamed_delivery(response, turn_ctx), timeout=5)

    assert "already_sent" not in response
    assert len([c for c in adapter.calls if c[0] == "edit"]) == 2  # no cut-short finishing edit was tried


def test_slack_measures_the_formatted_text_its_edit_cuts():
    """Slack escapes ``& < >`` before ``chat.update`` keeps one 39k chunk: a raw reply under the
    limit that formats past it is never finished in place, where its tail would be lost."""
    from hermes_agent.gateway.config import PlatformConfig
    from hermes_agent.gateway.run_turn_final_reconcile import owns_reconcilable_preview
    from plugins.platforms.slack.adapter import SlackAdapter

    slack = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    preview = SimpleNamespace(message_id="1700000000.000100", adapter=slack)
    heavy = ("a & b < c > d " * 3000)[:38990]
    assert len(heavy) < slack.MAX_MESSAGE_LENGTH < len(slack.format_message(heavy))
    assert not owns_reconcilable_preview(preview, "C1", heavy)
    assert owns_reconcilable_preview(preview, "C1", "a & b < c " * 100)


@pytest.mark.asyncio
async def test_a_replaced_turn_and_a_receiptless_edit_are_distinguishable():
    """The queued lane drops the reply only for a replaced turn; an edit with no receipt is not that."""
    from hermes_agent.gateway.run_turn_final_reconcile import TURN_REPLACED, edit_preview_to_final

    class _NoReceipt:
        async def edit_message(self, chat_id, message_id, content, *, finalize=False):
            return None

    assert await edit_preview_to_final(_NoReceipt(), "c", "1", "a", still_current=lambda: False) is TURN_REPLACED
    assert await edit_preview_to_final(_NoReceipt(), "c", "1", "a") is not TURN_REPLACED
