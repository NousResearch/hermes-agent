"""Crash between the first visible stream effect and the output-owner write.

The witness drives the real ``GatewayStreamConsumer`` transport seam: the
adapter accepts a send, then the process "dies" before any delivery-ledger
obligation exists.  A fresh recovery pass must not replay the turn.
"""
import asyncio

import pytest

from gateway import delivery_ledger as dl, shutdown_flush as spool
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_durable_batch_transfer import inputs, world  # noqa: F401


class Crash(BaseException):
    """Process death; not swallowed by the consumer's Exception handlers."""


def consumer_for(adapter, batch, sent):
    async def send(*args, **kwargs):
        sent.append(kwargs.get("content") or args[1:])
        raise Crash()  # transport accepted the send, then the process died

    adapter.send = send
    return GatewayStreamConsumer(
        adapter=adapter, chat_id="fixture-chat",
        config=StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1, cursor=""),
        on_before_first_effect=lambda: asyncio.to_thread(
            spool.mark_durable_inbound_output_started, batch))


@pytest.mark.asyncio
async def test_crash_after_first_stream_send_is_not_replayed(world):
    adapter, source, key = world
    events, batch = inputs(source, key)
    sent = []
    consumer = consumer_for(adapter, batch, sent)
    consumer.on_delta("partial answer")
    consumer.finish()
    with pytest.raises(Crash):
        await consumer.run()
    assert sent, "the stream effect must have reached the transport"
    # Fresh runner: no output owner exists, yet the turn must not execute again.
    assert spool.recover_durable_inbound_events() == []
    assert dl.sweep_recoverable() == []
    names = sorted(p.name for p in spool._get_flush_dir().glob("*inbound-*.json"))
    assert len(names) == len(events) and all(n.startswith("indeterminate-") for n in names)
    # A fresh same-text input is still admitted.
    fresh = type(events[0])(text=events[0].text, source=source, message_id="fresh")
    assert spool.record_durable_inbound_event(key, fresh)
    assert [e.message_id for e in spool.recover_durable_inbound_events()] == ["fresh"]


@pytest.mark.asyncio
async def test_no_stream_effect_stays_replayable(world):
    adapter, source, key = world
    events, batch = inputs(source, key)
    sent = []
    consumer = consumer_for(adapter, batch, sent)
    consumer.finish()  # nothing streamed: the gateway's own final send owns delivery
    await consumer.run()
    assert not sent
    assert len(spool.recover_durable_inbound_events()) == len(events)


@pytest.mark.asyncio
async def test_marked_spool_is_retired_by_obligation_write(world):
    adapter, source, key = world
    events, batch = inputs(source, key)
    assert spool.mark_durable_inbound_output_started(batch) == len(events)
    assert spool.mark_durable_inbound_output_started(batch) == 0  # idempotent
    await adapter._record_delivery_obligation(batch, key, "answer", adapter, False)
    assert not list(spool._get_flush_dir().glob("*inbound-*.json"))
    assert spool.recover_durable_inbound_events() == []
