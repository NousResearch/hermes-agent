"""A segment that delivered the complete answer must survive the segment reset.

Matrix incident (2026-09-30, live room): the reply streamed into the room message, a
trailing step closed the segment, and the turn-final edit was then refused with HTTP 429.
The consumer ended up reporting no delivery at all — and the gateway sent the complete
reply a second time.  The room kept two messages with the same 4690-character body and
neither was redacted.

Root cause: ``_reset_segment_state`` retains the segment's last ACKed text exactly as it
sat on the wire — which mid-segment is cursor-suffixed (``"<answer> ▉"``) — while the
gateway's durable-match predicate (``_run_agent_stream_confirmed_final_delivery`` →
``has_durably_delivered_text``) compares the record to the final text with an exact
``==``.  A retained ``"<answer> ▉"`` therefore never matched, so the retention did nothing
in precisely the case it exists for.

The invariant these tests pin:

    A reply the user already sees on screen must suppress the gateway's corrective send;
    a reply that is only partly on screen must not.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

CURSOR = " ▉"
PREFIX = ("Ack received. Starting the deploy now, this preamble is long enough to open "
          "a streamed message on its own.")
TAIL = " and here is the rest of the answer, generated after the last preview edit."
FULL = PREFIX + TAIL

TICK = 0.12  # let a consumer tick push a frame before the next step


class RefusingFinalizeAdapter(BasePlatformAdapter):
    """Renders the first ``after`` frames, then refuses every later call (flood control)."""

    def __init__(self, *, after: int):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._after = after
        self.wire = []  # (kind, payload) for every frame the platform accepted
        self._next_id = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    def _refused(self) -> bool:
        return len(self.wire) >= self._after

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if self._refused():
            return SendResult(success=False, error="429 too many requests")
        self._next_id += 1
        self.wire.append(("send", content))
        return SendResult(success=True, message_id=f"m-{self._next_id}")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        if self._refused():
            return SendResult(success=False, error="429 too many requests")
        self.wire.append(("edit", content))
        return SendResult(success=True, message_id=message_id)

    def rendered(self, text: str) -> bool:
        """True when an accepted frame carried *text*."""
        return any(
            kind in ("send", "edit") and text.strip() in payload
            for kind, payload in self.wire
        )


async def _drive(adapter, *, segment_break: bool):
    """Stream PREFIX, then TAIL, then close the segment (a trailing tool step) and finish."""
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor=CURSOR, edit_interval=0.0)
    )
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(PREFIX)
    await asyncio.sleep(TICK)
    consumer.on_delta(TAIL)
    await asyncio.sleep(TICK)
    if segment_break:
        consumer.on_segment_break()
        await asyncio.sleep(TICK)
    consumer.finish()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    return consumer


@pytest.mark.asyncio
async def test_segment_reset_keeps_the_delivered_answer_matchable():
    """The complete answer is on screen before the reset; it must stay matchable after it.

    The gateway suppresses its own final send when ``final_response_sent`` is set, or —
    since the reset clears that flag — when ``has_durably_delivered_text(final)`` matches
    what the consumer durably delivered.  A cursor-suffixed record never matches, so the
    gateway re-sends the whole reply beside the message that already shows it.
    """
    adapter = RefusingFinalizeAdapter(after=2)
    consumer = await _drive(adapter, segment_break=True)

    assert adapter.rendered(FULL), (
        "precondition: the complete answer must already be on the wire — "
        f"wire={adapter.wire!r}"
    )
    assert consumer.has_durably_delivered_text(FULL), (
        "the complete answer is on screen, but the retained segment record does not match "
        "it, so the gateway sends the same reply again and the user sees it twice; "
        f"records={consumer._delivered_segment_texts!r} "
        f"(cursor={CURSOR!r}) wire={adapter.wire!r}"
    )


@pytest.mark.asyncio
async def test_segment_reset_does_not_claim_a_prefix_only_delivery():
    """Mirror guard: only the preamble rendered, so the corrective send must still happen.

    The retained record holds the preamble alone.  It must not match the complete final,
    or the gateway would suppress the only delivery of the answer.
    """
    adapter = RefusingFinalizeAdapter(after=1)
    consumer = await _drive(adapter, segment_break=True)

    assert adapter.rendered(PREFIX), (
        f"precondition: the preamble must have rendered — wire={adapter.wire!r}"
    )
    assert not adapter.rendered(TAIL), (
        f"precondition: the tail must NOT have rendered — wire={adapter.wire!r}"
    )
    assert not consumer.has_durably_delivered_text(FULL), (
        "only a preamble is on screen — the record must not match the complete final, or "
        "the gateway's suppressed send would lose the answer; "
        f"records={consumer._delivered_segment_texts!r} wire={adapter.wire!r}"
    )
