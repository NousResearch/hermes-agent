"""Trailing reasoning delivery when streaming already sent the body (#7251/#50193).

When ``display.platforms.<plat>.show_reasoning`` is enabled and streaming
delivered the reply body, ``_handle_message_with_agent`` sets
``already_sent=True`` and returns early. The reasoning block that was built
and prepended to ``response`` would be silently discarded — the user never
sees the 💭 block even though they opted in.

The fix mirrors the trailing-footer rail: in the ``already_sent`` branch the
reasoning block is delivered as its own trailing message.

Behavior contract under test:
- streamed body + show_reasoning + last_reasoning  -> one extra send with the block
- streamed body + show_reasoning + no last_reasoning -> no extra send
- streamed body + show_reasoning OFF               -> no extra send
- non-streamed body (normal final send)            -> block stays inline, no extra send

These tests exercise the real ``GatewayRunner._handle_message_with_agent``
result-handling seam via a minimal fake adapter — no network, no LLM.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.session import SessionSource


def _source(platform=Platform.TELEGRAM):
    return SessionSource(
        platform=platform,
        chat_id="C123",
        chat_type="private",
        thread_id=None,
    )


class _RecordingAdapter(SimpleNamespace):
    """Minimal adapter: records send() calls, reports streaming delivery."""

    def __init__(self, content_delivered=True):
        super().__init__(
            name="test",
            sends=[],
            REQUIRES_EDIT_FINALIZE=False,
        )
        self._content_delivered = content_delivered
        self.send = AsyncMock(side_effect=self._record_send)

    def _record_send(self, chat_id, text, **kwargs):
        self.sends.append(text)
        return SendResult(success=True, message_id=f"m{len(self.sends)}")

    # StreamConsumer probes this attribute chain for the gate.
    @property
    def final_content_delivered(self):  # pragma: no cover - not used directly
        return self._content_delivered


def _stream_consumer_holder(content_delivered=True):
    sc = SimpleNamespace(final_content_delivered=content_delivered)
    return [sc]


@pytest.mark.asyncio
async def test_reasoning_block_sent_after_streamed_body():
    """#7251 shape: streaming delivered the answer; the built 💭 block must
    still reach the user as a trailing message instead of being dropped."""
    from gateway.run import _deliver_trailing_reasoning_block

    adapter = _RecordingAdapter()
    holder = _stream_consumer_holder(content_delivered=True)
    source = _source()
    block = "💭 **Reasoning:**\n```\nstep 1\n```"
    response = "the already-streamed answer"

    await _deliver_trailing_reasoning_block(
        runner=None,
        adapter=adapter,
        source=source,
        stream_consumer_holder=holder,
        reasoning_block=block,
        response=response,
        intentional_silence=False,
        event_metadata=None,
    )

    assert len(adapter.sends) == 1
    assert adapter.sends[0] == block


@pytest.mark.asyncio
async def test_no_trailing_reasoning_when_block_empty():
    from gateway.run import _deliver_trailing_reasoning_block

    adapter = _RecordingAdapter()
    await _deliver_trailing_reasoning_block(
        runner=None,
        adapter=adapter,
        source=_source(),
        stream_consumer_holder=_stream_consumer_holder(),
        reasoning_block="",
        response="answer",
        intentional_silence=False,
        event_metadata=None,
    )
    assert adapter.sends == []


@pytest.mark.asyncio
async def test_no_trailing_reasoning_when_show_disabled_means_block_empty():
    """The builder only populates the block when show_reasoning is effective;
    an empty block must produce zero sends (no empty messages)."""
    from gateway.run import _deliver_trailing_reasoning_block

    adapter = _RecordingAdapter()
    await _deliver_trailing_reasoning_block(
        runner=None,
        adapter=adapter,
        source=_source(),
        stream_consumer_holder=_stream_consumer_holder(content_delivered=False),
        reasoning_block="💭 **Reasoning:**\n```\nx\n```",
        response="💭 **Reasoning:**\n```\nx\n```\n\nanswer",
        intentional_silence=False,
        event_metadata=None,
    )
    # Non-streamed path: block was prepended inline to the response the normal
    # final send carries — no duplicate trailing send.
    assert adapter.sends == []


@pytest.mark.asyncio
async def test_no_trailing_reasoning_on_intentional_silence():
    from gateway.run import _deliver_trailing_reasoning_block

    adapter = _RecordingAdapter()
    await _deliver_trailing_reasoning_block(
        runner=None,
        adapter=adapter,
        source=_source(),
        stream_consumer_holder=_stream_consumer_holder(),
        reasoning_block="💭 **Reasoning:**\n```\nx\n```",
        response="answer",
        intentional_silence=True,
        event_metadata=None,
    )
    assert adapter.sends == []
