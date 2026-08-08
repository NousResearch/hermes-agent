"""A failed *first* send must not silently delete the pre-boundary segment.

Sibling of the #8124 guard.  ``_reset_segment_state`` clears ``_message_id``
at every tool boundary, so the next text segment re-enters the first-send
branch of ``_send_or_edit``.  That branch returns without ever assigning
``_message_id`` when the send fails, which is exactly the case the original
recovery flush excluded (it required ``self._message_id`` to be truthy).  The
result: the paragraph written between two tool calls is buffered, never shown,
and then wiped by the reset — the user sees the reply jump from one tool
bubble to the next with the explanation missing.

The fakes below fail only the *streaming-preview* send path, which the
consumer marks with ``expect_edits`` in its metadata (see
``_metadata_for_send``).  Plain sends — the ones the tail flush and the
commentary use — still succeed.  That mirrors a real platform split: Telegram
keeps editable previews on the legacy send path, so a preview render that the
platform rejects (or rate-limits) can fail for a whole turn while ordinary
messages go through.

Assertions are on delivered payloads, never on internal attributes.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

_SEGMENT_ONE = "Here's what I found in the config, let me check the processes too."
_SEGMENT_TWO = "The gateway is running under systemd."
_COMMENTARY = "Using the shell tool..."


def _is_preview_send(kwargs) -> bool:
    """True for sends the consumer routes through the editable-preview path."""
    metadata = kwargs.get("metadata")
    return isinstance(metadata, dict) and bool(metadata.get("expect_edits"))


def _preview_hostile_adapter():
    """Adapter whose editable-preview sends all fail; plain sends succeed.

    ``adapter.delivered`` collects only the payloads the platform accepted --
    a rejected send is an attempt, not something the user saw, so assertions
    must not read it out of ``send.call_args_list``.
    """
    adapter = MagicMock()
    adapter.delivered = []

    async def _send(**kwargs):
        if _is_preview_send(kwargs):
            return SimpleNamespace(success=False, error="preview send rejected")
        adapter.delivered.append(kwargs["content"])
        return SimpleNamespace(
            success=True, message_id="msg_%d" % len(adapter.delivered),
        )

    adapter.send = AsyncMock(side_effect=_send)
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter


def _delivered_count(adapter, needle):
    return sum(1 for payload in adapter.delivered if needle in payload)


def _consumer(adapter):
    config = StreamConsumerConfig(
        edit_interval=0.01, buffer_threshold=5, cursor=" ▉",
    )
    return GatewayStreamConsumer(adapter, "chat_123", config)


@pytest.mark.asyncio
async def test_failed_first_send_tail_survives_tool_boundary():
    """The segment-break reset must not discard a never-delivered buffer.

    Every preview send fails here, so ``_message_id`` is still ``None`` when
    the tool boundary arrives.  The old guard required a truthy
    ``_message_id``, so it skipped the flush and the reset wiped the only copy
    of the text.
    """
    adapter = _preview_hostile_adapter()
    consumer = _consumer(adapter)

    consumer.on_delta(_SEGMENT_ONE)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.08)
    consumer.on_delta(None)  # tool boundary
    await asyncio.sleep(0.08)
    consumer.on_delta(_SEGMENT_TWO)
    consumer.finish()
    await task

    assert _delivered_count(adapter, "let me check the processes too") == 1, (
        "pre-boundary segment was dropped by the reset: "
        "delivered=%r" % (adapter.delivered,)
    )


@pytest.mark.asyncio
async def test_failed_first_send_tail_survives_commentary_reset():
    """The commentary reset had no guard at all, so it dropped the same buffer.

    ``_send_commentary`` is preceded by an unconditional
    ``_reset_segment_state()``; if the preview send for the current segment
    never became visible, that reset destroys it and the user sees only the
    interim status line.
    """
    adapter = _preview_hostile_adapter()
    consumer = _consumer(adapter)

    consumer.on_delta(_SEGMENT_ONE)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.08)
    consumer.on_commentary(_COMMENTARY)
    await asyncio.sleep(0.08)
    consumer.on_delta(_SEGMENT_TWO)
    consumer.finish()
    await task

    payloads = adapter.delivered
    assert _delivered_count(adapter, "let me check the processes too") == 1, (
        "text buffered before the commentary reset was dropped: "
        "delivered=%r" % (payloads,)
    )
    # The commentary itself must still be delivered, and after the prose it
    # was interrupting.
    assert _COMMENTARY in payloads
    prose_at = next(
        i for i, p in enumerate(payloads) if "let me check the processes too" in p
    )
    assert prose_at < payloads.index(_COMMENTARY)


@pytest.mark.asyncio
async def test_no_edit_sentinel_does_not_double_send():
    """The ``__no_edit__`` exclusion must survive the widened guard.

    When a platform accepts a message but returns no editable id, the
    consumer parks ``_message_id`` on the ``__no_edit__`` sentinel and the
    matching reset preserves the segment so ``_send_fallback_final`` can
    deliver the whole continuation once.  Flushing there would re-send text
    the user has already seen, so dropping the ``_message_id`` truthiness
    term must not drop this exclusion too.
    """
    adapter = MagicMock()
    adapter.delivered = []

    async def _send(**kwargs):
        # Accepted, but with no editable id -- the __no_edit__ trigger.
        adapter.delivered.append(kwargs["content"])
        return SimpleNamespace(success=True, message_id=None)

    adapter.send = AsyncMock(side_effect=_send)
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.MAX_MESSAGE_LENGTH = 4096
    consumer = _consumer(adapter)

    consumer.on_delta(_SEGMENT_ONE)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.08)
    consumer.on_delta(None)  # tool boundary
    await asyncio.sleep(0.08)
    consumer.on_delta(" " + _SEGMENT_TWO)
    consumer.finish()
    await task

    payloads = adapter.delivered
    assert _delivered_count(adapter, "let me check the processes too") == 1, (
        "the __no_edit__ segment was delivered more than once: "
        "delivered=%r" % (payloads,)
    )
    assert _delivered_count(adapter, _SEGMENT_TWO) == 1, (
        "post-boundary continuation was lost or duplicated: "
        "delivered=%r" % (payloads,)
    )
