"""Regression: Slack native stream frames must stay prefix-stable inside a code block.

``chat.appendStream`` is append-only, so frame N must be a string prefix of frame N+1. The
direct Slack adapter did not declare its drafts prefix-stable, so the stream consumer closed the
open ``` fence on every mid-code-block frame; the next frame then failed the adapter's
``startswith`` check ("stream prefix mismatch"), the stream was sealed short, and the same answer
was posted again through the edit path (a visible duplicate that cuts off mid-snippet).
"""

import asyncio
from itertools import pairwise

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until

_CHUNKS = [
    "Intro text.\n\n```python\ncountries = [\n",
    '    {"country": "India",\n',
    '     "population": 341_784_',
    "857},\n]\n```\n",
]


def _adapter_with_slack_flags():
    """Fake draft adapter that takes its stream-stability flags from the real SlackAdapter class."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from plugins.platforms.slack.adapter import SlackAdapter

    cls = type("SlackLikeAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 39000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.DRAFT_STREAM_PREFIX_STABLE = SlackAdapter.DRAFT_STREAM_PREFIX_STABLE
    adapter.draft_stream_is_message = getattr(SlackAdapter, "draft_stream_is_message", False)
    adapter.draft_frames = []
    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: True

    async def send_draft(*, chat_id, draft_id, content, metadata=None):
        adapter.draft_frames.append(content)
        return SendResult(success=True, message_id=None)

    async def send(chat_id, content, reply_to=None, metadata=None, **kw):
        return SendResult(success=True, message_id="ts1")

    async def edit_message(chat_id, message_id, content, **kw):
        return SendResult(success=True, message_id=message_id)

    adapter.send_draft = send_draft
    adapter.send = send
    adapter.edit_message = edit_message
    return adapter


@pytest.mark.asyncio
async def test_slack_draft_frames_extend_each_other_inside_a_code_block():
    adapter = _adapter_with_slack_flags()
    cfg = StreamConsumerConfig(
        transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor="",
    )
    consumer = GatewayStreamConsumer(adapter, "D1", cfg)
    task = asyncio.create_task(consumer.run())
    for chunk in _CHUNKS:
        seen = len(adapter.draft_frames)
        consumer.on_delta(chunk)
        await async_wait_until(lambda: len(adapter.draft_frames) > seen, what="a draft frame for the chunk")
    consumer.finish()
    await task

    frames = adapter.draft_frames
    assert len(frames) >= 3, "the code block must have been streamed across several frames"
    for previous, current in pairwise(frames):
        assert current.startswith(previous), (
            f"frame is not a prefix extension: ...{previous[-30:]!r} -> ...{current[-30:]!r}"
        )
    assert not any(frame.rstrip().endswith("```") for frame in frames[:-1] if frame.count("```") % 2), (
        "a mid-code-block frame must not carry a synthetic closing fence"
    )
