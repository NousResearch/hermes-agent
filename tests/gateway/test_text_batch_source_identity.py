import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.helpers import TextBatchAggregator
from gateway.session import SessionSource


def _event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_shared_text_batch_marks_composite_source_identity_ambiguous():
    batcher = TextBatchAggregator(handler=AsyncMock(), batch_delay=60)

    batcher.enqueue(_event("first"), "session")
    batcher.enqueue(_event("second"), "session")

    pending = batcher._pending["session"]
    assert pending.text == "first\nsecond"
    assert pending.metadata["source_identity_ambiguous"] is True

    batcher.cancel_all()
    await asyncio.sleep(0)
