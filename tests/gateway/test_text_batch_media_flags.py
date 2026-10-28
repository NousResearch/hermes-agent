"""Text batching must preserve attachment semantics at native admission."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session_envelope import _media_clause
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [([False], [True]), ([], [False]), ([True], []), ([], [])])
async def test_text_batch_dispatch_preserves_each_attachment(flags, tmp_path):
    adapter = object.__new__(TelegramAdapter)
    adapter._drop_unresolved = lambda event: False
    adapter._text_batch_key = lambda event: "chat"
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._flush_text_batch = AsyncMock()
    adapter.handle_message = AsyncMock()
    paths = [str(tmp_path / f"reply-{i}.txt") for i in range(2)]
    for index, inlined in enumerate(flags):
        event = MessageEvent(text=f"part {index}", media_urls=[paths[index]],
                             media_types=["text/plain"], media_text_inlined=inlined,
                             media_filenames=[f"reply-{index}.txt"])
        BasePlatformAdapter._enqueue_text_event(adapter, event)
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    await adapter._flush_text_batch_now("chat")
    batch = adapter.handle_message.call_args.args[0]
    assert _media_clause(batch) is None
    assert batch.media_urls == paths
    assert batch.media_filenames == ["reply-0.txt", "reply-1.txt"]
    assert batch.media_text_inlined == [value for flag in flags for value in (flag or [None])]
    assert "part 0" in batch.text and "part 1" in batch.text
