"""Cached reply attachments are file references, never inlined document text."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter, CachedMedia
from gateway.platforms.event import MessageEvent
from gateway.session_envelope import _media_clause
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [[], [True]])
async def test_replied_document_preserves_existing_attachment_flags(flags, tmp_path):
    adapter = object.__new__(TelegramAdapter)
    cached = CachedMedia(str(tmp_path / "reply.txt"), "text/plain", "document", "reply.txt")
    adapter._download_observed_media = AsyncMock(return_value=("ok", cached))
    original = str(tmp_path / "original.txt")
    event = MessageEvent(text="read the reply", media_urls=[original],
                         media_types=["text/plain"], media_text_inlined=flags)
    await adapter._cache_replied_media(SimpleNamespace(reply_to_message=object()), event)
    assert _media_clause(event) is None
    assert event.media_urls == [original, cached.path]
    assert event.media_text_inlined == (flags or [None]) + [False]
    assert "read the reply" in event.text and cached.path in event.text


@pytest.mark.asyncio
async def test_two_document_replies_batch_into_an_admissible_turn(tmp_path):
    adapter = object.__new__(TelegramAdapter)
    adapter._drop_unresolved = lambda event: False
    adapter._text_batch_key = lambda event: "chat"
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._flush_text_batch = AsyncMock()
    adapter.handle_message = AsyncMock()
    paths = [str(tmp_path / f"reply-{i}.txt") for i in range(2)]
    adapter._download_observed_media = AsyncMock(side_effect=[
        ("ok", CachedMedia(path, "text/plain", "document", f"reply-{i}.txt"))
        for i, path in enumerate(paths)
    ])
    for i in range(2):
        event = MessageEvent(text=f"read reply {i}")
        await adapter._cache_replied_media(SimpleNamespace(reply_to_message=object()), event)
        BasePlatformAdapter._enqueue_text_event(adapter, event)
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())
    await adapter._flush_text_batch_now("chat")
    batch = adapter.handle_message.call_args.args[0]
    assert _media_clause(batch) is None
    assert batch.media_urls == paths
    assert batch.media_text_inlined == [False, False]
