"""Localization must retain the inlining flag belonging to each surviving file."""
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent
from gateway.relay.adapter import RelayAdapter
from gateway.session_envelope import _media_clause


@pytest.mark.asyncio
@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("flags", [[], [True, False, True]])
async def test_localization_filters_flags_with_attachments(download, flags, tmp_path):
    adapter = object.__new__(RelayAdapter)
    local = str(tmp_path / "kept.txt")
    client = AsyncMock()
    client.download_with_name.side_effect = [None, (local, "downloaded.txt"), None]
    adapter._get_media_client = lambda: client if download else None
    urls = ["https://connector.example/relay/media/dead", "https://example.com/kept.txt",
            "https://example.com/public.txt"]
    event = MessageEvent(text="read", media_urls=urls,
                         media_types=["application/pdf", "text/plain", "text/plain"],
                         media_filenames=["dropped.pdf", "kept.txt", "public.txt"],
                         media_text_inlined=flags)
    await adapter._localize_inbound_media(event)
    assert _media_clause(event) is None
    assert event.media_urls == [local if download else urls[1], urls[2]]
    assert event.media_types == ["text/plain", "text/plain"]
    assert event.media_filenames == ["downloaded.txt" if download else "kept.txt", "public.txt"]
    assert event.media_text_inlined == (flags[1:] if flags else [None, None])
