"""A file queued via an explicit MEDIA: tag must not be uploaded a second time just because
its bare path also appears in the response text. This regressed when ``show_reasoning``
prepends the model's reasoning — which often names the very files it is about to attach —
so ``extract_local_files`` re-found them alongside the MEDIA tags and Discord received each
clip twice (2 clips -> 4 attachments). Cross-dedup within the response drops the redundant
bare-path copy; MEDIA tags remain the explicit delivery intent."""

import os

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _clip(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 256)
    return str(p)


async def _extract(text: str):
    event = MessageEvent(text="send it", source=SessionSource(platform=Platform.DISCORD, chat_id="c"))
    return await _Adapter()._extract_response_content(text, event, "", is_ephemeral_response=False)


@pytest.mark.asyncio
async def test_bare_path_in_reasoning_not_reattached_alongside_media(tmp_path):
    n01 = _clip(tmp_path, "N01.mp4")
    n02 = _clip(tmp_path, "N02.mp4")
    # Reasoning subtext (show_reasoning) names the files bare; the answer attaches them via MEDIA.
    response = (f"-# Let me attach the 2 clips. Paths:\n-# {n01}\n-# {n02}\n\n"
                f"Here are your clips!\nMEDIA:{n01}\nMEDIA:{n02}")
    out = await _extract(response)
    assert [p for p, _ in out.media_files] == [n01, n02]
    assert out.local_files == [], "bare paths already queued via MEDIA must not upload again"


@pytest.mark.asyncio
async def test_bare_path_without_media_still_delivered(tmp_path):
    n01 = _clip(tmp_path, "N01.mp4")
    n02 = _clip(tmp_path, "N02.mp4")
    out = await _extract(f"clips:\n{n01}\n{n02}")
    assert out.media_files == []
    assert out.local_files == [n01, n02]


@pytest.mark.asyncio
async def test_distinct_media_and_bare_path_both_delivered(tmp_path):
    n01 = _clip(tmp_path, "N01.mp4")
    n02 = _clip(tmp_path, "N02.mp4")
    out = await _extract(f"here MEDIA:{n01}\nand also {n02}")
    assert [p for p, _ in out.media_files] == [n01]
    assert out.local_files == [n02], "cross-dedup must not drop a genuinely different file"


@pytest.mark.asyncio
async def test_duplicate_media_tags_deduplicated(tmp_path):
    n01 = _clip(tmp_path, "N01.mp4")
    out = await _extract(f"MEDIA:{n01}\nMEDIA:{n01}")
    assert [p for p, _ in out.media_files] == [n01]
    assert out.local_files == []