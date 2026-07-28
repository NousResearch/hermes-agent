"""Telegram media captions must be truncated by UTF-16 code units, not code points.

Telegram's caption cap (1024) is measured in UTF-16 code units, the same as
the message-length cap (see ``gateway.platforms.base.utf16_len``). Characters
outside the Basic Multilingual Plane (emoji, rare CJK) are surrogate pairs and
consume two UTF-16 code units each even though Python's ``len()``/``[:n]``
count them as one. A plain ``caption[:1024]`` can therefore pass a caption
whose true UTF-16 length is up to 2048 straight to the Bot API, which Telegram
rejects outright — the media never sends.
"""
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.base import utf16_len  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

# U+1F600 (astral plane) — one Python code point, two UTF-16 code units.
_ASTRAL_EMOJI = "\U0001F600"


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    return adapter


@pytest.fixture
def tmp_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"data")
    return str(path)


@pytest.mark.asyncio
async def test_send_document_caption_truncated_by_utf16_units(tmp_file):
    """1024 astral-plane code points (2048 UTF-16 units) must be cut to <=1024
    UTF-16 units, not passed through unchanged because len() sees only 1024."""
    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock(return_value=MagicMock(message_id=1))
    caption = _ASTRAL_EMOJI * 1024  # code-point len == 1024, UTF-16 len == 2048

    await adapter.send_document("123", tmp_file, caption=caption)

    sent_caption = adapter._bot.send_document.call_args.kwargs["caption"]
    assert utf16_len(sent_caption) <= 1024
    # Must not split a surrogate pair: re-encoding must round-trip cleanly.
    sent_caption.encode("utf-16-le").decode("utf-16-le")


@pytest.mark.asyncio
async def test_send_document_short_caption_untouched(tmp_file):
    """A caption well under the limit is passed through unchanged."""
    adapter = _make_adapter()
    adapter._bot.send_document = AsyncMock(return_value=MagicMock(message_id=1))

    await adapter.send_document("123", tmp_file, caption="short caption")

    sent_caption = adapter._bot.send_document.call_args.kwargs["caption"]
    assert sent_caption == "short caption"


@pytest.mark.asyncio
async def test_send_image_file_caption_truncated_by_utf16_units(tmp_path):
    """send_image_file shares the same caption cap and must be equally safe."""
    img_path = tmp_path / "pic.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _make_adapter()
    adapter._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=1))
    caption = _ASTRAL_EMOJI * 1024

    await adapter.send_image_file("123", str(img_path), caption=caption)

    sent_caption = adapter._bot.send_photo.call_args.kwargs["caption"]
    assert utf16_len(sent_caption) <= 1024
    sent_caption.encode("utf-16-le").decode("utf-16-le")
