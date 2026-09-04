"""Regression tests for mixed-attachment routing in gateway/run.py.

Issue #25935: when a message mixes a real image with a document (e.g. a .md
brief), Discord types the whole message MessageType.PHOTO. The per-attachment
loops must classify each attachment by its OWN mimetype:

  * A document must NOT be swept into image_paths just because the message-level
    type is PHOTO — mislabelling it as an image sent its bytes to the vision
    endpoint, which rejected them with a non-retryable HTTP 400 and killed the
    whole turn ("Could not process image").
  * That same document must STILL reach the agent as a readable cached file via
    the document context-note path, even though the message-level type isn't
    DOCUMENT.

The message-level fallback (PHOTO/VOICE/AUDIO/VIDEO) is preserved only for
attachments whose per-file mimetype is unknown (empty) — platforms that don't
populate media_types.
"""

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, merge_pending_message_event
from gateway.run import (
    GatewayRunner,
    _build_media_placeholder,
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_video,
)
from gateway.session import SessionSource


def _evt(media_urls, media_types, message_type):
    return SimpleNamespace(
        media_urls=media_urls,
        media_types=media_types,
        message_type=message_type,
    )


# ─── per-attachment classification helpers ───────────────────────────────────


def test_image_trusts_own_mime_over_photo_message_type():
    evt = _evt(["/c/pic.png", "/c/brief.md"], ["image/png", "text/markdown"], MessageType.PHOTO)
    assert _event_media_is_image(evt, 0) is True
    # The document must NOT be promoted to an image by the PHOTO fallback.
    assert _event_media_is_image(evt, 1) is False


# ─── _build_media_placeholder ────────────────────────────────────────────────


def test_placeholder_document_in_photo_message_is_not_an_image():
    evt = _evt(["/c/product.png", "/c/brief.md"], ["image/png", "text/markdown"], MessageType.PHOTO)
    out = _build_media_placeholder(evt)
    assert "[User sent an image: /c/product.png]" in out
    assert "[User sent an image: /c/brief.md]" not in out
    assert "[User sent a file: /c/brief.md]" in out


@pytest.mark.asyncio
async def test_mixed_document_event_preserves_audio_as_non_stt_file_path():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    runner.adapters = {}
    runner._pending_native_image_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="mixed-audio",
        chat_type="dm",
        user_id="42",
        user_name="Tester",
    )
    event = MessageEvent(
        text="inspect both",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/cache/audio.mp3", "/cache/report.pdf"],
        media_types=["audio/mpeg", "application/pdf"],
    )

    prepared = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert prepared is not None
    assert "/cache/audio.mp3" in prepared
    assert "/cache/report.pdf" in prepared
    assert "transcrib" in prepared.lower()


def test_pending_media_merge_preserves_per_attachment_inline_contract():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="merge",
        chat_type="dm",
        user_id="42",
    )
    existing = MessageEvent(
        text="first",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/cache/image.png"],
        media_types=["image/png"],
    )
    incoming = MessageEvent(
        text="second",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=["/cache/notes.txt"],
        media_types=["text/plain"],
        media_text_inlined=[False],
    )
    pending = {"session": existing}

    merge_pending_message_event(pending, "session", incoming)

    assert existing.media_urls == ["/cache/image.png", "/cache/notes.txt"]
    assert existing.media_text_inlined == [None, False]
