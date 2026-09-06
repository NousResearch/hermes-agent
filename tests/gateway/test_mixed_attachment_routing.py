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
from unittest.mock import AsyncMock, MagicMock, call, patch

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "message_type",
        "media_urls",
        "media_types",
        "allowed",
        "expected_route",
        "expected_stt_paths",
    ),
    [
        (
            MessageType.VOICE,
            ["/cache/voice.ogg"],
            ["audio/ogg"],
            False,
            "stt",
            ["/cache/voice.ogg"],
        ),
        (
            MessageType.AUDIO,
            ["/cache/audio.m4a"],
            ["audio/mp4"],
            True,
            "stt",
            ["/cache/audio.m4a"],
        ),
        (
            MessageType.AUDIO,
            ["/cache/audio.mp3"],
            ["audio/mpeg"],
            False,
            "file",
            [],
        ),
        (
            MessageType.DOCUMENT,
            ["/cache/audio.mp3"],
            ["audio/mpeg"],
            True,
            "stt",
            ["/cache/audio.mp3"],
        ),
        (
            MessageType.AUDIO,
            ["/cache/video.mp4"],
            ["video/mp4"],
            True,
            "video",
            [],
        ),
        (
            MessageType.DOCUMENT,
            ["/cache/application.mp4"],
            ["application/mp4"],
            True,
            "document",
            [],
        ),
        (
            MessageType.PHOTO,
            ["/cache/image.png", "/cache/audio.mp3", "/cache/report.pdf"],
            ["image/png", "audio/mpeg", "application/pdf"],
            True,
            "mixed",
            ["/cache/audio.mp3"],
        ),
    ],
    ids=["voice", "audio-mp4", "denied", "document-audio", "video", "application", "mixed"],
)
async def test_audio_stt_routing_uses_source_policy_and_attachment_mime(
    message_type,
    media_urls,
    media_types,
    allowed,
    expected_route,
    expected_stt_paths,
):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                extra={
                    "transcribe_audio_attachment_channels": ["mixed"] if allowed else []
                }
            )
        },
    )
    runner.adapters = {}
    runner._decide_image_input_mode = MagicMock(return_value="text")
    runner._enrich_message_with_vision = AsyncMock(
        return_value="[vision pipeline]\n\ncaption"
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="mixed",
        chat_type="dm",
        user_id="42",
    )
    event = MessageEvent(
        text="caption",
        message_type=message_type,
        source=source,
        media_urls=media_urls,
        media_types=media_types,
    )

    with (
        patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": "audio transcript"},
        ) as transcribe,
        patch("tools.transcription_tools.transcribe_audio_local_fallback") as fallback,
    ):
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[]
        )

    assert transcribe.call_args_list == [
        call(path, None, "gateway") for path in expected_stt_paths
    ]
    fallback.assert_not_called()
    if expected_route in {"stt", "mixed"}:
        assert '"audio transcript"' in prepared
        assert "audio file attachment" not in prepared
    if expected_route == "file":
        assert "audio file attachment" in prepared
    if expected_route == "video":
        assert "video attachment" in prepared
    if expected_route == "document":
        assert "The user sent a document" in prepared
    if expected_route == "mixed":
        runner._enrich_message_with_vision.assert_awaited_once()
        assert "report.pdf" in prepared
