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

from gateway.platforms.base import MessageType
from gateway.run import (
    _build_media_placeholder,
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_video,
)


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


def test_unknown_mime_falls_back_to_photo_message_type():
    # Platforms that don't populate media_types rely on the message-level type.
    evt = _evt(["/c/photo.jpg"], [""], MessageType.PHOTO)
    assert _event_media_is_image(evt, 0) is True


def test_audio_classified_per_attachment():
    evt = _evt(["/c/clip.ogg", "/c/shot.png"], ["audio/ogg", "image/png"], MessageType.PHOTO)
    assert _event_media_is_audio(evt, 0) is True
    assert _event_media_is_audio(evt, 1) is False
    assert _event_media_is_image(evt, 1) is True


def test_video_classified_per_attachment():
    evt = _evt(["/c/movie.mp4", "/c/notes.md"], ["video/mp4", "text/markdown"], MessageType.PHOTO)
    assert _event_media_is_video(evt, 0) is True
    assert _event_media_is_video(evt, 1) is False


# ─── _build_media_placeholder ────────────────────────────────────────────────


def test_placeholder_document_in_photo_message_is_not_an_image():
    evt = _evt(["/c/product.png", "/c/brief.md"], ["image/png", "text/markdown"], MessageType.PHOTO)
    out = _build_media_placeholder(evt)
    assert "[User sent an image: /c/product.png]" in out
    assert "[User sent an image: /c/brief.md]" not in out
    assert "[User sent a file: /c/brief.md]" in out


def test_placeholder_image_with_unknown_mime_uses_photo_fallback():
    evt = _evt(["/c/photo.jpg"], [""], MessageType.PHOTO)
    out = _build_media_placeholder(evt)
    assert "[User sent an image: /c/photo.jpg]" in out
