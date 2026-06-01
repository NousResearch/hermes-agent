"""Regression tests for _build_media_placeholder mixed-attachment routing.

When a message mixes a real image with a document (e.g. a .md brief), the
whole message is typed MessageType.PHOTO. The per-file loop must classify
each attachment by its OWN mimetype — a document must not be labelled an
image just because the message-level type is PHOTO. Mislabelling a non-image
file as an image caused its bytes to be sent to the vision endpoint, which
rejected them with a non-retryable HTTP 400 and killed the whole turn.

The message-level PHOTO fallback is preserved only for attachments whose
per-file mimetype is unknown (empty) — i.e. platforms that don't populate
media_types.
"""

from types import SimpleNamespace

from gateway.platforms.base import MessageType
from gateway.run import _build_media_placeholder


def test_document_in_photo_message_is_not_labelled_an_image():
    event = SimpleNamespace(
        media_urls=["/cache/product.png", "/cache/brief.md"],
        media_types=["image/png", "text/markdown"],
        message_type=MessageType.PHOTO,
    )
    out = _build_media_placeholder(event)
    assert "[User sent an image: /cache/product.png]" in out
    # The document must NOT be promoted to an image by the PHOTO fallback.
    assert "[User sent an image: /cache/brief.md]" not in out
    assert "[User sent a file: /cache/brief.md]" in out


def test_image_with_unknown_mimetype_still_uses_photo_fallback():
    # Platforms that don't populate media_types rely on the message-level type.
    event = SimpleNamespace(
        media_urls=["/cache/photo.jpg"],
        media_types=[""],
        message_type=MessageType.PHOTO,
    )
    out = _build_media_placeholder(event)
    assert "[User sent an image: /cache/photo.jpg]" in out


def test_audio_and_image_classified_by_own_mimetype():
    event = SimpleNamespace(
        media_urls=["/cache/clip.ogg", "/cache/shot.png"],
        media_types=["audio/ogg", "image/png"],
        message_type=MessageType.PHOTO,
    )
    out = _build_media_placeholder(event)
    assert "[User sent audio: /cache/clip.ogg]" in out
    assert "[User sent an image: /cache/shot.png]" in out
