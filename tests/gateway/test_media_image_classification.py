"""Tests for gateway media image-classification in _build_media_placeholder.

Covers the fix that stops upcasting non-image attachments to "image" just
because the event's message_type is PHOTO. A PHOTO event can carry a document
(text/*, application/*) or a mixed album; those must not be labelled as images.
"""
import os
import sys
import types

import pytest

# gateway.run imports MessageType from gateway.platforms.base
from gateway.platforms.base import MessageType
from gateway.run import _build_media_placeholder


def _event(media_urls, media_types, message_type):
    return types.SimpleNamespace(
        media_urls=media_urls,
        media_types=media_types,
        message_type=message_type,
    )


def test_image_mime_is_classified_as_image():
    ev = _event(["/tmp/a.png"], ["image/png"], MessageType.PHOTO)
    assert "[User sent an image: /tmp/a.png]" in _build_media_placeholder(ev)


def test_photo_event_with_text_document_is_not_upcast_to_image():
    # A PHOTO event carrying a text document must NOT be called an image.
    ev = _event(["/tmp/notes.txt"], ["text/plain"], MessageType.PHOTO)
    out = _build_media_placeholder(ev)
    assert "image" not in out
    assert "[User sent a file: /tmp/notes.txt]" in out


def test_photo_event_with_application_document_is_not_upcast():
    ev = _event(["/tmp/report.pdf"], ["application/pdf"], MessageType.PHOTO)
    out = _build_media_placeholder(ev)
    assert "image" not in out


def test_photo_album_with_missing_mime_is_not_upcast():
    # Multiple attachments + missing MIME → don't blanket-upcast the album.
    ev = _event(
        ["/tmp/1.bin", "/tmp/2.bin"],
        ["", ""],
        MessageType.PHOTO,
    )
    out = _build_media_placeholder(ev)
    assert "image" not in out


def test_photo_single_missing_mime_with_image_ext_is_image():
    # Single attachment, missing MIME, but an image extension → image.
    ev = _event(["/tmp/pic.jpg"], [""], MessageType.PHOTO)
    assert "[User sent an image: /tmp/pic.jpg]" in _build_media_placeholder(ev)


def test_audio_and_video_still_classified():
    ev_a = _event(["/tmp/a.mp3"], ["audio/mpeg"], MessageType.AUDIO)
    assert "[User sent audio: /tmp/a.mp3]" in _build_media_placeholder(ev_a)
    ev_v = _event(["/tmp/v.mp4"], ["video/mp4"], MessageType.VIDEO)
    assert "[User sent a video: /tmp/v.mp4]" in _build_media_placeholder(ev_v)
