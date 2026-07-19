"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

from types import SimpleNamespace

from gateway.attachment_message_runtime_service import (
    collect_audio_paths,
    document_context_note,
    has_visible_image_attachments,
    prepend_document_context_notes,
)
from gateway.platforms.base import MessageType


def test_has_visible_image_attachments_ignores_animated_images():
    attachments = [
        SimpleNamespace(kind="image", is_animated=True),
        SimpleNamespace(kind="image", is_animated=False),
    ]

    assert has_visible_image_attachments(attachments) is True


def test_collect_audio_paths_prefers_local_or_analysis_refs():
    attachments = [
        SimpleNamespace(
            local_path="",
            analysis_ref="/tmp/audio.ogg",
            mime_type="audio/ogg",
        ),
        SimpleNamespace(
            local_path="/tmp/image.jpg",
            analysis_ref="",
            mime_type="image/jpeg",
        ),
    ]

    assert collect_audio_paths(
        attachments,
        message_type=MessageType.TEXT,
        voice_type=MessageType.VOICE,
        audio_type=MessageType.AUDIO,
    ) == ["/tmp/audio.ogg"]


def test_document_context_note_infers_text_document_from_extension():
    note = document_context_note(
        "/tmp/doc_123456789abc_notes.md",
        "application/octet-stream",
    )

    assert note is not None
    assert "text document" in note
    assert "notes.md" in note


def test_prepend_document_context_notes_skips_non_document_messages():
    attachment = SimpleNamespace(
        local_path="/tmp/doc_123456789abc_notes.md",
        analysis_ref="",
        mime_type="text/markdown",
    )

    assert prepend_document_context_notes(
        "body",
        attachments=[attachment],
        message_type=MessageType.TEXT,
        document_type=MessageType.DOCUMENT,
    ) == "body"
