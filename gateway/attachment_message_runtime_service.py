"""Shared runtime helpers for attachment-driven message enrichment."""

from __future__ import annotations

import mimetypes
import os
import re
from typing import Any


_TEXT_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}


def has_visible_image_attachments(attachments: list[Any]) -> bool:
    """Return True when the event includes at least one non-animated image."""

    return any(
        getattr(attachment, "kind", None) == "image"
        and not bool(getattr(attachment, "is_animated", False))
        for attachment in attachments
    )


def collect_audio_paths(
    attachments: list[Any],
    *,
    message_type: Any,
    voice_type: Any,
    audio_type: Any,
) -> list[str]:
    """Collect audio/voice attachment paths for transcription."""

    audio_paths: list[str] = []
    for attachment in attachments:
        path = str(
            getattr(attachment, "local_path", None)
            or getattr(attachment, "analysis_ref", None)
            or ""
        ).strip()
        mime_type = str(getattr(attachment, "mime_type", None) or "").strip()
        is_audio = mime_type.startswith("audio/") or message_type in (
            voice_type,
            audio_type,
        )
        if is_audio and path:
            audio_paths.append(path)
    return audio_paths


def _infer_document_mime_type(path: str, mime_type: str) -> str:
    normalized = str(mime_type or "").strip()
    if normalized not in ("", "application/octet-stream"):
        return normalized

    extension = os.path.splitext(path)[1].lower()
    if extension in _TEXT_DOCUMENT_EXTENSIONS:
        return "text/plain"
    guessed, _ = mimetypes.guess_type(path)
    return guessed or normalized


def _display_document_name(path: str) -> str:
    basename = os.path.basename(path)
    parts = basename.split("_", 2)
    display_name = parts[2] if len(parts) >= 3 else basename
    return re.sub(r"[^\w.\- ]", "_", display_name)


def document_context_note(path: str, mime_type: str) -> str | None:
    """Return the user-facing note injected ahead of document messages."""

    normalized_path = str(path or "").strip()
    if not normalized_path:
        return None

    normalized_mime_type = _infer_document_mime_type(normalized_path, mime_type)
    if not normalized_mime_type.startswith(("application/", "text/")):
        return None

    display_name = _display_document_name(normalized_path)
    if normalized_mime_type.startswith("text/"):
        return (
            f"[The user sent a text document: '{display_name}'. "
            f"Its content has been included below. "
            f"The file is also saved at: {normalized_path}]"
        )
    return (
        f"[The user sent a document: '{display_name}'. "
        f"The file is saved at: {normalized_path}. "
        f"Ask the user what they'd like you to do with it.]"
    )


def prepend_document_context_notes(
    message_text: str,
    *,
    attachments: list[Any],
    message_type: Any,
    document_type: Any,
) -> str:
    """Prepend document context notes for document messages."""

    if not attachments or message_type != document_type:
        return message_text

    enriched_text = message_text
    for attachment in attachments:
        path = str(
            getattr(attachment, "local_path", None)
            or getattr(attachment, "analysis_ref", None)
            or ""
        ).strip()
        note = document_context_note(
            path,
            str(getattr(attachment, "mime_type", None) or "").strip(),
        )
        if note:
            enriched_text = f"{note}\n\n{enriched_text}"
    return enriched_text
