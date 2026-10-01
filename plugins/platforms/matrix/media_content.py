"""Classify transport filenames before constructing Matrix media messages."""

from __future__ import annotations

import mimetypes
from pathlib import Path

_MATRIX_IMAGE_FILENAME_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic", ".heif", ".avif"})
_MATRIX_MEDIA_FILENAME_EXTS = frozenset({
    ".ogg", ".oga", ".opus", ".m4a", ".mp3", ".wav", ".flac", ".aac", ".amr", ".mp4", ".webm", ".mov", ".mkv"})
def _looks_like_matrix_image_filename(text: str) -> bool:
    return _looks_like_transport_filename(text, "image/", _MATRIX_IMAGE_FILENAME_EXTS)


def _looks_like_transport_filename(text: str, mime_prefixes, exts: frozenset, reject_spaces: bool = False) -> bool:
    candidate = str(text or "").strip()
    if not candidate or "\n" in candidate or candidate.endswith("/"):
        return False
    if reject_spaces and any(ch.isspace() for ch in candidate):
        return False
    if Path(candidate).name != candidate:
        return False
    suffix = Path(candidate).suffix.lower()
    if not suffix:
        return False
    guessed_type, _ = mimetypes.guess_type(candidate)
    return bool(guessed_type and guessed_type.startswith(mime_prefixes)) or suffix in exts


def _looks_like_matrix_media_filename(text: str) -> bool:
    return _looks_like_transport_filename(text, ("audio/", "video/"), _MATRIX_MEDIA_FILENAME_EXTS, True)


def _is_bare_media_filename(msgtype: str, body: str) -> bool:
    if msgtype == "m.image":
        return _looks_like_matrix_image_filename(body)
    return msgtype in ("m.audio", "m.file", "m.video") and _looks_like_matrix_media_filename(body)


