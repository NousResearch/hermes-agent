"""Shared mime↔extension dispatch for inbound (downloaded) platform media.

Adapters historically hand-rolled divergent mime→extension maps on purpose (BlueBubbles
coerces ``image/heic`` to ``.jpg`` for vision tools; WhatsApp Cloud pins ``audio/ogg`` to
``.ogg`` for the STT extension whitelist). This module owns the agreed-upon union tables
plus lookup helpers taking per-adapter ``overrides`` (and ``use_defaults``/``use_mimetypes``
toggles) so each adapter's historical output stays byte-identical —
``tests/gateway/test_media_cache.py`` pins those outputs as the contract.
"""

from __future__ import annotations

import mimetypes
import uuid
from typing import Mapping, Optional

# Union of the per-adapter maps where they agree. Favors the common-in-the-wild extension
# over the RFC-correct one (``audio/ogg`` → ``.ogg``, not ``.oga``): downstream STT/vision
# pipelines whitelist real-world extensions. (weixin.py's private ``_mime_from_filename``
# is not folded in yet — another in-flight branch edits that file.)
DEFAULT_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
    "audio/ogg": ".ogg", "audio/x-opus+ogg": ".ogg", "audio/opus": ".ogg",  # whatsapp voice notes
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav",
    "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    "video/mp4": ".mp4", "application/pdf": ".pdf", "application/zip": ".zip",
}

# Explicit inverse (the forward table is many-to-one, so the inverse must pick
# the canonical mime). Byte-identical to Signal's historical ``_EXT_TO_MIME``.
DEFAULT_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac", ".mp4": "video/mp4", ".pdf": "application/pdf",
    ".zip": "application/zip",
}


def _normalize_mime(mime: str) -> str:
    """Lowercase and strip any ``; charset=...`` style parameters."""
    return (mime or "").split(";")[0].strip().lower()


def ext_for_mime(mime: str, *, overrides: Optional[Mapping[str, str]] = None,
                 use_defaults: bool = True, use_mimetypes: bool = True,
                 fallback: Optional[str] = None) -> Optional[str]:
    """Resolve a mime type to a dotted extension: ``overrides`` → ``DEFAULT_MIME_TO_EXT`` (if
    ``use_defaults``) → ``mimetypes.guess_extension`` (if ``use_mimetypes``) → ``fallback``."""
    primary = _normalize_mime(mime)
    if not primary:
        return fallback
    stages = [overrides.get if overrides else None,
              DEFAULT_MIME_TO_EXT.get if use_defaults else None,
              mimetypes.guess_extension if use_mimetypes else None]
    for lookup in stages:
        ext = lookup(primary) if lookup else None
        if ext:
            return ext
    return fallback


def mime_for_ext(ext: str, *, overrides: Optional[Mapping[str, str]] = None,
                 fallback: str = "application/octet-stream") -> str:
    """Inverse lookup: ``overrides`` → ``DEFAULT_EXT_TO_MIME`` → ``fallback``."""
    key = (ext or "").strip().lower()
    return (overrides or {}).get(key) or DEFAULT_EXT_TO_MIME.get(key, fallback)


def cache_media_bytes(data: bytes, mime: str, *, filename_hint: str = "",
                      kind_hint: Optional[str] = None,
                      ext_overrides: Optional[Mapping[str, str]] = None) -> str:
    """Cache downloaded media bytes and return the local file path.

    Picks the image / audio / document cache primitive by mime class (or explicit ``kind_hint``
    ``"image"``/``"audio"``/``"document"``). ``filename_hint`` names document files (else a
    generated name with the resolved extension); ``ext_overrides`` feeds :func:`ext_for_mime`.
    """
    # Local import: base is heavyweight and some adapters import this module very early.
    from gateway.platforms.base import (
        cache_audio_from_bytes, cache_document_from_bytes, cache_image_from_bytes)
    primary = _normalize_mime(mime)
    kind = kind_hint
    if kind is None:
        kind = ("image" if primary.startswith("image/")
                else "audio" if primary.startswith("audio/") else "document")
    if kind == "image":
        ext = ext_for_mime(primary, overrides=ext_overrides, fallback=".jpg") or ".jpg"
        return cache_image_from_bytes(data, ext)
    if kind == "audio":
        ext = ext_for_mime(primary, overrides=ext_overrides, fallback=".ogg") or ".ogg"
        return cache_audio_from_bytes(data, ext)
    filename = filename_hint
    if not filename:
        ext = ext_for_mime(primary, overrides=ext_overrides, fallback=".bin")
        filename = f"file_{uuid.uuid4().hex[:8]}{ext}"
    return cache_document_from_bytes(data, filename)
