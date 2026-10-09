"""Shared mime↔extension dispatch for inbound (downloaded) platform media.

Historically every gateway adapter hand-rolled its own mime→extension map
before handing downloaded bytes to the cache primitives in
``gateway.platforms.base`` (``cache_image_from_bytes``,
``cache_audio_from_bytes``, ``cache_document_from_bytes``).  Those maps
*disagree* with each other on purpose — e.g. BlueBubbles coerces
``image/heic`` to ``.jpg`` because downstream vision tools can't read HEIC,
while WhatsApp Cloud pins ``audio/ogg`` to ``.ogg`` (not the RFC-correct
``.oga`` Python's ``mimetypes`` returns) because the STT pipeline whitelists
extensions.

This module owns:

* ``DEFAULT_MIME_TO_EXT`` — the union table of entries the adapters already
  agree on (plus a few uncontroversial document types).
* ``DEFAULT_EXT_TO_MIME`` — the canonical inverse (used by Signal to map a
  sniffed extension back to a content type).
* ``ext_for_mime`` / ``mime_for_ext`` — lookup helpers that accept
  per-adapter ``overrides`` so each adapter's historical (divergent)
  behavior is preserved byte-for-byte.
* ``cache_media_bytes`` — one-call dispatch: classify the mime, resolve the
  extension, and write to the right cache (image / audio / document).

Behavior-preservation contract: adapters that had divergent maps pass them
as ``overrides`` (and, where their historical code never consulted
``mimetypes`` or a shared table, disable those fallbacks via
``use_defaults`` / ``use_mimetypes``).  The parity tests in
``tests/gateway/test_media_cache.py`` hardcode the historical outputs as
the contract.

NOTE: ``gateway/platforms/weixin.py`` also has a private mime map
(``_mime_from_filename``) but is intentionally NOT migrated here — another
in-flight branch edits that file.  Follow-up: fold it in once that lands.
"""

from __future__ import annotations

import mimetypes
import uuid
from typing import Mapping, Optional

# ---------------------------------------------------------------------------
# Shared tables
# ---------------------------------------------------------------------------

# Union of the per-adapter maps where the adapters already agree (or where
# only one adapter pinned the type and no other adapter contradicts it).
# Entries deliberately favor the common-in-the-wild extension over the
# RFC-correct one (``audio/ogg`` → ``.ogg``, not ``.oga``) because the
# downstream STT/vision pipelines whitelist real-world extensions.
DEFAULT_MIME_TO_EXT: dict[str, str] = {
    # --- images (bluebubbles + whatsapp_cloud agree; matches mimetypes) ---
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    # --- audio ---
    "audio/ogg": ".ogg",          # bluebubbles + whatsapp_cloud agree
    "audio/x-opus+ogg": ".ogg",   # whatsapp voice notes (opus-in-ogg)
    "audio/opus": ".ogg",         # whatsapp voice notes (opus-in-ogg)
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",          # non-standard but seen in the wild
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",          # bluebubbles + whatsapp_cloud agree
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    # --- video / documents (from signal's inverse table) ---
    "video/mp4": ".mp4",
    "application/pdf": ".pdf",
    "application/zip": ".zip",
}

# Canonical inverse.  Kept explicit (rather than mechanically inverted)
# because the forward table is many-to-one — e.g. both ``audio/mpeg`` and
# ``audio/mp3`` map to ``.mp3`` and the inverse must pick the canonical
# mime.  This is byte-identical to Signal's historical ``_EXT_TO_MIME``.
DEFAULT_EXT_TO_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
    ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".pdf": "application/pdf",
    ".zip": "application/zip",
}


def _normalize_mime(mime: str) -> str:
    """Lowercase and strip any ``; charset=...`` style parameters."""
    return (mime or "").split(";")[0].strip().lower()


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def ext_for_mime(
    mime: str,
    *,
    overrides: Optional[Mapping[str, str]] = None,
    use_defaults: bool = True,
    use_mimetypes: bool = True,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Resolve a mime type to a file extension (including the dot).

    Resolution order: ``overrides`` → ``DEFAULT_MIME_TO_EXT`` (if
    ``use_defaults``) → ``mimetypes.guess_extension`` (if
    ``use_mimetypes``) → ``fallback``.

    Adapters with historical divergent maps pass them via ``overrides``
    and disable the stages their old code never consulted, keeping their
    outputs byte-identical to the pre-refactor behavior.
    """
    primary = _normalize_mime(mime)
    if not primary:
        return fallback
    if overrides:
        ext = overrides.get(primary)
        if ext:
            return ext
    if use_defaults:
        ext = DEFAULT_MIME_TO_EXT.get(primary)
        if ext:
            return ext
    if use_mimetypes:
        ext = mimetypes.guess_extension(primary)
        if ext:
            return ext
    return fallback


def mime_for_ext(
    ext: str,
    *,
    overrides: Optional[Mapping[str, str]] = None,
    fallback: str = "application/octet-stream",
) -> str:
    """Inverse lookup: file extension → canonical mime type.

    Resolution order: ``overrides`` → ``DEFAULT_EXT_TO_MIME`` → ``fallback``.
    """
    key = (ext or "").strip().lower()
    if overrides:
        mime = overrides.get(key)
        if mime:
            return mime
    return DEFAULT_EXT_TO_MIME.get(key, fallback)


# ---------------------------------------------------------------------------
# One-call cache dispatch
# ---------------------------------------------------------------------------

def cache_media_bytes(
    data: bytes,
    mime: str,
    *,
    filename_hint: str = "",
    kind_hint: Optional[str] = None,
    ext_overrides: Optional[Mapping[str, str]] = None,
) -> str:
    """Cache downloaded media bytes and return the local file path.

    Picks the image / audio / document cache primitive from
    ``gateway.platforms.base`` based on the mime class (or an explicit
    ``kind_hint`` of ``"image"``, ``"audio"`` or ``"document"``).
    ``filename_hint`` is used for document caching (falls back to a
    generated name with the resolved extension).  ``ext_overrides`` is
    threaded through to :func:`ext_for_mime` for adapters that need their
    historical mappings.
    """
    # Local import: base is a large module and some adapters import this
    # module very early; keep import-time coupling minimal.
    from gateway.platforms.base import (
        cache_audio_from_bytes,
        cache_document_from_bytes,
        cache_image_from_bytes,
    )

    primary = _normalize_mime(mime)
    kind = kind_hint
    if kind is None:
        if primary.startswith("image/"):
            kind = "image"
        elif primary.startswith("audio/"):
            kind = "audio"
        else:
            kind = "document"

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
