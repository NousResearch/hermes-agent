"""Helpers for validating and sanitizing image data URLs.

Native multimodal routing stores local images as ``data:image/*;base64,...``
URLs.  Providers validate those payloads strictly; a malformed or truncated
image in persisted history can make every subsequent turn fail with a
non-retryable HTTP 400.  Keep this module small and side-effect free so provider
adapters can sanitize per-call copies without mutating persisted sessions.
"""

from __future__ import annotations

import base64
import re
from typing import Optional, Tuple


INVALID_IMAGE_ATTACHMENT_NOTE = "[Invalid image attachment removed before API call]"

_DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+)(?:;[^,]*)?;base64,", re.IGNORECASE)


def detect_image_mime(data: bytes) -> Optional[str]:
    """Return the MIME type implied by known image magic bytes, if any."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 2:
        return None
    raw = bytes(data)
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return None


def is_supported_image_bytes(data: bytes) -> bool:
    """True when *data* starts with a supported image magic-byte sequence."""
    return detect_image_mime(data) is not None


def _decode_base64_payload(payload: str) -> Optional[bytes]:
    compact = "".join(str(payload or "").split())
    if not compact:
        return None
    # Data URLs in the wild sometimes omit padding.  Accept that while keeping
    # ``validate=True`` so non-base64 characters do not slip through.
    compact += "=" * (-len(compact) % 4)
    try:
        return base64.b64decode(compact, validate=True)
    except Exception:
        return None


def parse_image_data_url(url: str) -> Optional[Tuple[str, bytes, str]]:
    """Parse a base64 image data URL into ``(declared_mime, bytes, payload)``.

    Returns ``None`` when the URL is not a syntactically valid image data URL.
    """
    if not isinstance(url, str):
        return None
    match = _DATA_IMAGE_RE.match(url.strip())
    if not match:
        return None
    declared_mime = match.group(1).lower()
    payload = url.strip()[match.end():]
    decoded = _decode_base64_payload(payload)
    if decoded is None:
        return None
    return declared_mime, decoded, payload


def sanitize_image_data_url(url: str) -> Tuple[Optional[str], bool]:
    """Return ``(sanitized_url, removed)`` for an image URL value.

    - Non-data URLs are returned unchanged.
    - Valid image data URLs are returned unchanged unless the declared MIME does
      not match the bytes, in which case only the MIME prefix is normalized.
    - Malformed/unsupported image data URLs return ``(None, True)`` so callers
      can replace them with a textual note instead of sending bad payloads to a
      provider.
    """
    if not isinstance(url, str) or not url:
        return url if isinstance(url, str) else None, False

    stripped = url.strip()
    if not stripped.lower().startswith("data:image/"):
        return url, False

    parsed = parse_image_data_url(stripped)
    if parsed is None:
        return None, True

    declared_mime, decoded, payload = parsed
    actual_mime = detect_image_mime(decoded)
    if actual_mime is None:
        return None, True

    if declared_mime != actual_mime:
        return f"data:{actual_mime};base64,{payload}", False
    return stripped, False


__all__ = [
    "INVALID_IMAGE_ATTACHMENT_NOTE",
    "detect_image_mime",
    "is_supported_image_bytes",
    "parse_image_data_url",
    "sanitize_image_data_url",
]
