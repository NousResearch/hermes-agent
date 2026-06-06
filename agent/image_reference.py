"""Safe validation helpers for image reference inputs.

Providers can accept local files, data URLs, or remote URLs in different ways.
This module performs provider-independent validation before any provider upload
so callers do not accidentally read arbitrary local files or accept malformed
image data.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

_ALLOWED_DATA_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_DATA_URL_RE = re.compile(
    r"^data:\s*(image/(?:png|jpeg|webp|gif))\s*;\s*base64\s*,(.*)$",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_MAX_IMAGE_REFERENCE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ImageReference:
    """A validated image reference ready for provider-specific handling."""

    kind: str
    value: str
    mime_type: Optional[str] = None
    path: Optional[Path] = None
    bytes_size: Optional[int] = None


class ImageReferenceError(ValueError):
    """Typed validation error for image references."""

    def __init__(self, message: str, *, error_type: str = "invalid_argument") -> None:
        super().__init__(message)
        self.error_type = error_type


def validate_image_reference(
    reference: str,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_REFERENCE_BYTES,
) -> ImageReference:
    """Validate and normalize a single image reference.

    Accepted references:
    - ``http://`` or ``https://`` URLs (not fetched here)
    - ``data:image/(png|jpeg|webp|gif);base64,...`` with valid image bytes
    - local files below ``$HERMES_HOME/cache/images`` or
      ``$HERMES_HOME/image_cache`` after symlink resolution

    Invalid references raise :class:`ImageReferenceError` with ``error_type`` of
    ``not_found`` or ``invalid_argument``.
    """

    if not isinstance(reference, str) or not reference.strip():
        raise ImageReferenceError("Image reference must be a non-empty string")

    ref = reference.strip()
    parsed = urlparse(ref)
    scheme = parsed.scheme.lower()

    if scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ImageReferenceError("Image URL must include a host")
        return ImageReference(kind="url", value=ref)

    if scheme == "data" or ref.lower().startswith("data:"):
        return _validate_data_url(ref, max_bytes=max_bytes)

    if scheme:
        raise ImageReferenceError(f"Unsupported image reference scheme: {scheme}")

    return _validate_local_path(ref, max_bytes=max_bytes)


def _validate_data_url(ref: str, *, max_bytes: int) -> ImageReference:
    match = _DATA_URL_RE.match(ref)
    if not match:
        raise ImageReferenceError(
            "Data URL must be data:image/(png|jpeg|webp|gif);base64,..."
        )

    mime_type = match.group(1).lower()
    b64_compact = re.sub(r"\s+", "", match.group(2))
    if not b64_compact:
        raise ImageReferenceError("Data URL does not contain image bytes")

    max_encoded_chars = 4 * math.ceil(max_bytes / 3)
    if len(b64_compact) > max_encoded_chars:
        raise ImageReferenceError(
            f"Image reference is too large (encoded length {len(b64_compact)} exceeds {max_encoded_chars} chars)"
        )

    try:
        raw = base64.b64decode(b64_compact, validate=True)
    except binascii.Error as exc:
        raise ImageReferenceError("Data URL contains invalid base64") from exc

    _ensure_size(raw, max_bytes=max_bytes)
    detected = _detect_mime(raw)
    if detected is None:
        raise ImageReferenceError("Data URL does not contain a supported image")
    if detected != mime_type:
        raise ImageReferenceError(
            f"Data URL MIME type {mime_type} does not match image bytes ({detected})"
        )

    return ImageReference(
        kind="data",
        value=f"data:{mime_type};base64,{b64_compact}",
        mime_type=mime_type,
        bytes_size=len(raw),
    )


def _validate_local_path(ref: str, *, max_bytes: int) -> ImageReference:
    path = Path(ref).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ImageReferenceError("Image file was not found", error_type="not_found") from exc
    except OSError as exc:
        raise ImageReferenceError(f"Could not resolve image file: {exc}") from exc

    allowed_roots = _allowed_local_roots()
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ImageReferenceError(
            "Local image references must be under $HERMES_HOME/cache/images or $HERMES_HOME/image_cache"
        )

    if not resolved.is_file():
        raise ImageReferenceError("Image reference is not a file")

    with resolved.open("rb") as fh:
        size = fh.seek(0, 2)
        if size > max_bytes:
            raise ImageReferenceError(
                f"Image reference is too large ({size} bytes > {max_bytes} bytes)"
            )
        fh.seek(0)
        sample = fh.read(16)
    mime_type = _detect_mime(sample)
    if mime_type is None:
        raise ImageReferenceError("Local file is not a supported image")

    return ImageReference(
        kind="file",
        value=str(resolved),
        mime_type=mime_type,
        path=resolved,
        bytes_size=size,
    )


def _allowed_local_roots() -> list[Path]:
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    except Exception:
        import os

        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    roots: list[Path] = []
    for child in ("cache/images", "image_cache"):
        root = home / child
        try:
            roots.append(root.resolve(strict=False))
        except OSError:
            roots.append(root.absolute())
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_size(raw: bytes, *, max_bytes: int) -> None:
    if len(raw) > max_bytes:
        raise ImageReferenceError(
            f"Image reference is too large ({len(raw)} bytes > {max_bytes} bytes)"
        )


def _detect_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw[0:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None
