"""Shared, security-checked resolution of user-supplied image sources.

Every image-gen backend accepts a source image for editing in one of three
shapes: a public ``http(s)`` URL, a ``data:`` URI, or a local file path (what a
user-attached image looks like once the gateway has saved it to disk and
surfaced it to the model). The shapes are equivalent to the model, but each
backend's API ingests them differently — FAL uploads or inlines, OpenAI sends
file objects, xAI and Krea pass a URL/``data:`` field, the Codex Responses tool
takes an ``input_image`` part.

What every backend *does* share is the validation a source must pass before it
is trusted:

* the read denylist (:func:`agent.file_safety.get_read_block_error`) — a
  credential store or secret-bearing file must not be shipped to an image API
  just because its bytes happen to look like an image;
* magic-byte sniffing (:func:`agent.image_routing._sniff_mime_from_bytes`) —
  the content type comes from the file's leading bytes, not its extension or a
  ``data:`` label, so a mislabelled or non-image source is rejected before it
  leaves the machine.

:func:`resolve_image_source` performs that validation once and returns an
:class:`ImageSource`; each backend then packages it for its own API via the
methods on that class. Remote URLs are trusted and passed through unchanged —
fetching them is the backend's concern, not this module's.
"""

from __future__ import annotations

import base64
import enum
import os
from dataclasses import dataclass
from typing import Optional

# Enough leading bytes to identify any format ``_sniff_mime_from_bytes`` knows
# (the longest check inspects offset 12).
SNIFF_BYTES = 64

ACCEPTED_IMAGE_TYPES = "PNG, JPEG, GIF, WEBP, BMP, HEIC"

_MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/heic": "heic",
}


class SourceKind(enum.Enum):
    """Which of the three source shapes a reference resolved to."""

    REMOTE = "remote"
    DATA_URI = "data_uri"
    LOCAL = "local"


def _require_image_mime(head: bytes, ref: str) -> str:
    """Return the sniffed image MIME for *head*, or raise if it isn't an image."""
    from agent.image_routing import _sniff_mime_from_bytes

    mime = _sniff_mime_from_bytes(head)
    if mime is None:
        raise ValueError(
            f"Source image is not a recognised image file: {ref}. "
            f"image sources must be one of {ACCEPTED_IMAGE_TYPES}."
        )
    return mime


def _validate_data_uri(value: str, ref: str) -> str:
    """Validate a ``data:`` URI declares an image and carries image bytes.

    The declared MIME is checked, and the base64 payload's leading bytes are
    decoded and sniffed — a label alone isn't trusted, mirroring the magic-byte
    check applied to local files. Only base64 image data URIs are accepted.
    Returns the sniffed MIME.
    """
    header, _, payload = value.partition(",")
    if not header.lower().startswith("data:image/"):
        raise ValueError(
            f"Source data URI is not an image: {ref}. image sources must be "
            f"one of {ACCEPTED_IMAGE_TYPES}."
        )
    if ";base64" not in header.lower():
        raise ValueError(f"Image data URIs must be base64-encoded: {ref}")

    # RFC 2397 permits whitespace (some encoders wrap at 76 cols), so drop
    # leading whitespace and strip it from a bounded slice before sniffing.
    compact = "".join(payload.lstrip()[: SNIFF_BYTES * 2].split())
    prefix = compact[:SNIFF_BYTES]
    prefix = prefix[: len(prefix) - (len(prefix) % 4)]
    try:
        head = base64.b64decode(prefix)
    except Exception as exc:  # noqa: BLE001 — malformed payload, reject clearly
        raise ValueError(f"Source data URI has malformed base64: {ref}") from exc
    return _require_image_mime(head, ref)


def _local_path_from_ref(value: str) -> str:
    """Map a non-URL reference to a filesystem path, accepting ``file://``.

    Rejects a remote ``file://`` host (``file://server/share/x``) rather than
    silently dropping it and reading the local ``/share/x``.
    """
    if not value.lower().startswith("file://"):
        return os.path.expanduser(value)

    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(value)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        raise ValueError(f"Unsupported remote file:// host: {value}")
    return urllib.request.url2pathname(parsed.path)


@dataclass(frozen=True)
class ImageSource:
    """A validated image source ready for a backend to package.

    ``value`` carries the URL (``REMOTE``), the ``data:`` URI (``DATA_URI``), or
    the filesystem path (``LOCAL``). ``mime`` is the sniffed content type for
    ``LOCAL`` and ``DATA_URI`` sources and ``None`` for ``REMOTE`` (whose bytes
    this module never fetches). ``reference`` is the caller's original argument,
    kept for error messages.
    """

    kind: SourceKind
    value: str
    reference: str
    mime: Optional[str] = None

    @property
    def path(self) -> str:
        """Filesystem path for a ``LOCAL`` source; raises otherwise."""
        if self.kind is not SourceKind.LOCAL:
            raise ValueError(f"Source is not a local file: {self.reference}")
        return self.value

    def read_bytes(self) -> bytes:
        """Return the raw image bytes.

        Reads the file for ``LOCAL`` and decodes the payload for ``DATA_URI``.
        Raises for ``REMOTE`` — fetching a remote URL is the backend's concern,
        not this module's.
        """
        if self.kind is SourceKind.LOCAL:
            with open(self.value, "rb") as handle:
                return handle.read()
        if self.kind is SourceKind.DATA_URI:
            _, _, payload = self.value.partition(",")
            return base64.b64decode("".join(payload.split()))
        raise ValueError(
            f"Cannot read bytes from a remote URL without fetching it: {self.reference}"
        )

    def as_data_uri(self) -> str:
        """Return the source as a base64 ``data:`` URI.

        Inlines the bytes for ``LOCAL``, returns the URI unchanged for
        ``DATA_URI``, and raises for ``REMOTE`` (which can't be inlined without
        a fetch).
        """
        if self.kind is SourceKind.DATA_URI:
            return self.value
        if self.kind is SourceKind.LOCAL:
            mime = self.mime or "image/png"
            encoded = base64.b64encode(self.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        raise ValueError(
            f"Cannot inline a remote URL as a data URI: {self.reference}"
        )

    def as_url_or_inline(self) -> str:
        """Return a string a URL-or-``data:`` field can carry.

        ``REMOTE`` passes its URL through; ``LOCAL`` and ``DATA_URI`` become a
        ``data:`` URI. This is the form xAI's ``image`` field, Krea's
        ``image_style_references`` entries, and the Codex ``input_image`` part
        all accept.
        """
        if self.kind is SourceKind.REMOTE:
            return self.value
        return self.as_data_uri()

    def filename(self) -> str:
        """A plausible filename, for APIs that want a named file object."""
        if self.kind is SourceKind.LOCAL:
            return os.path.basename(self.value) or "image.png"
        ext = _MIME_EXTENSIONS.get(self.mime or "", "png")
        return f"image.{ext}"


def resolve_image_source(ref: str, *, sniff_bytes: int = SNIFF_BYTES) -> ImageSource:
    """Classify and validate a source-image reference.

    ``http(s)`` URLs are trusted and returned as ``REMOTE`` without a fetch.
    ``data:`` URIs are validated as base64-encoded images. Local files
    (including ``file://``) must exist, pass the read denylist, and sniff as a
    real image. Raises :class:`ValueError` with an actionable message on any
    failure, so callers can surface it rather than handing a backend something
    it will reject.
    """
    value = (ref or "").strip()
    if not value:
        raise ValueError("Empty source image reference")

    # Only the scheme prefix matters here; avoid lowercasing a large data: URI.
    low = value[:8].lower()
    if low.startswith(("http://", "https://")):
        return ImageSource(SourceKind.REMOTE, value, ref)
    if low.startswith("data:"):
        mime = _validate_data_uri(value, ref)
        return ImageSource(SourceKind.DATA_URI, value, ref, mime=mime)

    path = _local_path_from_ref(value)
    if not os.path.isfile(path):
        raise ValueError(
            f"Source image not found: {ref}. Pass a public URL, a data: URI, "
            f"or a readable local image file."
        )

    # Apply the same read denylist the file Read tool uses, so a credential
    # store or secret-bearing file can't be sent to an image API just because
    # its bytes happen to look like an image.
    from agent.file_safety import get_read_block_error

    block_error = get_read_block_error(path)
    if block_error:
        raise ValueError(block_error)

    with open(path, "rb") as handle:
        head = handle.read(sniff_bytes)
    mime = _require_image_mime(head, ref)
    return ImageSource(SourceKind.LOCAL, path, ref, mime=mime)
