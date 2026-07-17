"""Fail-closed image validation for the Sol food workflow.

Two phases, both bounded and value-free:

- :func:`predownload_check` runs against platform metadata (file size,
  advertised dimensions, album membership) BEFORE any byte is downloaded.
- :func:`validate_image_bytes` runs against the downloaded bytes and
  performs magic-based format detection plus header-only dimension
  parsing (no full decode — decompression-bomb warnings become
  rejections by construction, because we never decode).

Accepted formats: JPEG, PNG, WebP only. Everything else fails closed with
a stable reason code. No paths, captions, or content are ever logged.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from plugins.sol_food.limits import (
    FOOD_IMAGE_FORMATS,
    FOOD_IMAGE_MAX_BYTES,
    FOOD_IMAGE_MAX_PIXELS,
    FOOD_IMAGE_MAX_SIDE,
)

__all__ = [
    "MediaRejected",
    "predownload_check",
    "validate_image_bytes",
    "probe_image",
]

# Stable value-free reason codes.
REASON_MEDIA_GROUP = "food_media_album_rejected"
REASON_TOO_LARGE = "food_media_too_large"
REASON_BAD_FORMAT = "food_media_bad_format"
REASON_BAD_DIMENSIONS = "food_media_bad_dimensions"
REASON_HEADER_UNPARSEABLE = "food_media_header_unparseable"


class MediaRejected(Exception):
    """Raised with a stable value-free ``reason_code``."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ImageProbe:
    format: str  # "jpeg" | "png" | "webp"
    width: int
    height: int


def _check_dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise MediaRejected(REASON_BAD_DIMENSIONS)
    if width > FOOD_IMAGE_MAX_SIDE or height > FOOD_IMAGE_MAX_SIDE:
        raise MediaRejected(REASON_BAD_DIMENSIONS)
    if width * height > FOOD_IMAGE_MAX_PIXELS:
        raise MediaRejected(REASON_BAD_DIMENSIONS)


def predownload_check(
    *,
    file_size: Optional[int],
    width: Optional[int],
    height: Optional[int],
    media_group_id: Optional[str],
) -> None:
    """Reject before download using platform-advertised metadata.

    Any update carrying a ``media_group_id`` (album / multi-photo burst)
    fails closed here — before download, model execution, or candidate
    creation. Advertised size/dimensions over the ceilings also fail here.
    Metadata the platform did not provide (None) is NOT trusted as "ok";
    it simply defers that specific bound to the post-download validation.
    """
    if media_group_id is not None:
        raise MediaRejected(REASON_MEDIA_GROUP)
    if file_size is not None and file_size > FOOD_IMAGE_MAX_BYTES:
        raise MediaRejected(REASON_TOO_LARGE)
    if width is not None and height is not None:
        _check_dimensions(width, height)


# ── Header-only probes ──────────────────────────────────────────────────

def _probe_png(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    # First chunk must be IHDR: length(4) type(4) width(4) height(4).
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _probe_jpeg(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    # Walk segments looking for a start-of-frame marker.
    offset = 2
    size = len(data)
    _SOF = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while offset + 4 <= size:
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        if marker == 0xD8:  # stray SOI
            offset += 2
            continue
        if 0xD0 <= marker <= 0xD9:  # RST/EOI have no length
            offset += 2
            continue
        if offset + 4 > size:
            return None
        seg_len = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        if seg_len < 2:
            return None
        if marker in _SOF:
            if offset + 9 > size:
                return None
            height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
            return width, height
        offset += 2 + seg_len
    return None


def _probe_webp(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        # 10-byte payload starting at 20: flags(4), canvas W-1 (24-bit LE),
        # canvas H-1 (24-bit LE).
        if len(data) < 30:
            return None
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 ":
        # Lossy: payload at 20; 3-byte frame tag, then start code
        # 0x9d 0x01 0x2a, then 14-bit width and height (16-bit LE fields).
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            return None
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        # Lossless: payload at 20; signature byte 0x2f, then 14-bit
        # width-1 and 14-bit height-1 packed little-endian.
        if len(data) < 25 or data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None


def probe_image(data: bytes) -> Optional[ImageProbe]:
    """Magic-detect format and parse dimensions from headers only."""
    dims = _probe_png(data)
    if dims is not None:
        return ImageProbe("png", dims[0], dims[1])
    dims = _probe_jpeg(data)
    if dims is not None:
        return ImageProbe("jpeg", dims[0], dims[1])
    dims = _probe_webp(data)
    if dims is not None:
        return ImageProbe("webp", dims[0], dims[1])
    return None


def validate_image_bytes(data: bytes) -> ImageProbe:
    """Validate downloaded bytes: size, magic/format, header dimensions.

    Order matters: the byte ceiling is enforced before any parsing, and
    dimensions are checked before anything could decode. Raises
    :class:`MediaRejected` with a stable reason code on any failure.
    """
    if len(data) > FOOD_IMAGE_MAX_BYTES:
        raise MediaRejected(REASON_TOO_LARGE)
    if len(data) == 0:
        raise MediaRejected(REASON_BAD_FORMAT)
    data = bytes(data)
    probe = probe_image(data)
    if probe is None:
        # Distinguish "recognized container, unparseable header" from
        # "not an accepted format" only via stable codes.
        head = bytes(data[:12])
        looks_known = (
            head[:2] == b"\xff\xd8"
            or head[:8] == b"\x89PNG\r\n\x1a\n"
            or (head[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP")
        )
        raise MediaRejected(
            REASON_HEADER_UNPARSEABLE if looks_known else REASON_BAD_FORMAT
        )
    if probe.format not in FOOD_IMAGE_FORMATS:
        raise MediaRejected(REASON_BAD_FORMAT)
    _check_dimensions(probe.width, probe.height)
    return probe
