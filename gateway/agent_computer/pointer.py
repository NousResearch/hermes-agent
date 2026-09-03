"""Screenshot ↔ viewport coordinate mapping for Human Takeover.

``pointer_click(x, y)`` is accepted in the last-observed screenshot pixel
space. Chromium ``Input.dispatchMouseEvent`` uses CSS viewport pixels.
When screenshot and viewport sizes match (the designed headless config:
deviceScaleFactor=1), the mapping is identity.
"""

from __future__ import annotations


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) from a JPEG SOF marker. (0, 0) if unknown."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return 0, 0
    i = 2
    while i + 8 < len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        if seglen < 2:
            break
        i += 2 + seglen
    return 0, 0


def map_screenshot_to_viewport(
    x: float,
    y: float,
    *,
    screenshot_width: int,
    screenshot_height: int,
    viewport_width: int,
    viewport_height: int,
) -> tuple[float, float]:
    """Map a screenshot pixel to CSS viewport coordinates.

    Identity when sizes match or either side is missing. Otherwise a
    deterministic uniform scale: ``viewport = screenshot * (vp / shot)``.
    """
    if (
        screenshot_width <= 0
        or screenshot_height <= 0
        or viewport_width <= 0
        or viewport_height <= 0
        or (screenshot_width == viewport_width and screenshot_height == viewport_height)
    ):
        return float(x), float(y)
    return (
        float(x) * viewport_width / screenshot_width,
        float(y) * viewport_height / screenshot_height,
    )


def mapping_kind(
    screenshot_width: int,
    screenshot_height: int,
    viewport_width: int,
    viewport_height: int,
) -> str:
    if screenshot_width <= 0 or viewport_width <= 0:
        return "unknown"
    if screenshot_width == viewport_width and screenshot_height == viewport_height:
        return "1:1"
    return "scale"