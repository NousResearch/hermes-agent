"""Plain-text rendering for Feishu message payloads.

Sibling of ``adapter.py``: it holds the message-type renderers that turn a raw
``body.content`` payload into text for the agent, so the adapter facade does not
grow for every new message type (``FILE_LINES`` ratchet in
``scripts/code_health``).  Nothing here imports ``adapter`` — the adapter passes
payloads in as plain dicts, which also keeps the module safe to load under the
plugin loader's synthetic namespace.
"""

from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import quote

FALLBACK_LOCATION_TEXT = "[Location]"

_WHITESPACE_RE = re.compile(r"\s+")


def location_text_content(payload: dict[str, Any]) -> str:
    """Render a ``location`` payload (``{"name", "longitude", "latitude"}``) as text.

    The result is never empty: an empty ``text_content`` makes the adapter drop
    the whole message as "no content", which is how a shared location used to
    vanish silently.  Coordinates keep the payload's original string form so
    precision survives, and the AMap marker link lets the agent open or route on
    the spot.  A missing/invalid name or coordinate degrades to what is left
    instead of failing.
    """
    name = _WHITESPACE_RE.sub(" ", str(payload.get("name") or "")).strip()
    longitude = _location_coordinate(payload, "longitude", "lng")
    latitude = _location_coordinate(payload, "latitude", "lat")
    lines = [f"Location: {name}"] if name else []
    if longitude and latitude:
        lines.append(f"Coordinates: {longitude},{latitude}")
        marker_url = f"https://uri.amap.com/marker?position={longitude},{latitude}"
        if name:
            marker_url += f"&name={quote(name, safe='')}"
        lines.append(f"Map: {marker_url}")
    return "\n".join(lines) or FALLBACK_LOCATION_TEXT


def _location_coordinate(payload: dict[str, Any], *keys: str) -> str:
    """First finite numeric coordinate among ``keys``, as its original string; else ``""``."""
    for key in keys:
        raw = payload.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        text = str(raw).strip()
        try:
            finite = math.isfinite(float(text))
        except ValueError:
            continue
        if finite:
            return text
    return ""
