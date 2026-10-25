"""Extract human-usable text from Teams Adaptive/other card attachments.

Inbound card messages (``application/vnd.microsoft.card.*``) arrive with an empty
``activity.text``; the card JSON lives in ``attachments[i].content``. Without this,
the adapter drops the whole message — Teams "share location", Approvals requests,
sign-in/poll/connector cards, and any card-only message from another bot all vanish.

Kept dependency-free (stdlib only) so it is importable and testable without the
microsoft-teams SDK.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

CARD_CONTENT_PREFIX = "application/vnd.microsoft.card"
_POS_RE = re.compile(r"pos\.(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)")


def _as_card(content: Any) -> Optional[dict]:
    """Attachment content → parsed card dict, or None when it is not a JSON card."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        body = content.strip()
        # Teams sometimes HTML-escapes the JSON inside the attachment payload.
        if body.startswith("&lt;"):
            body = body.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
        if not body.startswith("{"):
            return None
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _walk_text(node: Any, out: list[str]) -> None:
    """Collect human-visible text leaves in document order, de-duplicated by caller."""
    if isinstance(node, dict):
        for key in ("text", "title", "altText"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
        for value in node.values():
            if isinstance(value, (dict, list)):
                _walk_text(value, out)
    elif isinstance(node, list):
        for item in node:
            _walk_text(item, out)


def _walk_coords(node: Any, found: list[tuple[str, str]]) -> None:
    """Find ``pos.<lat>_<lon>`` deep links (Teams location cards use Bing Maps)."""
    if isinstance(node, dict):
        for value in node.values():
            _walk_coords(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_coords(item, found)
    elif isinstance(node, str):
        for match in _POS_RE.finditer(node):
            found.append((match.group(1), match.group(2)))


def attachments_to_text(attachments: Iterable[Any] | None) -> str:
    """Return message text for card attachments, or "" when there is nothing to say.

    Each attachment may be an object with ``content_type``/``content`` attributes
    (Bot Framework models) or a plain dict with those keys (tests, raw activities).
    """
    if not attachments:
        return ""

    seen: set[str] = set()
    parts: list[str] = []
    coords: list[tuple[str, str]] = []

    for att in attachments:
        if isinstance(att, dict):
            content_type = att.get("content_type") or att.get("contentType") or ""
            content = att.get("content")
        else:
            content_type = getattr(att, "content_type", None) or ""
            content = getattr(att, "content", None)

        if not str(content_type).lower().startswith(CARD_CONTENT_PREFIX):
            continue

        card = _as_card(content)
        if card is None:
            continue

        leaves: list[str] = []
        _walk_text(card, leaves)
        _walk_coords(card, coords)

        for leaf in leaves:
            if leaf not in seen:
                seen.add(leaf)
                parts.append(leaf)

    if coords:
        lat, lon = coords[0]
        location = f"location: {lat}, {lon}"
        if location not in seen:
            parts.append(location)

    if not parts:
        return ""
    return "[card] " + " | ".join(parts)
