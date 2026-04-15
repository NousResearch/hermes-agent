from __future__ import annotations

import re
from typing import Optional

LIVE_GATEWAY_SILENT_MARKERS = frozenset(
    {
        "[silent]",
        "silent",
        "no message",
        "no reply",
        "no response",
        "no response generated",
        "empty",
    }
)


def _unwrap_live_gateway_response_text(text: str) -> str:
    normalized = text
    for _ in range(6):
        updated = normalized.strip()
        changed = False

        for wrapper in ("**", "__", "~~", "`"):
            if updated.startswith(wrapper) and updated.endswith(wrapper):
                inner = updated[len(wrapper) : -len(wrapper)].strip()
                if inner:
                    normalized = inner
                    changed = True
                    break
        if changed:
            continue

        for left, right in (("(", ")"), ("[", "]"), ("{", "}"), ('"', '"'), ("'", "'")):
            if updated.startswith(left) and updated.endswith(right):
                inner = updated[len(left) : -len(right)].strip()
                if inner:
                    normalized = inner
                    changed = True
                    break

        if not changed:
            normalized = updated
            break

    return normalized


def _canonicalize_live_gateway_response(text: str) -> str:
    normalized = _unwrap_live_gateway_response_text(text)
    return re.sub(r"[\s\-_]+", " ", normalized).strip(" .!?:;").casefold()


def normalize_live_gateway_response(
    response: Optional[str], *, failed: bool = False
) -> str:
    """Suppress placeholder silence markers before live message delivery."""
    if response is None:
        return ""

    text = str(response).strip()
    if not text or failed:
        return text

    if _canonicalize_live_gateway_response(text) in LIVE_GATEWAY_SILENT_MARKERS:
        return ""

    return text
