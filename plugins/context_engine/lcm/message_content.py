"""Message content normalization helpers.

Hermes/OpenAI-format messages may carry ``content`` as plain text or as
structured content parts (for example text + image blocks). LCM persists and
accounts for message content as text, so all write/matching/token paths should
use deliberate normalization.
"""

from __future__ import annotations

import json
from typing import Any

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def _extract_text_part_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, str):
            return nested
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    return None


def normalize_content_value(content: Any) -> str | None:
    """Return a stable text representation for message content.

    ``None`` remains ``None`` so callers that distinguish SQL NULL from an empty
    string can preserve that behavior. Strings are returned unchanged. Structured
    content is serialized deterministically so storage, source-id matching, and
    token accounting all see the same value.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def text_content_for_pattern_matching(content: Any) -> str | None:
    """Return the operator-visible text string used by message filters.

    Structured multimodal payloads often arrive as lists of content parts. For
    ignore-pattern matching, prefer concatenated text parts so anchored patterns
    bind to the text an operator sees. If no text parts are present, fall back to
    the stable normalized representation used for storage.
    """
    if content is None or isinstance(content, str):
        return normalize_content_value(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in _TEXT_PART_TYPES:
                    text = _extract_text_part_value(part.get("text"))
                    if text is None:
                        text = _extract_text_part_value(part.get("content"))
                    if text:
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    return normalize_content_value(content)


def stored_text_content_for_pattern_matching(content: Any) -> str | None:
    """Return message-filter text for content read back from storage.

    Structured content is persisted as canonical JSON. Decode that legacy stored
    representation when it round-trips to the same normalized string so restart
    reconciliation applies the same text-first ignore policy to durable rows as
    it applies to live structured messages.
    """
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return text_content_for_pattern_matching(content)
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return text_content_for_pattern_matching(decoded)
    return text_content_for_pattern_matching(content)
