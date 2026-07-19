from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Canonical structured-content text projection (single owner).  Everything
# that needs the *visible text* of a message — reasoning scrubbers, interim
# dedup, assistant-message building, MoA aggregation — must go through this
# module instead of maintaining its own coercion logic.
#
# Contract:
#   known visible text      -> extracted
#   known non-text content  -> ignored
#   unknown structure       -> "" (never stringified, never reflectively read)
#
# Only these part shapes contribute text:
#   * plain ``str`` items
#   * typed text parts whose ``type`` is in ``_TEXT_PART_TYPES`` (typed
#     Mappings or provider SDK objects exposing ``.type``/``.text``)
#   * explicitly allowed summary parts (``_SUMMARY_PART_TYPES``)
#   * untyped PURE legacy wrapper Mappings — every key is in
#     ``_LEGACY_WRAPPER_KEYS`` (``{"text": str}`` / ``{"content": str}``,
#     in any position). An untyped Mapping carrying ANY extra
#     provider/tool/metadata/unknown field is not a wrapper and yields "".
# Images, URLs, base64, audio, files, tool results, encrypted/redacted
# reasoning, provider metadata, unknown typed parts and unknown objects
# contribute nothing.

_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_SUMMARY_PART_TYPES = frozenset({"summary_text"})
_LEGACY_WRAPPER_KEYS = ("text", "content")


def _part_type(part: Any) -> Any:
    if isinstance(part, Mapping):
        try:
            return part.get("type")
        except Exception:
            return None
    try:
        return getattr(part, "type", None)
    except Exception:
        return None


def _part_text(part: Any) -> Any:
    if isinstance(part, Mapping):
        try:
            return part.get("text")
        except Exception:
            return None
    try:
        return getattr(part, "text", None)
    except Exception:
        return None


def _text_from_part(part: Any) -> str:
    """Extract visible text from one explicitly textual content part.

    Returns "" for anything not in the allowlist — no ``str()`` fallback and
    no reflective attribute reads on unknown objects. Hostile Mappings whose
    accessors raise also yield "" (never propagated).
    """
    if part is None:
        return ""
    if isinstance(part, str):
        return part

    part_type = _part_type(part)
    if isinstance(part_type, str):
        normalized = part_type.strip().lower()
        if normalized in _TEXT_PART_TYPES or normalized in _SUMMARY_PART_TYPES:
            text = _part_text(part)
            if isinstance(text, str):
                return text
        return ""

    if part_type is None and isinstance(part, Mapping):
        # Untyped Mappings count as legacy text wrappers ONLY when they are
        # PURE wrappers: every key is one of the wrapper keys. Any extra
        # provider/tool/metadata/unknown field disqualifies the shape —
        # otherwise provider metadata or tool-call payloads would leak into
        # visible text.
        try:
            if not set(part.keys()) <= set(_LEGACY_WRAPPER_KEYS):
                return ""
            for key in _LEGACY_WRAPPER_KEYS:
                value = part.get(key)
                if isinstance(value, str):
                    return value
        except Exception:
            return ""
    return ""


def flatten_message_text(content: Any, *, sep: str = "\n") -> str:
    """Return the visible text from common chat/Responses message content shapes.

    ``sep`` joins multiple visible parts (default "\n" for display-oriented
    callers).  Pipelines that must reassemble provider-split tokens or tags
    exactly (reasoning scrubbers, regex sinks) pass ``sep=""`` so no
    character that was absent from the original content is inserted.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [_text_from_part(part) for part in content]
        return sep.join(chunk for chunk in chunks if chunk)

    # Top-level single part / legacy wrapper position.
    return _text_from_part(content)
