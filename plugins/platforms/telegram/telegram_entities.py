"""Inbound message text the adapter facade reads: ``text_link`` expansion and ``rich_message`` blocks.

A rich-text hyperlink arrives as visible anchor text plus a ``text_link`` entity carrying the
URL; the model only ever sees the anchor unless the URL is inlined (#31071). Long multi-paragraph
messages can instead arrive with an empty ``text`` and their content in ``rich_message.blocks``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Real clients nest a handful of levels; content deeper than this is dropped so a hostile payload
# (valid JSON thousands of levels deep) cannot hit the recursion limit and drop the whole update.
_MAX_RICH_DEPTH = 64
_CODE_TYPES = frozenset({"code", "pre"})


def _rich_inline(value: Any, depth: int = 0, skip_code: bool = False) -> str:
    """Plain text of a rich ``text`` value: a string, a list of runs, or a span dict (bold, link…)
    whose content sits in ``text`` or ``children``, falling back to a link's ``url``. ``skip_code``
    blanks ``code``/``pre`` spans."""
    if depth > _MAX_RICH_DEPTH:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_rich_inline(part, depth + 1, skip_code) for part in value)
    if isinstance(value, Mapping):
        if skip_code and value.get("type") in _CODE_TYPES:
            return " "
        for key in ("text", "children"):
            if value.get(key) is not None:
                return _rich_inline(value[key], depth + 1, skip_code)
        return str(value.get("url") or "")
    return ""


def _rich_lines(blocks: Any, prefix: str = "", depth: int = 0, skip_code: bool = False) -> list[str]:
    """One line per paragraph-like block; a list item's ``label`` (bullet/number) prefixes its first line."""
    lines: list[str] = []
    if depth > _MAX_RICH_DEPTH:
        return lines
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, Mapping) or (skip_code and block.get("type") in _CODE_TYPES):
            continue
        if "text" in block:
            lines.append(prefix + _rich_inline(block["text"], skip_code=skip_code))
            prefix = ""
        for item in block.get("items") or []:
            if isinstance(item, Mapping):
                label = str(item.get("label") or "-")
                lines += _rich_lines(item.get("blocks") or [item], f"{label} ", depth + 1, skip_code)
        if block.get("blocks"):
            lines += _rich_lines(block["blocks"], prefix, depth + 1, skip_code)
    return lines


def rich_message_text(message: Any, skip_code: bool = False) -> str:
    """Flattened text of a ``rich_message`` payload ("" when there is none), for inbound messages and
    the replied-to message alike; ``skip_code`` leaves out ``code``/``pre`` spans (mention scanning).
    PTB keeps the Bot API field in ``api_kwargs``, a ``mappingproxy`` (not a ``dict``), when it has no
    attribute for it."""
    rich = getattr(message, "rich_message", None)
    if rich is None and isinstance(kwargs := getattr(message, "api_kwargs", None), Mapping):
        rich = kwargs.get("rich_message")
    if not isinstance(rich, Mapping):
        return ""
    return "\n".join(_rich_lines(rich.get("blocks"), skip_code=skip_code)).strip()


def rich_message_filter(filters: Any) -> Any:
    """A PTB message filter matching a text-less message that carries ``rich_message`` content."""
    class _RichMessage(filters.MessageFilter):
        def filter(self, message: Any) -> bool:
            return not getattr(message, "text", None) and bool(rich_message_text(message))
    return _RichMessage(name="rich_message")


def _utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _code_point_index(text: str, utf16_offset: int) -> int | None:
    """Python index for a UTF-16 code-unit offset; None when it splits a surrogate pair."""
    try:
        return len(text.encode("utf-16-le")[: utf16_offset * 2].decode("utf-16-le"))
    except UnicodeDecodeError:
        return None


def expand_link_entities(message: Any) -> str:
    """Message text (or caption) with every hidden ``text_link`` URL inlined after its anchor.

    Entity offsets are UTF-16 code units (emoji before the anchor count twice), so they are mapped
    to code-point indices before slicing. Malformed entities are skipped; an anchor that already
    reads as its own URL, or text already carrying the inline form, is left untouched.
    """
    text = getattr(message, "text", None)
    if text:
        entities = getattr(message, "entities", None) or []
    else:
        text = getattr(message, "caption", None) or ""
        entities = getattr(message, "caption_entities", None) or []
    if not text:
        return rich_message_text(message)
    if not entities:
        return text

    utf16_length = _utf16_length(text)
    links: list[tuple[int, int, str]] = []
    for entity in entities:
        entity_type = str(getattr(entity, "type", "")).split(".")[-1].lower()
        raw_url = getattr(entity, "url", None)
        url = raw_url.strip() if isinstance(raw_url, str) else ""
        if entity_type != "text_link" or not url:
            continue
        try:
            offset = int(getattr(entity, "offset", -1))
            length = int(getattr(entity, "length", 0))
        except (TypeError, ValueError):
            continue
        if offset < 0 or length <= 0 or offset + length > utf16_length:
            continue
        start, end = _code_point_index(text, offset), _code_point_index(text, offset + length)
        if start is None or end is None or end <= start:
            continue
        if text[start:end].strip() == url:
            continue
        links.append((start, end, url))

    expanded = text
    for _start, end, url in sorted(links, reverse=True):
        inline = f" ({url})"
        if expanded[end:].startswith(inline):
            continue
        expanded = f"{expanded[:end]}{inline}{expanded[end:]}"
    return expanded
