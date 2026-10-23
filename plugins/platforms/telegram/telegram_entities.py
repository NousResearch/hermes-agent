"""Telegram ``text_link`` expansion, kept out of the adapter facade.

A rich-text hyperlink arrives as visible anchor text plus a ``text_link`` entity carrying the
URL; the model only ever sees the anchor unless the URL is inlined (#31071).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


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
    if not text or not entities:
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


def expand_quote_link_entities(quote: Any, replied: Any) -> str:
    """A native partial quote's text with the replied-to message's hidden links inlined.

    Telegram keeps only formatting entities in a quote, never ``text_link``, so the links are
    taken from the replied-to message. ``position`` is approximate, so they are only mapped when
    the replied-to text at that position is exactly the quote; otherwise the quote is unchanged.
    """
    text = getattr(quote, "text", None)
    source = getattr(replied, "text", None)
    if not text or not isinstance(text, str):
        return text
    if source:
        entities = getattr(replied, "entities", None) or []
    else:
        source = getattr(replied, "caption", None) or ""
        entities = getattr(replied, "caption_entities", None) or []
    if not isinstance(source, str):
        return text
    try:
        position = int(getattr(quote, "position", -1))
    except (TypeError, ValueError):
        return text
    start = _code_point_index(source, position) if 0 <= position <= _utf16_length(source) else None
    if not entities or start is None or source[start : start + len(text)] != text:
        return text
    quote_end = position + _utf16_length(text)
    shifted = []
    for entity in entities:
        try:
            offset, length = int(getattr(entity, "offset", -1)), int(getattr(entity, "length", 0))
        except (TypeError, ValueError):
            continue
        if position <= offset and offset + length <= quote_end:
            shifted.append(SimpleNamespace(
                type=getattr(entity, "type", ""), offset=offset - position, length=length,
                url=getattr(entity, "url", None)))
    return expand_link_entities(SimpleNamespace(text=text, entities=shifted))
