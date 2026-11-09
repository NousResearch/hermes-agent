"""Inbound message text the adapter facade reads: ``text_link`` expansion and ``rich_message`` blocks.

A rich-text hyperlink arrives as visible anchor text plus a ``text_link`` entity carrying the
URL; the model only ever sees the anchor unless the URL is inlined (#31071). Long multi-paragraph
messages can instead arrive with an empty ``text`` and their content in ``rich_message.blocks``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


# Real clients nest a handful of levels; content deeper than this is dropped so a hostile payload
# (valid JSON thousands of levels deep) cannot hit the recursion limit and drop the whole update.
_MAX_RICH_DEPTH = 64
_CODE_TYPES = frozenset({"code", "pre"})


def _rich_inline(value: Any, depth: int = 0, skip_code: bool = False) -> str:
    """Plain text of a rich ``text`` value: a string, a list of runs, or a span dict (bold, link…)
    whose content sits in ``text`` or ``children``, falling back to a link's ``url``, a math
    ``expression`` or a custom emoji's ``alternative_text``. ``skip_code`` blanks ``code``/``pre`` spans."""
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
                text = _rich_inline(value[key], depth + 1, skip_code)
                # A hidden link keeps its URL, as ``expand_link_entities`` inlines a ``text_link``; the
                # mention scan leaves it out so ``?u=@bot`` in a URL cannot wake the bot.
                url = None if skip_code else value.get("url")
                return f"{text} ({url})" if isinstance(url, str) and url and url != text else text
        return str(value.get("url") or value.get("expression") or value.get("alternative_text") or "")
    return ""


def _rich_lines(blocks: Any, prefix: str = "", depth: int = 0, skip_code: bool = False) -> list[str]:
    """One line per paragraph-like block (``details`` summary and math included) and per table row; a
    list item's ``label`` (bullet/number) prefixes its first line."""
    lines: list[str] = []
    if depth > _MAX_RICH_DEPTH:
        return lines
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, Mapping) or (skip_code and block.get("type") in _CODE_TYPES):
            continue
        for key in ("text", "summary", "expression"):  # paragraph-like, details summary, math block
            if key in block:
                lines.append(prefix + _rich_inline(block[key], skip_code=skip_code))
                prefix = ""
        cells, items = block.get("cells"), block.get("items")
        for row in cells if isinstance(cells, list) else []:  # table: one line per row
            if isinstance(row, list):
                lines.append(" | ".join(_rich_inline(c.get("text"), skip_code=skip_code)
                                        for c in row if isinstance(c, Mapping)))
        for item in items if isinstance(items, list) else []:
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
    """A PTB message filter matching a text-less, attachment-less message that carries ``rich_message``
    content (a caption-less photo with rich content stays with the media handler)."""
    class _RichMessage(filters.MessageFilter):
        def filter(self, message: Any) -> bool:
            return (not getattr(message, "text", None) and not getattr(message, "effective_attachment", None)
                    and bool(rich_message_text(message)))
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
