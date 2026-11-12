"""Inbound message text the adapter facade reads: ``text_link`` expansion and ``rich_message`` blocks.

A rich-text hyperlink arrives as visible anchor text plus a ``text_link`` entity carrying the
URL; the model only ever sees the anchor unless the URL is inlined (#31071). Long multi-paragraph
messages can instead arrive with an empty ``text`` and their content in ``rich_message.blocks``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any


# Real clients nest a handful of levels; content deeper than this is dropped so a hostile payload
# (valid JSON thousands of levels deep) cannot hit the recursion limit and drop the whole update.
_MAX_RICH_DEPTH = 64
_CODE_TYPES = frozenset({"code", "pre"})
# A rich link label is a text_link, not a bot_command or mention. Only the
# entity-synthesis view masks it; the agent-visible text keeps its anchor.
_ADDRESSING_MASK_TYPES = _CODE_TYPES | frozenset({"url", "link", "text_link"})
# Stands in for skipped code: leading content like the code in plain text, never whitespace, a separator or a handle.
_CODE_PLACEHOLDER = "\ufffc"


def _rich_inline(value: Any, depth: int = 0, skip_code: bool = False, urls: bool = True) -> str:
    """Plain text of a rich ``text`` value: a string, a list of runs, or a span dict (bold, link…)
    whose content sits in ``text`` or ``children``, falling back to a link's ``url``, a math
    ``expression`` or a custom emoji's ``alternative_text``. ``skip_code`` masks ``code``/``pre`` and linked-label spans with a placeholder;
    ``urls`` inlines a hidden link's URL after its anchor."""
    if depth > _MAX_RICH_DEPTH:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_rich_inline(part, depth + 1, skip_code, urls) for part in value)
    if isinstance(value, Mapping):
        if skip_code and value.get("type") in _ADDRESSING_MASK_TYPES:
            return _CODE_PLACEHOLDER
        for key in ("text", "children"):
            if value.get(key) is not None:
                text = _rich_inline(value[key], depth + 1, skip_code, urls)
                # A hidden link keeps its URL, as ``expand_link_entities`` inlines a ``text_link``; the
                # addressing views leave it out so ``?u=@bot`` in a URL cannot wake the bot.
                url = value.get("url") if urls else None
                return f"{text} ({url})" if isinstance(url, str) and url and url != text else text
        return str(value.get("url") or value.get("expression") or value.get("alternative_text") or "")
    return ""


def _rich_lines(blocks: Any, prefix: str = "", depth: int = 0, skip_code: bool = False, urls: bool = True) -> list[str]:
    """One line per paragraph-like block (``details`` summary and math included) and per table row; a
    list item's ``label`` (bullet/number) prefixes its first line."""
    lines: list[str] = []
    if depth > _MAX_RICH_DEPTH:
        return lines
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, Mapping):
            continue
        if skip_code and block.get("type") in _ADDRESSING_MASK_TYPES:
            lines.append(prefix + _CODE_PLACEHOLDER)
            prefix = ""
            continue
        for key in ("text", "summary", "expression"):  # paragraph-like, details summary, math block
            if key in block:
                lines.append(prefix + _rich_inline(block[key], skip_code=skip_code, urls=urls))
                prefix = ""
        cells, items = block.get("cells"), block.get("items")
        for row in cells if isinstance(cells, list) else []:  # table: one line per row
            if isinstance(row, list):
                lines.append(" | ".join(_rich_inline(c.get("text"), skip_code=skip_code, urls=urls)
                                        for c in row if isinstance(c, Mapping)))
        for item in items if isinstance(items, list) else []:
            if isinstance(item, Mapping):
                label = str(item.get("label") or "-")
                lines += _rich_lines(item.get("blocks") or [item], f"{label} ", depth + 1, skip_code, urls)
        if block.get("blocks"):
            lines += _rich_lines(block["blocks"], prefix, depth + 1, skip_code, urls)
    return lines


def rich_message_text(message: Any, skip_code: bool = False, urls: bool | None = None) -> str:
    """Flattened text of a ``rich_message`` payload ("" when there is none), for inbound messages and
    the replied-to message alike; ``skip_code`` masks ``code``/``pre`` and linked-label spans with a placeholder, ``urls`` (default: not
    ``skip_code``) inlines hidden link URLs. PTB keeps the Bot API field in ``api_kwargs``, a
    ``mappingproxy`` (not a ``dict``), when it has no attribute for it."""
    rich = getattr(message, "rich_message", None)
    if rich is None and isinstance(kwargs := getattr(message, "api_kwargs", None), Mapping):
        rich = kwargs.get("rich_message")
    if not isinstance(rich, Mapping):
        return ""
    return "\n".join(_rich_lines(rich.get("blocks"), skip_code=skip_code,
                                 urls=not skip_code if urls is None else urls)).strip()


# What Telegram itself marks as a mention / bot command in plain text; a rich-only message carries no entities.
_RICH_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_`/])@[A-Za-z0-9_]{2,31}\b")
_RICH_COMMAND_RE = re.compile(r"(?<!\S)/[A-Za-z0-9_]{1,64}(?:@[A-Za-z0-9_]{2,32})?(?![A-Za-z0-9_@/\\])")


def _rich_addressing_entities(text: str) -> list[SimpleNamespace]:
    found = [SimpleNamespace(type=etype, offset=_utf16_length(text[:m.start()]), length=_utf16_length(m.group()))
             for etype, pattern in (("bot_command", _RICH_COMMAND_RE), ("mention", _RICH_MENTION_RE))
             for m in pattern.finditer(text)]
    return sorted(found, key=lambda entity: entity.offset)


def addressing_sources(message: Any) -> list[tuple[str, list]]:
    """``(text, entities)`` every mention / command / addressee check reads: the text and the caption, or for a
    text-less ``rich_message`` its blocks with code and linked labels as placeholders and hidden URLs left out, carrying the
    ``mention``/``bot_command`` entities Telegram would have sent for that text as plain text (#151730)."""
    text, caption = getattr(message, "text", None) or "", getattr(message, "caption", None) or ""
    sources = [(text, getattr(message, "entities", None) or []), (caption, getattr(message, "caption_entities", None) or [])]
    if not text and not caption:
        rich = rich_message_text(message, skip_code=True)
        sources.append((rich, _rich_addressing_entities(rich)))
    return sources


def addressing_text(message: Any) -> str:
    """The text wake-word patterns and the group ``?``/``/`` reply rules read: text, caption, else the rich
    blocks with code kept and hidden link URLs left out (a link to ``chompy.example.com`` is not a wake word)."""
    return (getattr(message, "text", None) or getattr(message, "caption", None)
            or rich_message_text(message, urls=False))


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
