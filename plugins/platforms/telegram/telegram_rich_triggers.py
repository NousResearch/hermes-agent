"""Conservative trigger text for rich-only input; display projection is separate."""

import re


def entity_sources(adapter, message):
    """Keep legacy entity offsets bound to their original text/caption."""
    yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
    yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []
    if not (getattr(message, "text", None) or getattr(message, "caption", None)):
        for text in rich_trigger_text(adapter, message):
            yield text, []


def rich_trigger_text(adapter, message):
    """Expose plain prose to the existing mention/wake-word rules.

    Rich nodes have no legacy MessageEntity offsets. Never apply those offsets to
    a Markdown projection, or interpret URL/code/math/unknown nodes as addressing.
    Each leaf stays separate so joining nodes cannot manufacture a bot handle.
    """
    get = adapter._rich_get

    def inline(value):
        if isinstance(value, str):
            # Entity-less rich prose can contain literal URLs or backtick code too.
            yield re.sub(
                r"(?is)`+.*?`+|\b[a-z][a-z0-9+.-]*://\S+|\b(?:www\.|mailto:)\S+",
                " ", value,
            )
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from inline(item)
        elif get(value, "type") in ("text", "plain", "bold", "italic", "underline", "strikethrough", "spoiler"):
            yield from inline(get(value, "text"))

    def nested(block):
        yield from inline(get(block, "summary"))
        yield from blocks(get(block, "blocks"))

    def listing(block):
        items = get(block, "items", [])
        if isinstance(items, (list, tuple)):
            for item in items:
                yield from blocks(get(item, "blocks"))

    def table(block):
        yield from inline(get(block, "caption"))
        rows = get(block, "cells", [])
        if isinstance(rows, (list, tuple)):
            for row in rows:
                if isinstance(row, (list, tuple)):
                    for cell in row:
                        yield from inline(get(cell, "text"))

    handlers = dict.fromkeys(
        ("paragraph", "heading", "subheading", "title", "subtitle"),
        lambda block: inline(get(block, "text")),
    )
    handlers.update(blockquote=nested, details=nested, list=listing, table=table)

    def blocks(values):
        if isinstance(values, (list, tuple)):
            for block in values:
                kind = get(block, "type")
                handler = handlers.get(kind) if isinstance(kind, str) else None
                if handler:
                    yield from handler(block)

    yield from blocks(get(adapter._inbound_rich_message_payload(message), "blocks"))
