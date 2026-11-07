"""Render a json-render chat card spec into Slack Block Kit blocks.

Opt-in (``slack.extra.card_blocks: true``) escape hatch from markdown
rendering: the agent wraps a [json-render](https://json-render.dev) card spec
in a `` ```card `` fenced code block, and the adapter posts the spec as a
native Block Kit card instead of rendering the fence as markdown.

Design constraints (mirroring :mod:`block_kit.py`):

* **Never raise.** Any unexpected input — invalid JSON, an unknown component,
  a bad prop, a structure over Slack's limits — returns ``None`` and the
  caller falls back to the ordinary markdown path. A card is a nice-to-have;
  it must never lose a message.
* **The text fallback is a plain mrkdwn summary** of the card (title, label:
  value lines, an aligned monospace table), not the raw JSON: Slack uses it
  for notifications, screen readers and old clients, and
  ``_call_with_block_fallback`` posts it when Slack rejects the blocks.
* **Supported components** are the standard json-render chat catalog subset
  that maps 1:1 onto Block Kit: Card, Section, Text, Fields, Field, Table,
  Divider, Image, Actions, Button, LinkButton. ``Chart`` has no Slack
  equivalent, so a spec containing one is rejected (``None``), never
  degraded silently.
* Pure functions of their input — no Slack client, no adapter state — so
  they are trivially unit-testable.

The mapping follows ``@json-render/chat``'s ``@chat-adapter/slack`` output
(``cardToSlackBlocks``): ``header`` + ``context`` for the card title/subtitle,
``section`` (with ``fields``) for Field grids, native ``data_table`` blocks
for tables, and ``actions`` blocks with ``plain_text`` buttons. LinkButtons
become buttons with a ``link-<url>`` action id, exactly as the adapter does.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

Block = dict[str, Any]

logger = logging.getLogger(__name__)

# Slack Block Kit hard limits (https://docs.slack.dev/reference/block-kit/blocks)
MAX_BLOCKS = 50
MAX_HEADER_TEXT = 150
MAX_SECTION_TEXT = 3000
# Native table block limits
# (https://docs.slack.dev/reference/block-kit/blocks/table-block)
MAX_TABLE_ROWS = 100
MAX_TABLE_COLS = 20
#: Slack actions blocks hold at most 5 elements.
MAX_ACTIONS = 5
#: Nesting guard: a card is header/sections/actions, not a tree.
_MAX_DEPTH = 8
# Slack section blocks hold at most 10 fields.
MAX_FIELDS_PER_SECTION = 10
# action_id cap (https://docs.slack.dev/reference/block-kit/block-objects/button)
MAX_ACTION_ID = 255

#: Aggregate character budget across all cells of one table block.
MAX_DATA_TABLE_CHARS = 20000  # data_table (https://docs.slack.dev/reference/block-kit/blocks/data-table-block)
MAX_TABLE_CHARS = 10000  # table (https://docs.slack.dev/reference/block-kit/blocks/table-block)
#: data_table ``page_size`` bounds; Slack defaults to 5 rows per page when omitted.
MAX_TABLE_PAGE_SIZE = 100
#: ``caption`` is REQUIRED on data_table; @chat-adapter/slack defaults it the same way.
DEFAULT_TABLE_CAPTION = "Table"

#: Slack's three control characters in ``mrkdwn`` text (links, mentions, dates).
#: Slack has NO backslash escape, so ``*``/``_`` cannot be neutralised; entity
#: escaping is the only escaping Slack performs
#: (https://docs.slack.dev/messaging/formatting-message-text/#escaping-text).
_ENTITY_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))


def _mrkdwn(text: Any) -> str:
    """Escape model-provided data for an ``mrkdwn`` text object.

    Card props are data (ticket titles, device names, field values), not
    formatting. Unescaped ``<!channel>``, ``<@U…>`` or ``<url|label>`` in a
    prop would ping the room, mention a person or render a spoofed link, so
    ``& < >`` become entities exactly as Slack requires. Emphasis markers are
    left alone: Slack ignores backslashes, so a backslash-escaped asterisk
    would render literally.
    """
    out = str(text)
    for raw, entity in _ENTITY_ESCAPES:
        out = out.replace(raw, entity)
    return out


def _plain(text: Any) -> str:
    """Text for a ``plain_text`` object (header, button label, caption).

    ``plain_text`` is never parsed for links or mentions and entities are not
    decoded there, so the value goes through verbatim.
    """
    return str(text)


_FENCE_LINE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(\S*)[ \t]*$")


def _top_level_card_fences(content: str) -> list[tuple[int, int, str]]:
    """``(start, end, body)`` of every CLOSED `` ```card `` fence that is not
    nested inside another fence.

    A ```` ````markdown ```` block quoting a card example must not be mistaken
    for the card, and a fence the model never closed (truncated reply) is not
    a card either: both stay in the text and render as ordinary code.
    """
    out: list[tuple[int, int, str]] = []
    open_marker: Optional[str] = None  # marker chars of the fence currently open
    open_is_card = False
    open_start = 0
    body_start = 0
    pos = 0
    for line in content.splitlines(keepends=True):
        m = _FENCE_LINE_RE.match(line.rstrip("\r\n"))
        if m:
            marker, info = m.group(1), m.group(2)
            if open_marker is None:
                open_marker, open_is_card, open_start = marker, info == "card", pos
                body_start = pos + len(line)
            elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not info:
                if open_is_card:
                    # end excludes the closing fence's newline, so the prose
                    # around the card keeps its own line breaks
                    out.append((open_start, pos + len(line.rstrip("\r\n")), content[body_start:pos]))
                open_marker, open_is_card = None, False
        pos += len(line)
    return out


def extract_card(content: str) -> tuple[Optional[dict[str, Any]], str]:
    """Pull the `` ```card `` fenced JSON spec out of ``content``.

    Returns ``(spec, text_without_the_fence)``. When there is no (valid) card
    fence, returns ``(None, content)`` unchanged so the caller can fall back
    to the markdown path without touching the text. Only top-level, closed
    fences count; when the reply holds several, the LAST well-formed one is
    the card (a model that quotes the format first and then answers).
    """
    if not content:
        return None, content
    for start, end, body in reversed(_top_level_card_fences(content)):
        try:
            spec = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        stripped = (content[:start] + content[end:]).strip()
        return spec, stripped
    return None, content


# ----------------------------------------------------------------------------
# Spec → blocks
# ----------------------------------------------------------------------------


def _prop_str(props: dict[str, Any], key: str, *, required: bool = False) -> Optional[str]:
    """Read a display-string prop (numbers tolerated, like the catalog's displayText)."""
    value = props.get(key)
    if value is None:
        return "" if not required else None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _prop_text(props: dict[str, Any], key: str, *, required: bool = False) -> Optional[str]:
    """Read a strict string prop (the catalog's z.string(): no number coercion).

    Card.title/subtitle and Text.content are formatting surface, not data —
    a number there means the model misunderstood the schema, so reject."""
    value = props.get(key)
    if value is None:
        return "" if not required else None
    return value if isinstance(value, str) else None


def _header_block(title: str) -> Optional[Block]:
    clean = _plain(title).strip()
    if not clean:
        return None
    return {"type": "header", "text": {"type": "plain_text", "text": clean[:MAX_HEADER_TEXT], "emoji": True}}


def _text_element(text: str) -> dict[str, Any]:
    return {"type": "mrkdwn", "text": text[:MAX_SECTION_TEXT]}


def _section_with_fields(pairs: list[tuple[str, str]]) -> Block:
    return {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": f"*{_mrkdwn(label)}*\n{_mrkdwn(value)}"[:MAX_SECTION_TEXT]}
            for label, value in pairs
        ],
    }


def _button_element(label: str, action_id: str, style: Optional[str], url: Optional[str]) -> Optional[Block]:
    if not label or not action_id or len(action_id) > MAX_ACTION_ID:
        return None
    button: Block = {
        "type": "button",
        "text": {"type": "plain_text", "text": label[:75], "emoji": True},
        "action_id": action_id,
    }
    if style in ("primary", "danger"):
        button["style"] = style
    if url is not None:
        button["url"] = url
    return button


def _table_block(props: dict[str, Any]) -> Optional[Block]:
    headers = props.get("headers")
    rows = props.get("rows")
    if not isinstance(headers, list) or not headers or not isinstance(rows, list):
        return None
    if len(headers) > MAX_TABLE_COLS or len(rows) > MAX_TABLE_ROWS:
        return None
    display = lambda v: ("" if v is None else str(v)) if isinstance(v, (str, int, float)) and not isinstance(v, bool) else None
    header_cells = [display(h) for h in headers]
    if any(cell is None for cell in header_cells):
        return None
    out_rows: list[list[Block]] = [
        [{"type": "raw_text", "text": cell} for cell in header_cells]
    ]
    total_chars = sum(len(cell) for cell in header_cells)
    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            return None
        cells = [display(cell) for cell in row]
        if any(cell is None for cell in cells):
            return None
        total_chars += sum(len(cell) for cell in cells)
        out_rows.append([{"type": "raw_text", "text": cell} for cell in cells])
    if not rows:
        # data_table needs a header plus at least one row; the JS adapter emits
        # a plain ``table`` for an empty result set, so mirror it.
        if total_chars > MAX_TABLE_CHARS:
            return None
        return {"type": "table", "rows": out_rows}
    # Slack rejects the WHOLE message past the aggregate cell budget; the
    # renderer checks it here so the caller falls back instead of losing the card.
    if total_chars > MAX_DATA_TABLE_CHARS:
        return None
    caption = props.get("caption")
    caption_text = _plain(caption).strip() if isinstance(caption, str) else ""
    block: Block = {
        "type": "data_table",
        # REQUIRED by Slack; @chat-adapter/slack defaults it to "Table" too.
        "caption": (caption_text or DEFAULT_TABLE_CAPTION)[:200],
        "rows": out_rows,
    }
    page_size = props.get("pageSize")
    if isinstance(page_size, (int, float)) and not isinstance(page_size, bool):
        block["page_size"] = max(1, min(MAX_TABLE_PAGE_SIZE, int(page_size)))
    return block


ButtonPolicy = Callable[[str], bool]


def render_card_blocks(
    spec: dict[str, Any], *, button_allowed: Optional[ButtonPolicy] = None
) -> Optional[list[Block]]:
    """Convert a json-render card spec to a Block Kit block list.

    Returns ``None`` (never raises) when the spec is empty, uses an unknown
    component or an invalid prop, or expands past Slack's block budget — the
    caller then falls back to the ordinary markdown path.

    ``button_allowed(action_id)`` decides whether an interactive ``Button`` has
    a registered click handler. A button nobody handles is not harmless: Slack
    shows "This app is not responding" on the first tap. Unhandled buttons are
    DROPPED (and an ``Actions`` left empty is dropped with them) rather than
    rejecting the card, because the rest of the card is still the answer.
    ``None`` keeps every button (pure renderer parity with the JS adapter)."""
    if not isinstance(spec, dict):
        return None
    elements = spec.get("elements")
    root = spec.get("root")
    if not isinstance(elements, dict) or not root or root not in elements:
        return None
    try:
        blocks = _render_children(elements, [root], depth=0, button_allowed=button_allowed)
    except Exception:  # defensive boundary: a bad card must never lose a message
        logger.warning("[Slack] card_blocks render failed; falling back to markdown", exc_info=True)
        return None
    if not blocks or len(blocks) > MAX_BLOCKS:
        return None
    return blocks


def _render_children(
    elements: dict[str, Any], ids: list[str], depth: int,
    button_allowed: Optional[ButtonPolicy] = None,
) -> Optional[list[Block]]:
    """Depth-first render of the children ``ids``; ``None`` on any problem."""
    if depth > _MAX_DEPTH:  # a card is header/sections/actions, not a tree
        return None
    blocks: list[Block] = []
    for element_id in ids:
        element = elements.get(element_id)
        if not isinstance(element, dict):
            return None
        etype = element.get("type")
        props = element.get("props") or {}
        if not isinstance(props, dict):
            return None
        children = element.get("children") or []
        if not isinstance(children, list):
            return None
        handler = _COMPONENT_HANDLERS.get(etype) if isinstance(etype, str) else None
        if handler is None:
            # Unknown component (e.g. Chart — no Slack equivalent): reject.
            return None
        rendered = handler(props, children, (elements, depth, button_allowed))
        if rendered is None:
            return None
        blocks.extend(rendered)
    return blocks


def _render_card(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    title = _prop_text(props, "title")
    if title is None:
        return None
    blocks: list[Block] = []
    if title:
        header = _header_block(title)
        if header:
            blocks.append(header)
    subtitle = _prop_text(props, "subtitle")
    if subtitle:
        blocks.append({"type": "context", "elements": [_text_element(_mrkdwn(subtitle))]})
    rendered = _render_children(ctx[0], children, ctx[1] + 1, ctx[2])
    if rendered is None:
        return None
    blocks.extend(rendered)
    image_url = _prop_str(props, "imageUrl")
    if image_url:
        if not image_url.startswith("https://"):
            return None  # same rule as Image: Slack fetches it, so https only
        blocks.append({"type": "image", "image_url": image_url, "alt_text": _plain(title) or "card image"})
    return blocks


def _render_section(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    return _render_children(ctx[0], children, ctx[1] + 1, ctx[2])


def _render_text(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    content = _prop_text(props, "content", required=True)
    if content is None or not content.strip():
        return None
    style = props.get("style")
    if style not in (None, "plain", "bold", "muted"):
        return None
    text = _mrkdwn(content)
    if style == "bold":
        text = f"*{text}*"
    elif style == "muted":
        text = f"_{text}_"
    return [{"type": "section", "text": _text_element(text)}]


def _render_fields(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    pairs = _collect_fields(ctx[0], children)
    if pairs is None:
        return None
    # Split on the 10-fields-per-section cap.
    return [
        _section_with_fields(pairs[i : i + MAX_FIELDS_PER_SECTION])
        for i in range(0, len(pairs), MAX_FIELDS_PER_SECTION)
    ]


def _render_field(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    # A bare Field outside Fields: render it as a one-pair section so the
    # slot-free usage still shows up rather than silently vanishing.
    label = _prop_str(props, "label", required=True)
    value = _prop_str(props, "value")
    if label is None or value is None:
        return None
    return [_section_with_fields([(label, value)])]


def _render_table(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    block = _table_block(props)
    return [block] if block else None


def _render_divider(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    return [{"type": "divider"}]


def _render_image(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    url = _prop_str(props, "url", required=True)
    alt = _prop_str(props, "alt", required=True)
    if url is None or alt is None or not url.startswith("https://"):
        return None
    return [{"type": "image", "image_url": url, "alt_text": _plain(alt)[:200]}]


def _render_actions(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    rendered = _render_children(ctx[0], children, ctx[1] + 1, ctx[2])
    if rendered is None or not rendered or len(rendered) > MAX_ACTIONS:
        return None
    action_ids = [b.get("action_id") for b in rendered if b.get("type") == "button"]
    if len(action_ids) != len(rendered) or len(set(action_ids)) != len(action_ids):
        return None
    policy = ctx[2]
    if policy is not None:
        kept = [b for b in rendered if "url" in b or policy(str(b.get("action_id")))]
        dropped = len(rendered) - len(kept)
        if dropped:
            logger.warning(
                "[Slack] card_blocks: dropped %d Button(s) with no registered action handler "
                "(register one via ctx.register_slack_action_handler, or use LinkButton)", dropped)
        if not kept:
            return []  # an Actions row with nothing left simply disappears
        rendered = kept
    return [{"type": "actions", "elements": rendered}]


def _render_button(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    label = _prop_str(props, "label", required=True)
    action_id = _prop_str(props, "id", required=True)
    if label is None or action_id is None:
        return None
    style = props.get("style")
    if style not in (None, "primary", "danger", "default"):
        return None
    button = _button_element(_plain(label), action_id, style, None)
    return [button] if button else None


def _render_link_button(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    label = _prop_str(props, "label", required=True)
    url = _prop_str(props, "url", required=True)
    if label is None or url is None:
        return None
    button = _button_element(_plain(label), f"link-{url}", None, url)
    return [button] if button else None


#: Per-component renderers; a type missing here (Chart, ...) rejects the card.
_COMPONENT_HANDLERS = {
    "Card": _render_card,
    "Section": _render_section,
    "Text": _render_text,
    "Fields": _render_fields,
    "Field": _render_field,
    "Table": _render_table,
    "Divider": _render_divider,
    "Image": _render_image,
    "Actions": _render_actions,
    "Button": _render_button,
    "LinkButton": _render_link_button,
}


def _collect_fields(
    elements: dict[str, Any], ids: list[str]
) -> Optional[list[tuple[str, str]]]:
    pairs: list[tuple[str, str]] = []
    for element_id in ids:
        element = elements.get(element_id)
        if not isinstance(element, dict) or element.get("type") != "Field":
            return None
        props = element.get("props") or {}
        if not isinstance(props, dict):
            return None
        label = _prop_str(props, "label", required=True)
        value = _prop_str(props, "value")
        if label is None or value is None:
            return None
        pairs.append((label, value))
    return pairs or None


# ----------------------------------------------------------------------------
# Text fallback
# ----------------------------------------------------------------------------


def _table_fallback_lines(props: dict[str, Any], lines: list[str]) -> None:
    """Append an aligned monospace table (max 10 rows in the fallback)."""
    headers = props.get("headers")
    rows = props.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        return
    table = [headers, *rows][: 1 + 10]
    str_rows = [[str(cell) for cell in row] for row in table if isinstance(row, list)]
    if not str_rows:
        return
    width = [max(len(row[col]) if col < len(row) else 0 for row in str_rows)
             for col in range(max(len(r) for r in str_rows))]
    for i, row in enumerate(str_rows):
        padded = " | ".join(
            (row[col] if col < len(row) else "").ljust(width[col])
            for col in range(len(width)))
        lines.append(padded)
        if i == 0:
            lines.append("-+-".join("-" * w for w in width))


def _card_lines(props: dict[str, Any], lines: list[str]) -> None:
    for key in ("title", "subtitle"):
        value = _prop_str(props, key)
        if value:
            lines.append(f"*{_mrkdwn(value)}*" if key == "title" else _mrkdwn(value))


def _text_lines(props: dict[str, Any], lines: list[str]) -> None:
    content = _prop_str(props, "content")
    if content:
        lines.append(_mrkdwn(content))


def _field_lines(props: dict[str, Any], lines: list[str]) -> None:
    label = _prop_str(props, "label")
    value = _prop_str(props, "value")
    if label is not None and value is not None:
        lines.append(f"{_mrkdwn(label)}: {_mrkdwn(value)}")


def _button_lines(props: dict[str, Any], lines: list[str]) -> None:
    label = _prop_str(props, "label")
    if label:
        lines.append(f"[{_mrkdwn(label)}]")


def _link_button_lines(props: dict[str, Any], lines: list[str]) -> None:
    label = _prop_str(props, "label")
    url = _prop_str(props, "url")
    if label:
        lines.append(f"[{_mrkdwn(label)}]({_mrkdwn(url or '')})" if url else f"[{_mrkdwn(label)}]")


#: Per-component fallback-line emitters; types without one contribute nothing.
_FALLBACK_LINES = {
    "Card": _card_lines,
    "Text": _text_lines,
    "Field": _field_lines,
    "Table": _table_fallback_lines,
    "Button": _button_lines,
    "LinkButton": _link_button_lines,
}


def card_fallback_text(spec: dict[str, Any]) -> str:
    """A plain mrkdwn summary of the card: title, subtitle, lines, ASCII table.

    Mirrors ``@chat-adapter/slack``'s ``cardToFallbackText`` closely enough
    for notifications and screen readers; this is what Slack shows when blocks
    are dropped (and what ``_call_with_block_fallback`` posts)."""
    elements = spec.get("elements") if isinstance(spec, dict) else None
    if not isinstance(elements, dict):
        return ""
    lines: list[str] = []
    for element in elements.values():
        if not isinstance(element, dict):
            continue
        props = element.get("props")
        if not isinstance(props, dict):
            continue
        etype = element.get("type")
        emitter = _FALLBACK_LINES.get(etype) if isinstance(etype, str) else None
        if emitter is not None:
            emitter(props, lines)
    return "\n".join(lines)


def card_blocks_payload(
    content: str, *, button_allowed: Optional[ButtonPolicy] = None,
) -> Optional[tuple[str, str, list[Block]]]:
    """Extract + render + summarize in one step.

    Returns ``(text_without_fence, fallback_text, blocks)`` when ``content``
    carries a renderable card spec, else ``None`` — the caller falls back to
    the markdown path with the content untouched. Never raises."""
    spec, stripped = extract_card(content)
    if spec is None:
        return None
    try:
        blocks = render_card_blocks(spec, button_allowed=button_allowed)
    except Exception:  # defensive boundary: a bad card must never lose a message
        logger.warning("[Slack] card_blocks render failed; falling back to markdown", exc_info=True)
        return None
    if not blocks:
        return None
    return stripped, card_fallback_text(spec), blocks
