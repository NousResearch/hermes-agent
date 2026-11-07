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
from typing import Any, Optional

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

#: A `` ```card `` fenced code block. The JSON body is everything up to the
#: closing fence; a fence with no closing fence on the same message is not a
#: card (let the markdown path render it as a normal code block).
_CARD_FENCE_RE = re.compile(r"```card[ \t]*\r?\n(.*?)```", re.DOTALL)

_MRKDWN_ESCAPE_RE = re.compile(r"([*_`~])")


def _mrkdwn(text: str) -> str:
    """Escape Slack mrkdwn emphasis markers in model-provided strings.

    Card props are data (device names, values), not formatting: an unescaped
    ``*`` or ``_`` would pair up across fields and garble the fallback text.
    """
    return _MRKDWN_ESCAPE_RE.sub(r"\\\1", str(text))


def extract_card(content: str) -> tuple[Optional[dict[str, Any]], str]:
    """Pull the first `` ```card `` fenced JSON spec out of ``content``.

    Returns ``(spec, text_without_the_fence)``. When there is no (valid) card
    fence, returns ``(None, content)`` unchanged so the caller can fall back
    to the markdown path without touching the text.
    """
    if not content:
        return None, content
    match = _CARD_FENCE_RE.search(content)
    if not match:
        return None, content
    try:
        spec = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None, content
    if not isinstance(spec, dict):
        return None, content
    stripped = (content[: match.start()] + content[match.end() :]).strip()
    return spec, stripped


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
    clean = _mrkdwn(title).strip()
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
    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            return None
        cells = [display(cell) for cell in row]
        if any(cell is None for cell in cells):
            return None
        out_rows.append([{"type": "raw_text", "text": cell} for cell in cells])
    block: Block = {"type": "data_table", "rows": out_rows}
    caption = props.get("caption")
    if isinstance(caption, str) and caption.strip():
        block["caption"] = _mrkdwn(caption)[:200]
    return block


def render_card_blocks(spec: dict[str, Any]) -> Optional[list[Block]]:
    """Convert a json-render card spec to a Block Kit block list.

    Returns ``None`` (never raises) when the spec is empty, uses an unknown
    component or an invalid prop, or expands past Slack's block budget — the
    caller then falls back to the ordinary markdown path."""
    if not isinstance(spec, dict):
        return None
    elements = spec.get("elements")
    root = spec.get("root")
    if not isinstance(elements, dict) or not root or root not in elements:
        return None
    try:
        blocks = _render_children(elements, [root], depth=0)
    except Exception:  # defensive boundary: a bad card must never lose a message
        logger.warning("[Slack] card_blocks render failed; falling back to markdown", exc_info=True)
        return None
    if not blocks or len(blocks) > MAX_BLOCKS:
        return None
    return blocks


def _render_children(
    elements: dict[str, Any], ids: list[str], depth: int
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
        rendered = handler(props, children, (elements, depth))
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
    rendered = _render_children(ctx[0], children, ctx[1] + 1)
    if rendered is None:
        return None
    blocks.extend(rendered)
    image_url = _prop_str(props, "imageUrl")
    if image_url:
        blocks.append({"type": "image", "image_url": image_url, "alt_text": title or "card image"})
    return blocks


def _render_section(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    return _render_children(ctx[0], children, ctx[1] + 1)


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
    return [{"type": "image", "image_url": url, "alt_text": alt[:200]}]


def _render_actions(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    rendered = _render_children(ctx[0], children, ctx[1] + 1)
    if rendered is None or not rendered or len(rendered) > MAX_ACTIONS:
        return None
    action_ids = [b.get("action_id") for b in rendered if b.get("type") == "button"]
    if len(action_ids) != len(rendered) or len(set(action_ids)) != len(action_ids):
        return None
    return [{"type": "actions", "elements": rendered}]


def _render_button(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    label = _prop_str(props, "label", required=True)
    action_id = _prop_str(props, "id", required=True)
    if label is None or action_id is None:
        return None
    style = props.get("style")
    if style not in (None, "primary", "danger", "default"):
        return None
    button = _button_element(_mrkdwn(label), action_id, style, None)
    return [button] if button else None


def _render_link_button(props: dict[str, Any], children: list, ctx: tuple) -> Optional[list[Block]]:
    label = _prop_str(props, "label", required=True)
    url = _prop_str(props, "url", required=True)
    if label is None or url is None:
        return None
    button = _button_element(_mrkdwn(label), f"link-{url}", None, url)
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
    content: str,
) -> Optional[tuple[str, str, list[Block]]]:
    """Extract + render + summarize in one step.

    Returns ``(text_without_fence, fallback_text, blocks)`` when ``content``
    carries a renderable card spec, else ``None`` — the caller falls back to
    the markdown path with the content untouched. Never raises."""
    spec, stripped = extract_card(content)
    if spec is None:
        return None
    try:
        blocks = render_card_blocks(spec)
    except Exception:  # defensive boundary: a bad card must never lose a message
        logger.warning("[Slack] card_blocks render failed; falling back to markdown", exc_info=True)
        return None
    if not blocks:
        return None
    return stripped, card_fallback_text(spec), blocks
