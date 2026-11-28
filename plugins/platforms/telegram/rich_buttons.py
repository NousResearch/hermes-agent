"""`` ```buttons `` fence -> native buttons inside a rich message (Bot API 10.3 ``<tg-button>``).

Opt-in via ``telegram.extra.rich_buttons`` (default off: payloads stay byte-identical to today).

Why this exists: ``sendRichMessage`` takes no ``reply_markup`` and MarkdownV2 has no button
syntax, so a rich reply that needs buttons had to either degrade to MarkdownV2 (losing native
tables) or spend a second message. The rich grammar accepts ``<tg-button-row>`` with
``<tg-button>`` children, so the buttons can live *inside* the rich message.

Syntax — one button per line, ``Etichetta | [stile] | azione``, a leading ``+`` appends the
button to the previous row::

    ```buttons
    Approva | success | callback:approve_42
    Rifiuta | danger  | callback:reject_42
    + Docs  | link    | url:https://core.telegram.org/bots/api
    Copia   |         | copy:testo da copiare
    Spento  |         | disabled
    ```

Supported actions: ``callback:<data>`` (<= 64 bytes), ``url:<https|http|tg|mailto link>``,
``copy:<text>``, ``disabled``. Supported styles: ``primary``, ``success``, ``danger``,
``link``. Malformed lines are dropped, never rendered as literal text; a fence with no valid
button disappears entirely, so the syntax never leaks into the chat.
"""
from __future__ import annotations

import re

# A fence opens with ```buttons (nothing else on the line) and closes at the next bare ```.
_FENCE_OPEN = re.compile(r"(?m)^[ \t]*```buttons[ \t]*$")
_FENCE_CLOSE = re.compile(r"(?m)^[ \t]*```[ \t]*$")

_STYLES = frozenset({"primary", "success", "danger", "link"})
_URL_SCHEMES = ("https://", "http://", "tg://", "mailto:")
_MAX_CALLBACK_BYTES = 64
_MAX_URL_CHARS = 2048
_MAX_ROW_BUTTONS = 8
_ROW_ALIGN = "left"
_LABEL_ESCAPES = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})
_ATTR_FORBIDDEN = ("\n", "\r")


def has_button_fence(text: str) -> bool:
    """True when *text* carries a ``buttons`` fence — the cue that rich rendering is required."""
    return bool(text) and _FENCE_OPEN.search(text) is not None


def _escape_label(raw: str) -> str:
    return " ".join(raw.split()).translate(_LABEL_ESCAPES)


def _attr(value: str) -> str:
    return value.translate(_LABEL_ESCAPES)


def _button_html(line: str) -> str | None:
    """One ``label | [style] | action`` line -> ``<tg-button>`` tag, or None when unusable."""
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 2 or len(parts) > 3:
        return None
    label = _escape_label(parts[0])
    if not label:
        return None
    style = parts[1].lower() if len(parts) == 3 else ""
    if style and style not in _STYLES:
        return None
    action = parts[-1]
    kind, _, value = action.partition(":")
    kind, value = kind.strip().lower(), value.strip()
    if any(bad in value for bad in _ATTR_FORBIDDEN):
        return None
    style_attr = f' style="{style}"' if style else ""
    if kind == "disabled":
        return f'<tg-button type="disabled"{style_attr}>{label}</tg-button>'
    if kind == "callback":
        if not value or len(value.encode("utf-8")) > _MAX_CALLBACK_BYTES:
            return None
        return f'<tg-button type="callback_data"{style_attr} data="{_attr(value)}">{label}</tg-button>'
    if kind == "url":
        if len(value) > _MAX_URL_CHARS or not value.startswith(_URL_SCHEMES):
            return None
        return f'<tg-button type="url"{style_attr} url="{_attr(value)}">{label}</tg-button>'
    if kind == "copy":
        if not value:
            return None
        return f'<tg-button type="copy_text"{style_attr} text="{_attr(value)}">{label}</tg-button>'
    return None


def _render_fence(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        same_row = line.startswith("+")
        if same_row:
            line = line[1:].strip()
        button = _button_html(line)
        if button is None:
            continue
        if same_row and rows and len(rows[-1]) < _MAX_ROW_BUTTONS:
            rows[-1].append(button)
        elif not same_row or not rows:
            rows.append([button])
    if not rows:
        return ""
    return "\n".join(f'<tg-button-row align="{_ROW_ALIGN}">{"".join(row)}</tg-button-row>' for row in rows)


def render_rich_buttons(text: str) -> str:
    """Replace every ``buttons`` fence with its ``<tg-button-row>`` rows; leave the rest untouched."""
    if not has_button_fence(text):
        return text
    chunks: list[str] = []
    pos = 0
    while True:
        opening = _FENCE_OPEN.search(text, pos)
        if opening is None:
            break
        closing = _FENCE_CLOSE.search(text, opening.end())
        if closing is None:  # unclosed fence: never eat the rest of the message
            break
        chunks.append(text[pos:opening.start()])
        rendered = _render_fence(text[opening.end():closing.start()].splitlines())
        if not rendered and chunks:
            chunks[-1] = chunks[-1].rstrip("\n")
        chunks.append(rendered)
        pos = closing.end()
    chunks.append(text[pos:])
    return "".join(chunks)
