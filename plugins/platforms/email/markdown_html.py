"""Render agent Markdown as inline-styled email HTML.

The email adapter sends ``text/plain`` only, so Markdown the agent writes for chat
(``**bold**``, ``## Headings``, bullets) reaches a mail reader as literal asterisks and
hashes. This converts the subset the agent actually emits into simple HTML, which the
adapter attaches as a ``text/html`` alternative *beside* the unchanged plain part:
HTML-capable clients render it, and every other client still gets the original text.
"""

from __future__ import annotations

import html
import re

# Anchored block markers. A heading requires its space, so "#hashtag" stays prose.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\d+[.)]\s+(.*)$")

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# The lookbehind skips a URL already inside an href or a wrapped parenthetical.
_BARE_URL = re.compile(r"""(?<!["'=(])(https?://[^\s<>()]+)""")

_SENTENCE_END = ".,;:!?"
_HEADING_STYLE = {
    2: "margin:18px 0 8px;font-size:1.25em;font-weight:600",
    3: "margin:16px 0 8px;font-size:1.15em;font-weight:600",
    4: "margin:14px 0 6px;font-size:1.05em;font-weight:600",
    5: "margin:12px 0 6px;font-size:1em;font-weight:600",
    6: "margin:12px 0 6px;font-size:1em;font-weight:600",
}
_PARAGRAPH_STYLE = "margin:0 0 12px"
_LIST_STYLE = "margin:0 0 12px;padding-left:22px"
_ITEM_STYLE = "margin:0 0 4px"


def _linkify(match: re.Match) -> str:
    """Wrap a bare URL, leaving sentence punctuation outside the anchor."""
    url, trailing = match.group(1), ""
    while url and url[-1] in _SENTENCE_END:
        trailing = url[-1] + trailing
        url = url[:-1]
    return f'<a href="{url}">{url}</a>{trailing}' if url else match.group(0)


def _inline(text: str) -> str:
    """Escape, then apply inline Markdown. Escaping first means content cannot inject markup."""
    text = html.escape(text)
    text = _CODE.sub(r"<code>\1</code>", text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    text = _LINK.sub(r'<a href="\2">\1</a>', text)
    return _BARE_URL.sub(_linkify, text)


def markdown_to_email_html(text: str) -> str:
    """Convert the Markdown subset the agent emits into inline-styled email HTML."""
    if not text.strip():
        return ""
    blocks: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []
    list_tag = ""

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f'<p style="{_PARAGRAPH_STYLE}">{_inline(" ".join(paragraph))}</p>')
            paragraph.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if items:
            rendered = "".join(f'<li style="{_ITEM_STYLE}">{item}</li>' for item in items)
            blocks.append(f'<{list_tag} style="{_LIST_STYLE}">{rendered}</{list_tag}>')
            items.clear()
        list_tag = ""

    def start_list(tag: str) -> None:
        nonlocal list_tag
        if list_tag != tag:
            flush_list()
            list_tag = tag

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue
        if heading := _HEADING.match(stripped):
            flush_paragraph()
            flush_list()
            # h1 stays reserved for the message subject line.
            tag = min(len(heading.group(1)) + 1, 6)
            blocks.append(f'<h{tag} style="{_HEADING_STYLE[tag]}">{_inline(heading.group(2))}</h{tag}>')
            continue
        if bullet := _BULLET.match(stripped):
            flush_paragraph()
            start_list("ul")
            items.append(_inline(bullet.group(1)))
            continue
        if numbered := _NUMBERED.match(stripped):
            flush_paragraph()
            start_list("ol")
            items.append(_inline(numbered.group(1)))
            continue
        flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    return "\n".join(blocks)
