"""Render markdown as styled plain text for chat surfaces with no markdown renderer (iMessage, SMS).

``strip_markdown`` deletes formatting, so a reply's emphasis, lists and structure flatten into one
run of text. ``render_plain_text`` keeps that structure readable with three opt-in styles that
combine freely:

``caps``     headings upper-case; bold too when ``unicode`` is off
``unicode``  bold / italic / bold-italic as Mathematical Sans-Serif glyphs, strikethrough as a
             combining stroke, headings as bold glyphs
``emoji``    task boxes ⬜/✅, 🔹 bullets, keycap numbers 1️⃣-🔟, 📌 headings, ➖ rules

With no style (``plain``) the base renderer still turns bullets into •, tables into bullet groups,
rules into a line and quotes into a bar. Code spans, fenced code and URLs are never restyled, so
commands stay copyable and links stay tappable.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

PLAIN_TEXT_STYLES = ("caps", "unicode", "emoji")
_LEGACY = {"", "strip", "none", "off", "false"}
_ALL = {"all", "rich"}


def _glyph_table(upper: int, lower: int, digit: Optional[int] = None) -> dict[int, str]:
    table = {0x41 + i: chr(upper + i) for i in range(26)}
    table.update({0x61 + i: chr(lower + i) for i in range(26)})
    if digit is not None:
        table.update({0x30 + i: chr(digit + i) for i in range(10)})
    return table


_BOLD = _glyph_table(0x1D5D4, 0x1D5EE, 0x1D7EC)
_ITALIC = _glyph_table(0x1D608, 0x1D622)
_BOLD_ITALIC = _glyph_table(0x1D63C, 0x1D656, 0x1D7EC)
_STROKE = "̶"
_KEYCAPS = ("0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟")


def parse_plain_text_style(value: Any) -> Optional[frozenset[str]]:
    """``None`` for the legacy strip (unset, ``strip``); otherwise the chosen styles, which may be
    empty (``plain``). Accepts ``caps,unicode``, ``caps+emoji``, ``all``; unknown words are ignored."""
    raw = str(value or "").strip().lower()
    if raw in _LEGACY:
        return None
    if raw in _ALL:
        return frozenset(PLAIN_TEXT_STYLES)
    words = set(re.split(r"[\s,+|]+", raw))
    return frozenset(w for w in words if w in PLAIN_TEXT_STYLES)


class _Vault:
    """Swaps spans that must stay literal (code, links, URLs) for single private-use characters,
    so no emphasis rule, ``str.upper`` or glyph map can touch them, then puts them back."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def put(self, literal: str) -> str:
        self._items.append(literal)
        return chr(0xF0000 + len(self._items) - 1)

    def restore(self, text: str) -> str:
        # Twice: a link label kept in the vault can itself hold a vaulted code span.
        for _ in range(2):
            text = re.sub("[\U000F0000-\U000FFFFD]", lambda m: self._items[ord(m.group()) - 0xF0000], text)
        return text


_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"(!?)\[([^\]\n]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)


def _vault_literals(text: str, vault: _Vault, styles: frozenset[str], keep_link_targets: bool) -> str:
    text = _FENCE_RE.sub(lambda m: vault.put(m.group(1).rstrip("\n")), text)
    text = _CODE_RE.sub(lambda m: vault.put(m.group(1)), text)

    def link(m: re.Match[str]) -> str:
        label = _render_inline(m.group(2), styles)
        target = m.group(3)
        if not (keep_link_targets and target.lower().startswith(("http://", "https://"))):
            return vault.put(label)
        return vault.put(target if m.group(2).strip() == target else f"{label}\n{target}")

    text = _LINK_RE.sub(link, text)
    return _URL_RE.sub(lambda m: vault.put(m.group()), text)


def _stylers(styles: frozenset[str]) -> dict[str, Callable[[str], str]]:
    unicode, caps = "unicode" in styles, "caps" in styles
    plain = str
    bold = (lambda s: s.translate(_BOLD)) if unicode else (str.upper if caps else plain)
    return {
        "bold": bold,
        "italic": (lambda s: s.translate(_ITALIC)) if unicode else plain,
        "bold_italic": (lambda s: s.translate(_BOLD_ITALIC)) if unicode else bold,
        "strike": (lambda s: "".join(c + _STROKE if not c.isspace() else c for c in s)) if unicode else plain,
        "heading": (lambda s: bold(s.upper() if caps else s)) if unicode else (str.upper if caps else plain),
    }


_INLINE_RULES = (
    ("bold_italic", re.compile(r"\*\*\*(?!\s)([^*\n]+?)(?<!\s)\*\*\*")),
    ("bold", re.compile(r"\*\*(?!\s)([^*\n]+?)(?<!\s)\*\*")),
    ("bold", re.compile(r"(?<!\w)__(?!\s)([^_\n]+?)(?<!\s)__(?!\w)")),
    ("italic", re.compile(r"(?<![\w*])\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?![\w*])")),
    ("italic", re.compile(r"(?<!\w)_(?![\s_])([^_\n]+?)(?<![\s_])_(?!\w)")),
    ("strike", re.compile(r"~~(?!\s)([^~\n]+?)(?<!\s)~~")),
)


def _render_inline(text: str, styles: frozenset[str]) -> str:
    stylers = _stylers(styles)
    for kind, pattern in _INLINE_RULES:
        text = pattern.sub(lambda m, f=stylers[kind]: f(m.group(1)), text)
    return text


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TASK_RE = re.compile(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+•]\s+(.*)$")  # • is what convert_table_to_bullets emits
_NUMBERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def _render_line(line: str, styles: frozenset[str]) -> str:
    emoji = "emoji" in styles
    if _RULE_RE.match(line):
        return "➖➖➖" if emoji else "──────────"
    if m := _HEADING_RE.match(line):
        heading = _stylers(styles)["heading"](_render_inline(m.group(2), frozenset()))
        return f"📌 {heading}" if emoji and len(m.group(1)) <= 2 else heading
    if m := _TASK_RE.match(line):
        done = m.group(2) != " "
        box = ("✅" if done else "⬜") if emoji else ("☑" if done else "☐")
        return f"{m.group(1)}{box} {_render_inline(m.group(3), styles)}"
    if m := _BULLET_RE.match(line):
        mark = ("🔹" if emoji else "•") if not m.group(1) else "◦"
        return f"{m.group(1)}{mark} {_render_inline(m.group(2), styles)}"
    if m := _NUMBERED_RE.match(line):
        n = int(m.group(2))
        number = _KEYCAPS[n] if emoji and n < len(_KEYCAPS) else f"{n}."
        return f"{m.group(1)}{number} {_render_inline(m.group(3), styles)}"
    if m := _QUOTE_RE.match(line):
        return f"│ {_render_inline(m.group(1), styles)}"
    return _render_inline(line, styles)


def render_plain_text(text: str, styles: frozenset[str] = frozenset(), *, keep_link_targets: bool = False) -> str:
    r"""Markdown -> styled plain text. ``styles`` is any subset of ``PLAIN_TEXT_STYLES``;
    ``keep_link_targets`` keeps ``[label](https://url)`` as ``label\nurl`` for surfaces that
    auto-link bare URLs (iMessage)."""
    from gateway.platforms.helpers import convert_table_to_bullets

    vault = _Vault()
    text = convert_table_to_bullets(text or "")
    text = _vault_literals(text, vault, styles, keep_link_targets)
    text = "\n".join(_render_line(line, styles) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", vault.restore(text)).strip()
