"""Markdown -> Matrix HTML text passes: TeX sentinel tokens and the CommonMark gaps in
Python-Markdown (``#123`` is not a heading; a list may follow a label line directly)."""

from __future__ import annotations

import re
from html import escape as _html_escape
from typing import Any


# --- LaTeX math ($...$, $$...$$) -> Element data-mx-maths markup ---
# Element (feature_latex_maths) typesets <div|span data-mx-maths="TEX"> at display time.
# Our sanitizer allowlists tags/attrs, so data-mx-maths cannot pass through HTML
# sanitization directly. Instead, math is swapped for opaque sentinel tokens before
# Markdown conversion (protecting TeX from escaping) and expanded back to math
# markup after sanitization. Tokens are plain printable text with no special
# HTML/Markdown meaning, so both the Markdown converter and the sanitizer
# pass them through verbatim.
_TEX_TOKEN_RE = re.compile(r"HERMESTEX(?:DISPLAY|INLINE)(\d+)HERMESTEXEND")
_TEX_DISPLAY_TOKEN = "HERMESTEXDISPLAY%dHERMESTEXEND"
_TEX_INLINE_TOKEN = "HERMESTEXINLINE%dHERMESTEXEND"


def _latex_to_tokens(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace ``$$...$$``/``$...$`` with sentinel tokens.

    Returns the tokenized text plus an ordered ``(tag, tex)`` store, where tag
    is ``div`` for display math and ``span`` for inline math. Dollars that do
    not form a pair (prices, literals) are left untouched.
    """
    if not text or "$" not in text:
        return text, []
    store: list[tuple[str, str]] = []

    def _sub_display(match: re.Match[str]) -> str:
        store.append(("div", match.group(1).strip()))
        return _TEX_DISPLAY_TOKEN % (len(store) - 1)

    def _sub_inline(match: re.Match[str]) -> str:
        store.append(("span", match.group(1).strip()))
        return _TEX_INLINE_TOKEN % (len(store) - 1)

    text = re.sub(r"\$\$([^\n$]+?)\$\$", _sub_display, text)
    text = re.sub(r"(?<![\\$\w])\$([^\n$]+?)\$(?!\w)", _sub_inline, text)
    return text, store


def _tokens_to_mx_maths(html: str, store: list[tuple[str, str]]) -> str:
    """Expand sentinel tokens into ``data-mx-maths`` markup (TeX HTML-escaped)."""

    def _expand(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= len(store):
            # Not one of our tokens (user-typed text that collides with the
            # sentinel format) — leave it verbatim.
            return match.group(0)
        tag, tex = store[idx]
        escaped = _html_escape(tex, quote=True)
        return f'<{tag} data-mx-maths="{escaped}">{escaped}</{tag}>'

    return _TEX_TOKEN_RE.sub(_expand, html)


_MD_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_MD_LIST_ITEM = re.compile(r"^ {0,3}(?:[-*+]|1[.)])[ \t]+\S")
# Python-Markdown's stock ATX pattern takes "#310 are tests" as an <h1> (no space after the
# hashes). CommonMark and Matrix clients require the space, and agent replies routinely start a
# line with an issue/task id, so require it.
_MD_ATX_HEADING = re.compile(r"(?:^|\n)(?P<level>#{1,6})(?=[ \t]|\n|$)(?P<header>(?:\\.|[^\\])*?)#*(?:\n|$)")


def _list_interrupts_paragraph(text: str) -> str:
    """Insert the blank line Python-Markdown needs before a list that directly follows a text
    line. CommonMark lets a list interrupt a paragraph, so ``**Today**\\n- item`` renders as a
    label plus a list everywhere else; without this it became one paragraph of literal dashes."""
    out: list[str] = []
    fence = None
    prev_text = in_list = False
    for line in text.split("\n"):
        opener = _MD_FENCE.match(line)
        if fence:
            # A closing fence is the same character, at least as long, and nothing but
            # whitespace after it (CommonMark); "```python" inside a block is content.
            if (opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence)
                    and not line[opener.end():].strip()):
                fence = None
            out.append(line)
            continue
        if opener:
            fence, prev_text, in_list = opener.group(1), False, False
        elif not line.strip():
            prev_text = in_list = False
        elif _MD_LIST_ITEM.match(line):
            if prev_text and not in_list:
                out.append("")
            prev_text = in_list = True
        else:
            prev_text = True
        out.append(line)
    return "\n".join(out)


def _matrix_markdown(md_module: Any, extensions: list[str]) -> Any:
    """A ``markdown.Markdown`` whose ATX headings need a space after the hashes."""
    md = md_module.Markdown(extensions=extensions)
    if "hashheader" in md.parser.blockprocessors:
        md.parser.blockprocessors["hashheader"].RE = _MD_ATX_HEADING
    return md
