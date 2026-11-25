"""Approval-prompt display sanitizer: make hidden content in a command string visible.

Re-exported by ``agent.redact``; moved out of it unchanged.
"""

from __future__ import annotations

import re
import unicodedata

# Approval-prompt display integrity: the copy a human approves must show everything that will run.
# Code points that render as nothing or as a line break become visible escapes: every control,
# format, private-use, unassigned and surrogate code point (category C*, except tab/newline), line
# and paragraph separators (Zl/Zp), and blank-rendering marks/letters in other categories (variation
# selectors incl. U+E0100-E01EF, CGJ, Khmer/Mongolian selectors, Hangul fillers, Braille blank).
# Ordinary non-ASCII text is untouched. Padding that pushes the tail off screen or past a preview
# cut (space runs, whitespace-only lines) collapses to a marker; real indentation is kept.
_APPROVAL_NON_ASCII_RE = re.compile(r"[^\x20-\x7e\t\n]+")
_APPROVAL_BLANK_GLYPHS = frozenset(
    "\u034f\u115f\u1160\u17b4\u17b5\u180b\u180c\u180d\u180f\u2800\u3164\uffa0"
    + "".join(map(chr, range(0xFE00, 0xFE10))) + "".join(map(chr, range(0xE0100, 0xE01F0))))
_APPROVAL_SPACE = r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
# Mid-line runs (after visible text) of 21+, or line-leading runs past any sane indentation.
_APPROVAL_PAD_RE = re.compile(rf"(?<=\S){_APPROVAL_SPACE}{{21,}}|(?m:^){_APPROVAL_SPACE}{{65,}}")
# 3+ consecutive blank or whitespace-only lines; the match ends at a newline so the next line keeps its indent.
_APPROVAL_BLANK_RE = re.compile(rf"\n(?:{_APPROVAL_SPACE}*\n){{3,}}")
_APPROVAL_SAME_CHAR_RUN_RE = re.compile(r"(.)\1*", re.DOTALL)
# Prose mode (questions shown to a human, never executed) surfaces only what can hide or reorder
# text: controls, bidi embeddings/overrides/isolates, line/paragraph separators and padding. ZWJ,
# ZWNJ and variation selectors stay, because emoji and scripts such as Persian need them.
_PROSE_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _escape_approval_run(run: str, prose: bool = False) -> str:
    """Escape the invisible code points of a non-ASCII run; 4+ repeats of one show once with their count."""
    def one(m: re.Match) -> str:
        ch, n = m.group(1), len(m.group())
        cat = unicodedata.category(ch)
        if prose:
            hidden = cat == "Cc" or cat in ("Zl", "Zp") or ch in _PROSE_BIDI_CONTROLS
        else:
            hidden = ch in _APPROVAL_BLANK_GLYPHS or cat[0] == "C" or cat in ("Zl", "Zp")
        if not hidden:
            return m.group()
        esc = ch.encode("unicode_escape").decode("ascii")
        return f"⟨{esc} ×{n}⟩" if n >= 4 else esc * n
    return _APPROVAL_SAME_CHAR_RUN_RE.sub(one, run)


def sanitize_command_for_display(text: str | None, *, prose: bool = False) -> str:
    """Make hidden content in an approval-prompt string visible; clean text comes back unchanged.

    Display-only: the executed command, smart assessment and allowlist keys never see this copy.
    ``prose=True`` is the lighter pass for question text (see ``_PROSE_BIDI_CONTROLS``)."""
    if not text:
        return ""
    out = _APPROVAL_NON_ASCII_RE.sub(lambda m: _escape_approval_run(m.group(), prose), text)
    out = _APPROVAL_BLANK_RE.sub(lambda m: f"\n⟨+{m.group().count(chr(10)) - 1} blank lines⟩\n", out)
    return _APPROVAL_PAD_RE.sub(lambda m: f" ⟨+{len(m.group())} whitespace chars⟩ ", out)
