"""Normalization for Telegram's permissive Rich Markdown block parser."""

import re

# An email-shaped token (incl. provider-prefixed ``openai:user@example.com``): Telegram's rich entity
# detection can auto-link it into an invalid email entity and reject the whole message
# (RICH_MESSAGE_EMAIL_INVALID), so the payload builder turns detection off when one is present.
# The lookbehind starts a match only at a token boundary, so a long run with no valid ``@domain``
# is scanned once instead of once per start offset (quadratic on a 40k-char draft).
RICH_EMAIL_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9.!#$%&'*+/=?^_`{|}~-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9]"
    r"(?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+",
    re.IGNORECASE,
)


# Literal code/math regions: a block fence (opened only at a line start, after any quote/list
# prefixes, and unfinished in a streaming draft), an inline code span (which also owns inline
# ```x```), or display math. Both transforms below consume these first so their contents survive.
_BLOCK_PREFIX = r"[ \t]*(?:(?:>[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))*"
_RICH_LITERAL_REGION = (
    r"^" + _BLOCK_PREFIX + r"(?P<fence>`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n.*?(?:(?P=fence)(?![`~])|\Z)"
    r"|(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`)"
    r"|\$\$.*?(?:\$\$|\Z)"
)
_RICH_LITERAL_REGION_RE = re.compile(_RICH_LITERAL_REGION, re.MULTILINE | re.DOTALL)
_LITERAL_HASH_RE = re.compile(
    _RICH_LITERAL_REGION + r"|(?P<prefix>^" + _BLOCK_PREFIX + r")(?P<hashes>#+)(?=[^\s#])",
    re.MULTILINE | re.DOTALL,
)


def escape_literal_hash_prefixes(text: str) -> str:
    """Keep ``#89``/``#tag`` as prose, including in lists and blockquotes.

    Telegram accepts these as headings even without the whitespace required by
    standard Markdown. Escape only such block-start hashes, not real headings,
    inline references, links, code, or already escaped prefixes.
    """
    def replace(match: re.Match[str]) -> str:
        hashes = match.group("hashes")
        if hashes is None:
            return match.group(0)
        return match.group("prefix") + r"\#" * len(hashes)

    return _LITERAL_HASH_RE.sub(replace, text)


# Rich-message regions whose internal newlines must stay bare (Telegram renders them natively):
# fenced code blocks OR GFM pipe-table blocks (header row, delimiter row, data rows).
_RICH_PROTECTED_REGION_RE = re.compile(
    r'(?:```[^\n]*\n[\s\S]*?```)'                       # fenced code block
    r'|(?:^[^\n]*\|[^\n]*\n'                            # table header row (has a pipe)
    r'[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*'  # delimiter
    r'(?:\n[^\n]*\|[^\n]*)*)',                          # data rows (newline-led, trailing \n left for prose)
    re.MULTILINE)


# CommonMark lets an ordered list interrupt a paragraph only when it starts at 1, so a numbered
# list that begins above 1 right after prose ("**Label**\n7. item") would otherwise stay glued to
# the paragraph and reach Telegram as literal text with no list layout (#124552).
_RICH_INTERRUPTING_ORDERED_ITEM_RE = re.compile(r" {0,3}(?!1[.)])\d{1,9}[.)][ \t]+\S")
_RICH_CONTAINER_LINE_RE = re.compile(r" {0,3}(?:>|[-+*][ \t]|\d{1,9}[.)][ \t])")
_RICH_FENCE_LINE_RE = re.compile(r" {0,3}(?:```|~~~)")
_RICH_THEMATIC_BREAK_RE = re.compile(r"(?:[-*_][ \t]*){3,}")


def _rich_separate_ordered_lists(text: str) -> str:
    """Insert one blank line before an ordered list that directly follows a paragraph.

    ``**Label**\\n7. item`` parses as one paragraph (the numbers cannot interrupt it), so the
    rich path renders the items as plain text. A blank line restores the list block; list and
    blockquote continuations, indented lines, fences (closed, or unclosed in a streaming draft)
    and lists starting at ``1`` are left untouched, and the pass is idempotent.
    """
    # Inline code / display math spanning lines is literal text; block fences keep the line scan below.
    # A backtick span cannot cross a blank line (CommonMark ends the paragraph first).
    spans = [m.span() for m in _RICH_LITERAL_REGION_RE.finditer(text)
             if not m.group("fence") and not (m.group("ticks") and "\n\n" in m.group(0))]
    out: list[str] = []
    state = None  # None, "paragraph", "container", or "fence" — the block the previous line built
    offset = span_idx = 0
    for line in text.split("\n"):
        start, offset = offset, offset + len(line) + 1
        while span_idx < len(spans) and spans[span_idx][1] <= start:
            span_idx += 1
        if span_idx < len(spans) and spans[span_idx][0] < start:
            pass  # continuation inside a multiline code span or math block: never a list item
        elif _RICH_FENCE_LINE_RE.match(line):
            state = None if state == "fence" else "fence"
        elif state == "fence":
            pass
        elif not line.strip():
            state = None
        elif state == "paragraph" and _RICH_INTERRUPTING_ORDERED_ITEM_RE.match(line):
            out.append("")
            state = "container"
        elif _RICH_CONTAINER_LINE_RE.match(line):
            state = "container"
        elif line[:1].isspace():
            pass  # indented continuation keeps the current block
        elif not _RICH_THEMATIC_BREAK_RE.fullmatch(line.strip()):
            state = state or "paragraph"
        else:
            state = None
        out.append(line)
    return "\n".join(out)


def rich_normalize_linebreaks(text: str) -> str:
    """Convert lone ``\\n`` (a Markdown soft break) to hard breaks for sendRichMessage; ``\\n\\n``,
    fenced code and pipe tables are left untouched."""
    if not text or '\n' not in text:
        return text
    out: list[str] = []
    pos = 0
    for m in _RICH_PROTECTED_REGION_RE.finditer(text):
        out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', _rich_separate_ordered_lists(text[pos:m.start()])))
        out.append(m.group(0))  # protected region kept verbatim
        pos = m.end()
    out.append(re.sub(r'(?<!\n)\n(?!\n)', '  \n', _rich_separate_ordered_lists(text[pos:])))
    return ''.join(out)
