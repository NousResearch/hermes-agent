"""Normalization for Telegram's permissive Rich Markdown block parser."""

import re
from bisect import bisect_left, bisect_right

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
# Code wins over math: ``$$`` opens math only when its closing ``$$`` comes before the next
# backtick, blank line or fence opener (a shell answer's ``PID is $$`` must not swallow the fence
# below it and end at ``echo $$`` inside the code); otherwise it is literal text.
# A fence closes as CommonMark closes it: on a line of its own holding the opener's character at
# least as many times and then only whitespace (so ``text ``` trailing`` inside it is code), or
# where the quote/list item that holds it ends. An inline span is a backtick run (an escaped first
# tick is literal) closed by the next run of exactly its length inside the same paragraph; a run
# with no such closer is literal text. A regex ``(`+).*?\1`` rescans to the end of the text for
# every unclosed run (6.5 s on a 32 KB draft of distinct-length runs), so the regex only spots a
# backtick and the closer comes from an index of runs by length, and a fence body is walked once:
# the whole scan stays linear.
_BLOCK_PREFIX = r"[ \t]*(?:(?:>[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))*"
_FENCE_OPEN = r"(?P<fence>`{3,}(?=[^`\n]*$)|~{3,})"
_RICH_LITERAL_REGION = (
    r"^(?P<fence_prefix>" + _BLOCK_PREFIX + r")" + _FENCE_OPEN + r"[^\n]*"
    r"|(?P<tick>`)"
    r"|(?P<math>\$\$)"
)
_RICH_LITERAL_REGION_RE = re.compile(_RICH_LITERAL_REGION, re.MULTILINE | re.DOTALL)
_LITERAL_HASH_RE = re.compile(
    _RICH_LITERAL_REGION + r"|(?P<prefix>^" + _BLOCK_PREFIX + r")(?P<hashes>#+)(?=[^\s#])",
    re.MULTILINE | re.DOTALL,
)
_BACKTICK_RUN_RE = re.compile(r"`+")
_MATH_DELIM_RE = re.compile(r"\$\$")
# Line ends display math cannot cross: a blank line (also inside a quote) or a tilde fence opener
# (a backtick fence already stops it at its first backtick).
_MATH_BREAK_RE = re.compile(
    r"\n(?=[ \t]*(?:>[ \t]*)*(?:\n|$)| {0,3}(?:(?:>[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))*~{3,})", re.MULTILINE)
# Line ends a code span cannot cross: a blank line, a one-line ATX heading, a list item or a fence
# opener ends the paragraph first (an ordered item interrupts only at 1).
_PARAGRAPH_BREAK_RE = re.compile(
    r"\n(?=[ \t]*\n| {0,3}#{1,6}(?:[ \t]|$)| {0,3}(?:>[ \t]*)*(?:[-+*]|1[.)])[ \t]+\S"
    r"| {0,3}(?:(?:>[ \t]*|[-+*][ \t]+|1[.)][ \t]+))*(?:`{3,}[^`\n]*$|~{3,}))"
    r"|(?<=\n) {0,3}#{1,6}(?:[ \t][^\n]*)?(?=\n)|\A {0,3}#{1,6}(?:[ \t][^\n]*)?(?=\n)", re.MULTILINE)
_CONTAINER_MARKER_RE = re.compile(r"[ \t]*(?:(?P<quote>>)[ \t]?|(?:[-+*]|\d+[.)])[ \t]+)")
_QUOTE_MARKER_RE = re.compile(r" {0,3}> ?")
_LIST_ITEM_RE = re.compile(r" *(?:[-+*]|\d{1,9}[.)])(?:[ \t]+(?=\S)|[ \t]*$)")  # an empty item's content sits one column past its marker
_FENCE_CLOSE_RE = re.compile(r"( *)(`{3,}|~{3,})[ \t]*")


def _container_markers(prefix: str) -> tuple[list[int], int]:
    """Containers a fence's line prefix opens (-1 = a quote marker each line repeats; n = a list
    item's content indent) and the opener's own indent past them."""
    containers: list[int] = []
    pos = 0
    while (marker := _CONTAINER_MARKER_RE.match(prefix, pos)) and marker.end() > pos:
        containers.append(-1 if marker.group("quote") else len(marker.group(0).expandtabs(4)))
        pos = marker.end()
    return containers, len(prefix[pos:].expandtabs(4))


def _previous_line(text: str, pos: int) -> str | None:
    """The last non-blank line before the line starting at *pos* (``None`` at the top)."""
    end = pos - 1
    while end > 0:
        start = text.rfind("\n", 0, end) + 1
        if text[start:end].strip():
            return text[start:end].expandtabs(4)
        end = start - 1
    return None


def _fence_layout(text: str, match: re.Match[str]) -> tuple[list[int], int] | None:
    """Containers holding the fence *match* opens and the most a closer may be indented past
    them, or ``None`` when the line is not a fence. An indented opener with no markers of its
    own is the content of the list item on the line above (it closes where that item ends); one
    indented 4+ columns past every container is paragraph text or an indented code block."""
    containers, indent = _container_markers(match.group("fence_prefix"))
    if containers or not indent:
        return (containers, 3) if indent < 4 else None
    line = _previous_line(text, match.start())
    item = _LIST_ITEM_RE.match(line) if line is not None else None
    width = (len(item.group(0)) if item.end() < len(line) else len(item.group(0).rstrip()) + 1) if item else 0
    above = text[text.rfind("\n", 0, max(match.start() - 1, 0)) + 1:max(match.start() - 1, 0)]
    if item and item.end() == len(line) and not above.strip():
        item = None  # an empty item followed by a blank line has ended: the fence is top-level
    if item and width <= indent:
        indent -= width
        containers = [width]
    elif line is not None and indent >= 4 and len(line) - len(line.lstrip(" ")) >= 2:
        return [], indent  # deeper continuation of a list item: closes at its own indent
    return (containers, 3) if indent < 4 else None


def _fence_end(text: str, match: re.Match[str], layout: tuple[list[int], int]) -> int:
    """End of the fenced block *match* opens: its closing line, the end of the quote/list item
    holding it, or the end of an unfinished streaming draft."""
    fence = match.group("fence")
    containers, max_indent = layout
    line_end = match.end()
    while line_end < len(text):
        line_start = line_end + 1
        line_end = text.find("\n", line_start)
        line_end = len(text) if line_end < 0 else line_end
        rest = text[line_start:line_end]
        for need in containers:
            if need < 0:
                quote = _QUOTE_MARKER_RE.match(rest)
                if quote is None:
                    return line_start - 1
                rest = rest[quote.end():]
            elif rest.strip():
                rest = rest.expandtabs(4)
                if len(rest) - len(rest.lstrip(" ")) < need:
                    return line_start - 1
                rest = rest[need:]
        close = _FENCE_CLOSE_RE.fullmatch(rest.expandtabs(4))
        if (close and close.group(2)[0] == fence[0] and len(close.group(2)) >= len(fence)
                and len(close.group(1)) <= max_indent):
            return line_end
    return len(text)


def _math_end(after: int, delims: list[int], breaks: list[int], ticks: list[int], size: int) -> int:
    """End of the display math whose ``$$`` ends at *after*, or *after* when that ``$$`` is literal:
    its closer must precede the next backtick and blank line/fence opener. With neither a closer
    nor anything after it (a streaming draft mid-formula) the math runs to the end."""
    i, b, t = bisect_left(delims, after), bisect_left(breaks, after), bisect_left(ticks, after)
    limit = min(breaks[b] if b < len(breaks) else size, ticks[t] if t < len(ticks) else size)
    if i < len(delims) and delims[i] + 2 <= limit:
        return delims[i] + 2
    return size if i == len(delims) and limit == size else after


def _iter_literal_tokens(pattern: re.Pattern[str], text: str):
    """Yield ``(start, end, match)`` for each *pattern* token in order; ``match`` is ``None`` for
    an inline code span."""
    run_starts: list[int] = []
    run_ends: list[int] = []
    starts_by_len: dict[int, list[int]] = {}
    for run in _BACKTICK_RUN_RE.finditer(text):
        run_starts.append(run.start())
        run_ends.append(run.end())
        starts_by_len.setdefault(run.end() - run.start(), []).append(run.start())
    breaks = [brk.end() if brk.group(0) else brk.start() for brk in _PARAGRAPH_BREAK_RE.finditer(text)]
    math_delims = [d.start() for d in _MATH_DELIM_RE.finditer(text)]
    math_breaks = [brk.start() for brk in _MATH_BREAK_RE.finditer(text)]
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        if m.group("fence") is not None:
            layout = _fence_layout(text, m)
            if layout is None:
                pos = m.start("fence")  # paragraph or indented-code line: its backticks are inline runs
                continue
            pos = _fence_end(text, m, layout)
            yield m.start(), pos, m
            continue
        if m.group("math") is not None:
            pos = _math_end(m.end(), math_delims, math_breaks, run_starts, len(text))
            if pos > m.end():
                yield m.start(), pos, m
            continue
        if m.group("tick") is None:
            yield m.start(), m.end(), m
            pos = m.end()
            continue
        start = m.start()
        end = run_ends[bisect_right(run_starts, start) - 1]
        slash = start
        while slash and text[slash - 1] == "\\":
            slash -= 1
        start += (start - slash) % 2  # an escaped first tick is literal: ``\``x`` is ` then a span
        same_len = starts_by_len.get(end - start, [])
        i = bisect_right(same_len, end)
        b = bisect_left(breaks, end)
        if start < end and i < len(same_len) and (b == len(breaks) or same_len[i] < breaks[b]):
            pos = same_len[i] + end - start
            yield start, pos, None
        else:
            pos = end


def escape_literal_hash_prefixes(text: str) -> str:
    """Keep ``#89``/``#tag`` as prose, including in lists and blockquotes.

    Telegram accepts these as headings even without the whitespace required by
    standard Markdown. Escape only such block-start hashes, not real headings,
    inline references, links, code, or already escaped prefixes.
    """
    out: list[str] = []
    last = 0
    for start, end, match in _iter_literal_tokens(_LITERAL_HASH_RE, text):
        if match is not None and match.group("hashes") is not None:
            out.append(text[last:start] + match.group("prefix") + r"\#" * len(match.group("hashes")))
            last = end
    return "".join(out) + text[last:]


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
_RICH_THEMATIC_BREAK_RE = re.compile(r"(?:[-*_][ \t]*){3,}")


def _rich_separate_ordered_lists(text: str) -> str:
    """Insert one blank line before an ordered list that directly follows a paragraph.

    ``**Label**\\n7. item`` parses as one paragraph (the numbers cannot interrupt it), so the
    rich path renders the items as plain text. A blank line restores the list block; list and
    blockquote continuations, indented lines, fences (closed, or unclosed in a streaming draft)
    and lists starting at ``1`` are left untouched, and the pass is idempotent.
    """
    # Code spans, math and fences come from the shared literal-region scan, so a fence closes only
    # where the hash pass and CommonMark close it; a line starting inside one is never a list item.
    spans = [(start, end, bool(m and m.group("fence")))
             for start, end, m in _iter_literal_tokens(_RICH_LITERAL_REGION_RE, text)]
    out: list[str] = []
    state = None  # None, "paragraph", or "container": the block the previous line built
    offset = span_idx = 0
    for line in text.split("\n"):
        start, offset = offset, offset + len(line) + 1
        while span_idx < len(spans) and spans[span_idx][1] <= start:
            span_idx += 1
        if span_idx < len(spans) and spans[span_idx][0] < start:
            if spans[span_idx][2]:
                state = None  # inside a fence; the block after it starts fresh
            # else: continuation inside a multiline code span or math block keeps the block
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
