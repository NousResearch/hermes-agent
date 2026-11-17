"""Code regions of a rich-message markdown text, found in one linear pass.

Text inside code is shown verbatim (CommonMark backslash escapes do not apply there), so a pass
that rewrites prose (the currency ``$`` escape, unrenderable-link degrading) must skip it. A
backreference regex (``(`+).*?\\1``) rescans to the end of the line from every backtick run that
has no same-width closer, which made a 32 KB line of mismatched runs block the gateway event loop
for seconds (#151725), and it only knew closed ``` fences and single-line spans.
"""

import re
from bisect import bisect_left

# Quote and list markers a fence may sit behind ("> ```", "- ```", "1. ```").
_BLOCK_PREFIX = r"[ \t]*(?:(?:>[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))*"
_FENCE_OPEN_RE = re.compile(_BLOCK_PREFIX + r"(?P<fence>`{3,}(?![^`\n]*`)|~{3,})")
# Models also open a fence after lead-in prose ("Here is the code: ```bash"); the MarkdownV2 path
# reads that as a fence too, so its body must not be rewritten here either. Only that shape counts
# (the run ends the line, at most an info word after it) and only when a fence line closes it: a
# fence opens at a line start (CommonMark), so a mid-line run in prose ("Use ``` to open a fence.
# Cost is $5") is literal text, not code to the end of the reply.
_LEADIN_FENCE_RE = re.compile(r"(?:(?!```)[^\n])*(?P<fence>`{3,})[^`\s]*[ \t]*\r?$")
_FENCE_CLOSE_RE = re.compile(_BLOCK_PREFIX + r"(?P<fence>`{3,}|~{3,})[ \t]*\r?$")
_LIST_ITEM_RE = re.compile(r"[ \t]*(?:[-+*]|\d+[.)])[ \t]")
_INDENTED_RE = re.compile(r"(?: {4}| {0,3}\t)")
_BACKTICK_RUN_RE = re.compile(r"`+")
_HTML_CODE_OPEN_RE = re.compile(r"<(pre|code)\b[^<>\n]*>", re.IGNORECASE)
_HTML_CODE_CLOSE_RE = re.compile(r"</(pre|code)[ \t]*>", re.IGNORECASE)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*(?=\n)")


def _line_spans(text: str):
    """``(start, end)`` of every line, ``end`` excluding its newline."""
    start = 0
    for m in re.finditer("\n", text):
        yield start, m.start()
        start = m.end()
    yield start, len(text)


def _widest_backtick_closer_after(lines: list, text: str) -> list:
    """For each line index, the widest ``` closing-fence line after it (0 when none)."""
    widest, out = 0, [0] * len(lines)
    for i in range(len(lines) - 1, -1, -1):
        out[i] = widest
        close = _FENCE_CLOSE_RE.match(text, lines[i][0], lines[i][1])
        if close and close["fence"][0] == "`":
            widest = max(widest, len(close["fence"]))
    return out


def _block_regions(text: str) -> list:
    """Fenced blocks (``` or ~~~, closed by a fence of the same char at least as long, or open to
    the end of the text as in a streamed draft; a lead-in fence only when closed) and indented
    code blocks."""
    regions: list = []
    fence = None  # (char, width, region start) of the open fence
    indented_start = None
    prev_blank, in_list = True, False
    lines = list(_line_spans(text))
    closer_after = None  # computed on the first lead-in fence only
    for i, (start, end) in enumerate(lines):
        line = text[start:end]
        if fence is not None:
            close = _FENCE_CLOSE_RE.match(line)
            if close and close["fence"][0] == fence[0] and len(close["fence"]) >= fence[1]:
                regions.append((fence[2], end))
                fence, prev_blank = None, False
            continue
        blank = not line.strip()
        if indented_start is not None and not blank and not _INDENTED_RE.match(line):
            regions.append((indented_start, start - 1))
            indented_start = None
        opened = _FENCE_OPEN_RE.match(line)
        if not opened and (leadin := _LEADIN_FENCE_RE.match(line)):
            if closer_after is None:
                closer_after = _widest_backtick_closer_after(lines, text)
            opened = leadin if closer_after[i] >= len(leadin["fence"]) else None
        if indented_start is None and opened:
            fence = (opened["fence"][0], len(opened["fence"]), start + opened.start("fence"))
            continue
        if indented_start is None and not blank and prev_blank and not in_list and _INDENTED_RE.match(line):
            indented_start = start  # an indented line can't interrupt a paragraph or a list item
        elif not blank and indented_start is None:
            in_list = bool(_LIST_ITEM_RE.match(line)) or (in_list and line[:1] in " \t")
        prev_blank = blank
    if fence is not None:
        regions.append((fence[2], len(text)))
    elif indented_start is not None:
        regions.append((indented_start, len(text)))
    return regions


def _inline_regions(text: str, lo: int, hi: int, html: bool) -> list:
    """Backtick spans and ``<pre>``/``<code>`` elements in ``text[lo:hi]``, left to right.

    A backtick run closes at the next run of exactly its width before a blank line (CommonMark);
    an unclosed run is literal and the scan goes on after it. Each run is indexed once by width,
    so pairing is a bisect, never a rescan.
    """
    runs = [(m.start(), m.end()) for m in _BACKTICK_RUN_RE.finditer(text, lo, hi)]
    by_width: dict = {}
    for k, (s, e) in enumerate(runs):
        by_width.setdefault(e - s, []).append(k)
    blanks = [m.start() for m in _BLANK_LINE_RE.finditer(text, lo, hi)]
    closes: dict = {}
    for m in _HTML_CODE_CLOSE_RE.finditer(text, lo, hi):
        closes.setdefault(m[1].lower(), []).append(m.start())
    html = [(m.start(), m.end(), m[1].lower()) for m in _HTML_CODE_OPEN_RE.finditer(text, lo, hi)] if html else []

    regions: list = []
    pos, k, h = lo, 0, 0
    while k < len(runs) or h < len(html):
        if h < len(html) and (k == len(runs) or html[h][0] < runs[k][0]):
            start, open_end, name = html[h]
            h += 1
            if start < pos:
                continue
            ends = closes.get(name, [])
            j = bisect_left(ends, open_end)
            if j < len(ends):
                end = text.index(">", ends[j]) + 1
                regions.append((start, end))
                pos = end
            continue
        s, e = runs[k]
        k += 1
        if s < pos:
            continue
        same = by_width[e - s]
        j = bisect_left(same, k)
        if j == len(same):
            continue
        close_start, close_end = runs[same[j]]
        b = bisect_left(blanks, e)
        if b < len(blanks) and blanks[b] < close_start:
            continue  # a code span ends with its paragraph
        regions.append((s, close_end))
        pos, k = close_end, same[j] + 1
    return regions


def rich_code_regions(text: str, *, html: bool = True) -> list:
    """Sorted, disjoint ``(start, end)`` spans of *text* that are code; ``html=False`` leaves
    ``<pre>``/``<code>`` elements out (for a caller that is looking for HTML tags)."""
    regions, pos = [], 0
    for start, end in _block_regions(text) + [(len(text), len(text))]:
        regions.extend(_inline_regions(text, pos, start, html))
        if end > start:
            regions.append((start, end))
        pos = end
    return regions


def map_outside_code(text: str, transform, *, html: bool = True) -> str:
    """*text* with ``transform`` applied to each stretch between code regions; code is verbatim."""
    out, pos = [], 0
    for start, end in rich_code_regions(text, html=html):
        out.append(transform(text[pos:start]))
        out.append(text[start:end])
        pos = end
    out.append(transform(text[pos:]))
    return "".join(out)


def strip_code(text: str) -> str:
    """*text* with its Markdown code removed (``<pre>``/``<code>`` elements are kept)."""
    out, pos = [], 0
    for start, end in rich_code_regions(text, html=False):
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)
