"""Markdown fenced code blocks (CommonMark §4.5): which lines sit inside a fence, so a ``#`` line or a link there
is read as code, not as a heading or prose."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

# A fenced code block opens with 3+ backticks or 3+ tildes indented at most 3 spaces (CommonMark
# §4.5); a backtick fence's info string may not contain a backtick. It closes only on a line whose
# fence uses the same marker, is at least as long, and carries nothing else — so a ``~~~`` line
# inside a backtick fence, a shorter fence, or a fence line with an info string is all content.
_FENCE_LINE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
# A fence may open (and close) inside a container: a bullet ``- ```sh``, an ordered item ``1. ```sh``,
# a blockquote ``> ```sh``, or a nest of them (§5.1/§5.2). Strip those prefixes before fence matching.
_CONTAINER_PREFIX = re.compile(r"^(?: {0,3}(?:>|(?:[-*+]|\d{1,9}[.)]) {1,4}))+")


def fenced_flags(lines: Iterable[str]) -> Iterator[bool]:
    """For each line, True when it opens, closes or sits inside a fenced block. Fence state is
    ``(marker_char, opener_length)`` rather than a bool so a mismatched fence line cannot drop back into
    prose; an unclosed fence stays code to the end (fail-safe)."""
    fence = None  # (marker char, opener length) while a fenced block is open
    for line in lines:
        match = _FENCE_LINE.match(_CONTAINER_PREFIX.sub("", line.rstrip("\r\n")))
        if fence is not None:
            if (match and match["marker"][0] == fence[0] and len(match["marker"]) >= fence[1]
                    and not match["info"].strip()):
                fence = None
            yield True
        else:
            if match and not (match["marker"][0] == "`" and "`" in match["info"]):
                fence = (match["marker"][0], len(match["marker"]))
            yield fence is not None


def unfenced_lines(text: str) -> Iterator[tuple[int, str]]:
    """``(offset, line)`` for every line of *text* outside a fenced block; *line* keeps its newline."""
    offset = 0
    lines = text.splitlines(keepends=True)
    for line, fenced in zip(lines, fenced_flags(lines)):
        if not fenced:
            yield offset, line
        offset += len(line)
