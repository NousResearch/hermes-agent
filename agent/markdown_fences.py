"""Markdown lines outside fenced code blocks: a ``#`` or ``## …`` line inside a fence is content, not a heading."""

from __future__ import annotations

import re
from collections.abc import Iterator

# CommonMark: up to 3 spaces, then a run of 3+ backticks or tildes (a backtick opener's info string has none).
_FENCE_RE = re.compile(r"^ {0,3}(?P<run>`{3,}|~{3,})(?P<info>.*)$")


def unfenced_lines(text: str) -> Iterator[tuple[int, str]]:
    """``(offset, line)`` for every line of *text* outside a fenced block; *line* keeps its newline. A fence closes
    only on a bare run of its own character at least as long as its opener, so a four-backtick example may quote
    a triple-backtick one."""
    offset, opener = 0, ""
    for line in text.splitlines(keepends=True):
        fence = _FENCE_RE.match(line.rstrip("\r\n"))
        if opener:
            if fence and fence.group("run")[0] == opener[0] and len(fence.group("run")) >= len(opener) \
                    and not fence.group("info").strip():
                opener = ""
        elif fence and not (fence.group("run")[0] == "`" and "`" in fence.group("info")):
            opener = fence.group("run")
        else:
            yield offset, line
        offset += len(line)
