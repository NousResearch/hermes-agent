"""Markdown lines outside fenced code blocks: a ``#`` or ``## …`` line inside a fence is content, not a heading."""

from __future__ import annotations

from collections.abc import Iterator


def unfenced_lines(text: str) -> Iterator[tuple[int, str]]:
    """``(offset, line)`` for every line of *text* outside a ```````/``~~~`` fence; *line* keeps its newline."""
    offset, fenced = 0, False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced:
            yield offset, line
        offset += len(line)
