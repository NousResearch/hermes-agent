"""Parser for markdown AUTO blocks used by Hermes self-knowledge docs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


AUTO_START_RE = re.compile(r"<!--\s*AUTO-START:\s*([a-z0-9_-]+)\s*-->")
AUTO_END_RE = re.compile(r"<!--\s*AUTO-END:\s*([a-z0-9_-]+)\s*-->")


@dataclass(frozen=True)
class AutoBlock:
    """Location and content for one AUTO block in a markdown document."""

    name: str
    start: int
    end: int
    body_start: int
    body_end: int
    body: str


def _line_end_after(text: str, index: int) -> int:
    """Return the position immediately after the line ending at or after *index*."""
    newline = text.find("\n", index)
    if newline == -1:
        return len(text)
    return newline + 1


def _line_start_before(text: str, index: int) -> int:
    """Return the start position of the line containing *index*."""
    newline = text.rfind("\n", 0, index)
    return 0 if newline == -1 else newline + 1


def parse_auto_blocks(text: str) -> dict[str, AutoBlock]:
    """Parse AUTO blocks from *text*.

    Blocks are delimited by matching HTML comments:
    ``<!-- AUTO-START: name -->`` and ``<!-- AUTO-END: name -->``.
    Returned blocks preserve byte offsets into the original Python string so
    callers can replace only generated bodies and leave hand-written prose
    untouched.
    """
    blocks: dict[str, AutoBlock] = {}
    position = 0

    while True:
        start_match = AUTO_START_RE.search(text, position)
        if not start_match:
            break

        name = start_match.group(1)
        if name in blocks:
            raise ValueError(f"Duplicate AUTO block: {name}")

        body_start = _line_end_after(text, start_match.end())
        end_match = AUTO_END_RE.search(text, body_start)
        if not end_match:
            raise ValueError(f"Missing AUTO-END marker for block: {name}")

        end_name = end_match.group(1)
        if end_name != name:
            raise ValueError(
                f"Mismatched AUTO block markers: start {name!r}, end {end_name!r}"
            )

        body_end = _line_start_before(text, end_match.start())
        blocks[name] = AutoBlock(
            name=name,
            start=start_match.start(),
            end=end_match.end(),
            body_start=body_start,
            body_end=body_end,
            body=text[body_start:body_end],
        )
        position = end_match.end()

    return blocks


def _line_ending_near(text: str, index: int) -> str:
    """Infer the line ending around *index*, defaulting to LF."""
    previous = text.rfind("\n", 0, index)
    if previous != -1 and previous > 0 and text[previous - 1] == "\r":
        return "\r\n"
    next_newline = text.find("\n", index)
    if next_newline != -1 and next_newline > 0 and text[next_newline - 1] == "\r":
        return "\r\n"
    return "\n"


def _format_replacement_body(text: str, block: AutoBlock, replacement: str) -> str:
    """Normalize a replacement so markers stay on their own lines."""
    newline = _line_ending_near(text, block.body_start)
    body = replacement.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        body = body.replace("\n", newline)
    if body and not body.endswith(newline):
        body += newline
    return body


def replace_auto_blocks(text: str, replacements: Mapping[str, str]) -> str:
    """Replace AUTO block bodies using *replacements*.

    Unknown replacement keys are ignored. Blocks not present in replacements are
    left untouched, making a no-replacement round trip an exact no-op.
    """
    blocks = parse_auto_blocks(text)
    if not replacements:
        return text

    rendered = text
    for block in sorted(blocks.values(), key=lambda b: b.body_start, reverse=True):
        if block.name not in replacements:
            continue
        body = _format_replacement_body(text, block, replacements[block.name])
        rendered = rendered[: block.body_start] + body + rendered[block.body_end :]
    return rendered
