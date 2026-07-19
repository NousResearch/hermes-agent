"""Markdown heading utilities for complete, fence-aware section retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class MarkdownHeading:
    start: int
    level: int
    title: str


def markdown_headings(content: str) -> list[MarkdownHeading]:
    """Return headings outside fenced code blocks with character offsets."""
    headings: list[MarkdownHeading] = []
    offset = 0
    fence_char = ""
    fence_length = 0
    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        fence = _FENCE_RE.match(stripped)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""
                fence_length = 0
            offset += len(line)
            continue
        if not fence_char:
            match = _HEADING_RE.match(stripped)
            if match:
                headings.append(
                    MarkdownHeading(
                        start=offset,
                        level=len(match.group(1)),
                        title=match.group(2).strip().rstrip("#").strip(),
                    )
                )
        offset += len(line)
    return headings


def split_markdown_sections(content: str) -> list[tuple[str, str, int]]:
    """Split into complete sections; child headings stay inside parents."""
    headings = markdown_headings(content)
    if not headings:
        return [("", content.strip(), 0)] if content.strip() else []

    sections: list[tuple[str, str, int]] = []
    preamble = content[: headings[0].start].strip()
    if preamble:
        sections.append(("", preamble, 0))
    for index, heading in enumerate(headings):
        end = len(content)
        for candidate in headings[index + 1 :]:
            if candidate.level <= heading.level:
                end = candidate.start
                break
        sections.append(
            (heading.title, content[heading.start:end].strip(), heading.level)
        )
    return sections


def select_markdown_section(
    content: str,
    requested: str,
) -> tuple[str | None, str | None, list[str]]:
    """Select one exact case-insensitive heading and its complete section."""
    headings = markdown_headings(content)
    available = [heading.title for heading in headings]
    wanted = requested.strip().casefold()
    matches = [
        (index, heading)
        for index, heading in enumerate(headings)
        if heading.title.casefold() == wanted
    ]
    if len(matches) != 1:
        return None, None, available

    index, heading = matches[0]
    end = len(content)
    for candidate in headings[index + 1 :]:
        if candidate.level <= heading.level:
            end = candidate.start
            break
    return content[heading.start:end].strip(), heading.title, available
