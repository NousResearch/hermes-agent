"""Parse, format, and validate YouTube chapter markers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

CHAPTER_LINE_RE = re.compile(r"^(\S+)\s+(.+?)\s*$")
GENERIC_TITLE_RE = re.compile(r"^(?:part|section|topic)\s*\d*$", re.IGNORECASE)


@dataclass(frozen=True)
class Chapter:
    seconds: int
    title: str


def parse_timestamp(value: str) -> int:
    """Parse MM:SS or HH:MM:SS into whole seconds."""
    parts = value.strip().split(":")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid timestamp: {value}")
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        if seconds >= 60:
            raise ValueError(f"Invalid timestamp: {value}")
        return (minutes * 60) + seconds

    hours, minutes, seconds = map(int, parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid timestamp: {value}")
    return (hours * 3600) + (minutes * 60) + seconds


def format_timestamp(seconds: int) -> str:
    """Format whole seconds using YouTube-compatible chapter syntax."""
    if seconds < 0:
        raise ValueError("Timestamp cannot be negative")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_chapter_lines(text: str) -> list[Chapter]:
    """Parse plain chapter lines into Chapter objects."""
    chapters: list[Chapter] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = CHAPTER_LINE_RE.fullmatch(line)
        if not match:
            raise ValueError(f"Line {line_number} is not a chapter line")
        timestamp, title = match.groups()
        chapters.append(Chapter(parse_timestamp(timestamp), title.strip()))
    return chapters


def format_chapters(chapters: Iterable[Chapter]) -> str:
    """Return paste-ready plain-text chapters."""
    return "\n".join(f"{format_timestamp(chapter.seconds)} {chapter.title}" for chapter in chapters)


def validate_chapters(
    chapters: Iterable[Chapter],
    *,
    duration_seconds: float | None = None,
    require_minimum: bool = True,
) -> list[str]:
    """Return deterministic validation errors; an empty list means valid."""
    items = list(chapters)
    errors: list[str] = []
    if not items:
        return ["Chapter list is empty"]
    if items[0].seconds != 0:
        errors.append("First chapter must start at 00:00")
    if require_minimum and len(items) < 3:
        errors.append("At least 3 chapters are required when the transcript supports them")

    previous: int | None = None
    for index, chapter in enumerate(items, start=1):
        if not chapter.title.strip():
            errors.append(f"Chapter {index} has an empty title")
        if GENERIC_TITLE_RE.fullmatch(chapter.title.strip()):
            errors.append(f"Chapter {index} has a generic title")
        if previous is not None and chapter.seconds <= previous:
            errors.append(f"Chapter {index} timestamp must be strictly increasing")
        if duration_seconds is not None and chapter.seconds > duration_seconds:
            errors.append(f"Chapter {index} exceeds the known video duration")
        previous = chapter.seconds
    return errors
