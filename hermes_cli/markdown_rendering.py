"""Pure Markdown source normalization used by the classic CLI renderers."""
from __future__ import annotations

import re

from rich.text import Text

from agent.markdown_tables import realign_markdown_tables


_WINDOWS_PATH_WITH_DOT_SEGMENT_RE = re.compile(r"(?i)(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*")


def plain_source(text: str) -> str:
    """Return literal source text, removing ANSI styling but not Markdown."""
    return Text.from_ansi(text or "").plain


def rich_text_from_ansi(text: str) -> Text:
    """Return Rich text from ANSI output without interpreting bracket markup."""
    return Text.from_ansi(text or "")


def preserve_windows_dot_segments(text: str) -> str:
    r"""Protect ``\.hidden`` Windows path segments from CommonMark's escape rules."""
    if "\\." not in text:
        return text

    def protect(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", match.group(0))

    return _WINDOWS_PATH_WITH_DOT_SEGMENT_RE.sub(protect, text)


def normalize_source(text: str, width: int) -> str:
    """Normalize ANSI-bearing source and realign actual top-level Markdown tables."""
    source = preserve_windows_dot_segments(plain_source(text))
    if "|" not in source:
        return source

    from markdown_it import MarkdownIt

    lines = source.splitlines(keepends=True)
    tables = [token for token in MarkdownIt().enable("table").parse(source)
              if token.type == "table_open" and token.level == 0 and token.map]
    for token in reversed(tables):
        start, end = token.map
        lines[start:end] = [realign_markdown_tables("".join(lines[start:end]), max(1, width))]
    return "".join(lines)


def strip_markdown_syntax(text: str) -> str:
    """Best-effort marker removal for the existing plain-text display mode."""
    plain = plain_source(text)
    plain = re.sub(r"^\s{0,3}(?:[-_]\s*){3,}$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}(?:\*\s*){3}\s*$", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"^\s{0,3}#{1,6}\s+", "", plain, flags=re.MULTILINE)
    plain = re.sub(r"(```+|~~~+)", "", plain)
    plain = re.sub(r"`([^`]*)`", r"\1", plain)
    plain = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", plain)
    plain = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)___([^_]+)___(?!\w)", r"\1", plain)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", plain)
    plain = re.sub(r"\*([^\s*][^*]*?[^\s*])\*", r"\1", plain)
    plain = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", plain)
    plain = re.sub(r"~~([^~]+)~~", r"\1", plain)
    return re.sub(r"\n{3,}", "\n\n", plain).strip("\n")
