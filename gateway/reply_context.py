"""Formatting helpers for reply metadata injected into inbound messages."""

from __future__ import annotations

import re

from gateway.platforms.helpers import strip_markdown


# Keep synthetic reply metadata useful without letting it bury the new message.
REPLY_CONTEXT_EXCERPT_MAX_CHARS = 280

_BLOCK_PREFIX_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|>\s?|[-+*]\s+|\d+[.)]\s+)")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([\\`*{}\[\]()#+\-.!_|>~])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_END_RE = re.compile(r"[.!?。！？](?:[\"'”’）)]*)")


def _strip_block_prefixes(line: str) -> str:
    """Remove nested Markdown block markers while preserving their text."""
    while True:
        cleaned = _BLOCK_PREFIX_RE.sub("", line, count=1)
        if cleaned == line:
            return line
        line = cleaned


def _split_table_cells(line: str) -> list[str] | None:
    """Return cells when *line* contains an unescaped table delimiter."""
    parts = re.split(r"(?<!\\)\|", line)
    if len(parts) < 2:
        return None
    return [cell.strip() for cell in parts if cell.strip()]


def _is_table_separator(line: str) -> bool:
    cells = _split_table_cells(line)
    return bool(cells) and all(
        _TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells
    )


def _flatten_markdown_lines(lines: list[str]) -> list[str]:
    """Flatten confirmed Markdown tables while preserving ordinary pipes."""
    lines = [_strip_block_prefixes(line.strip()) for line in lines]
    flattened: list[str] = []
    in_table = False

    for index, line in enumerate(lines):
        cells = _split_table_cells(line)
        next_is_separator = (
            cells is not None
            and index + 1 < len(lines)
            and _is_table_separator(lines[index + 1])
        )
        if next_is_separator:
            flattened.append("; ".join(cells))
            in_table = True
            continue
        if _is_table_separator(line):
            continue
        if in_table and cells:
            flattened.append("; ".join(cells))
            continue

        in_table = False
        flattened.append(line)

    return flattened


def _truncate_at_readable_boundary(text: str, max_chars: int) -> str:
    """Bound text, preferring a nearby sentence or word boundary."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"[:max_chars]

    budget = max_chars - 1
    candidate = text[:budget]
    minimum_boundary = budget // 2

    sentence_ends = [
        match.end()
        for match in _SENTENCE_END_RE.finditer(candidate)
        if match.end() >= minimum_boundary
    ]
    if sentence_ends:
        end = sentence_ends[-1]
    else:
        word_end = candidate.rfind(" ")
        end = word_end if word_end >= minimum_boundary else budget

    return candidate[:end].rstrip(" ,;:-") + "…"


def format_reply_context_excerpt(
    text: str,
    *,
    max_chars: int = REPLY_CONTEXT_EXCERPT_MAX_CHARS,
) -> str:
    """Return a single-line, wrapper-safe plaintext excerpt of a reply target.

    Markdown is removed before truncation so the excerpt cannot end halfway
    through a table or formatting delimiter. The returned value, including an
    ellipsis when needed, never exceeds ``max_chars``.
    """
    plaintext = strip_markdown(text)
    lines = _flatten_markdown_lines(plaintext.splitlines())
    plaintext = " ".join(line for line in lines if line)
    plaintext = _MARKDOWN_ESCAPE_RE.sub(r"\1", plaintext)
    plaintext = plaintext.replace("`", "").replace("*", "").replace("~", "")
    plaintext = plaintext.replace("[", "(").replace("]", ")")
    plaintext = _CONTROL_CHAR_RE.sub("", plaintext)
    plaintext = re.sub(r"\s+", " ", plaintext).strip()
    return _truncate_at_readable_boundary(plaintext, max_chars)
