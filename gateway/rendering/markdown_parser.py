"""Small Markdown-to-document parser for gateway final reply rendering.

This is intentionally not a full CommonMark parser. It only extracts the block
shapes that platform renderers need for native components while preserving text
for lossless fallback.
"""

from __future__ import annotations

import re
from typing import Iterable

from gateway.rendering.document import (
    CodeBlock,
    DividerBlock,
    HeadingBlock,
    MessageDocument,
    ParagraphBlock,
    TableBlock,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```([^`]*)\s*$")
_DIVIDER_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def parse_markdown_document(text: str) -> MessageDocument:
    """Parse a conservative subset of Markdown into a platform-neutral document."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        content = "\n".join(paragraph).strip()
        if content:
            blocks.append(ParagraphBlock(content))
        paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        fence = _FENCE_RE.match(stripped)
        if fence:
            flush_paragraph()
            language = fence.group(1).strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == "```":
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            blocks.append(CodeBlock(language=language, code="\n".join(code_lines)))
            continue

        table = _parse_table_at(lines, i)
        if table is not None:
            flush_paragraph()
            block, next_i = table
            blocks.append(block)
            i = next_i
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            blocks.append(HeadingBlock(level=len(heading.group(1)), text=heading.group(2).strip()))
            i += 1
            continue

        if _DIVIDER_RE.match(stripped):
            flush_paragraph()
            blocks.append(DividerBlock())
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    return MessageDocument(blocks=blocks)


def _parse_table_at(lines: list[str], start: int) -> tuple[TableBlock, int] | None:
    if start + 1 >= len(lines):
        return None
    header = _split_table_row(lines[start])
    separator = _split_table_row(lines[start + 1])
    if not header or not separator or len(header) != len(separator):
        return None
    if not all(_is_separator_cell(cell) for cell in separator):
        return None

    raw_lines = [lines[start], lines[start + 1]]
    rows: list[list[str]] = []
    i = start + 2
    while i < len(lines):
        row = _split_table_row(lines[i])
        if not row:
            break
        raw_lines.append(lines[i])
        rows.append(_fit_row(row, len(header)))
        i += 1

    if not rows:
        return None
    return TableBlock(headers=header, rows=rows, raw_markdown="\n".join(raw_lines)), i


def _split_table_row(line: str) -> list[str] | None:
    if "|" not in line:
        return None
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    if len(cells) < 2:
        return None
    return cells


def _is_separator_cell(cell: str) -> bool:
    compact = cell.replace(" ", "")
    return bool(_TABLE_SEPARATOR_CELL_RE.match(compact))


def _fit_row(row: Iterable[str], width: int) -> list[str]:
    fitted = list(row)[:width]
    if len(fitted) < width:
        fitted.extend("" for _ in range(width - len(fitted)))
    return fitted
