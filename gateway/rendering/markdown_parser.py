"""Small Markdown-to-document parser for gateway final reply rendering.

This is intentionally not a full CommonMark parser. It only extracts the block
shapes that platform renderers need for native components while preserving text
for lossless fallback.
"""

from __future__ import annotations

import re

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


def _parse_table_at(lines: list[str], start: int) -> tuple[TableBlock | ParagraphBlock, int] | None:
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
    uncertain = False
    i = start + 2
    while i < len(lines):
        candidate = lines[i]
        if not candidate.strip() or "|" not in candidate:
            break
        raw_lines.append(candidate)
        row = _split_table_row(candidate)
        if row is None or len(row) != len(header):
            uncertain = True
        else:
            rows.append(row)
        i += 1

    # Do not commit a partial TableBlock. Once a header and separator establish
    # a table candidate, every contiguous pipe-bearing row must be safe to
    # split; otherwise preserve the complete candidate as raw Markdown.
    if uncertain:
        return ParagraphBlock("\n".join(raw_lines)), i
    if not rows:
        return None
    return TableBlock(headers=header, rows=rows, raw_markdown="\n".join(raw_lines)), i


def _split_table_row(line: str) -> list[str] | None:
    if "|" not in line:
        return None
    stripped = line.strip()
    if not stripped:
        return None
    # Track whether outer pipes were present — a single-cell row like
    # "| content |" is valid only when the pipes wrap the content.
    had_leading_pipe = stripped.startswith("|")
    had_trailing_pipe = stripped.endswith("|")
    if had_leading_pipe:
        stripped = stripped[1:]
    if had_trailing_pipe:
        stripped = stripped[:-1]

    # Minimal state machine: track backslash-escape and backtick spans
    # so that \| and `a|b` are not treated as column delimiters. A code
    # span closes only on a backtick run matching its opening run length.
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    backtick_delimiter = 0
    i = 0

    while i < len(stripped):
        ch = stripped[i]

        if backtick_delimiter:
            if ch == "`":
                run_end = i + 1
                while run_end < len(stripped) and stripped[run_end] == "`":
                    run_end += 1
                run = stripped[i:run_end]
                current.append(run)
                if len(run) == backtick_delimiter:
                    backtick_delimiter = 0
                i = run_end
                continue
            current.append(ch)
            i += 1
            continue

        if escaped:
            # Previous char was backslash; this char is literal (including |)
            current.append(ch)
            escaped = False
            i += 1
            continue

        if ch == "\\":
            escaped = True
            current.append(ch)
            i += 1
            continue

        if ch == "`":
            run_end = i + 1
            while run_end < len(stripped) and stripped[run_end] == "`":
                run_end += 1
            run = stripped[i:run_end]
            backtick_delimiter = len(run)
            current.append(run)
            i = run_end
            continue

        if ch == "|":
            cells.append("".join(current).strip())
            current = []
            i += 1
            continue

        current.append(ch)
        i += 1

    # Unclosed backtick → ambiguous; signal failure so caller degrades safely
    if backtick_delimiter:
        return None

    # Trailing backslash (dangling escape) → ambiguous
    if escaped:
        return None

    cells.append("".join(current).strip())

    # A pipe-wrapped row "| content |" may have just one cell.
    # A bare row without outer pipes needs at least 2 cells to be a table row.
    if had_leading_pipe and had_trailing_pipe:
        if len(cells) < 1:
            return None
    else:
        if len(cells) < 2:
            return None
    return cells


def _is_separator_cell(cell: str) -> bool:
    compact = cell.replace(" ", "")
    return bool(_TABLE_SEPARATOR_CELL_RE.match(compact))
