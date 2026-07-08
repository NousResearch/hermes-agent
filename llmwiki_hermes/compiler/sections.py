"""Markdown section parsing and merge helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParsedMarkdownNote:
    """Structured view of a note body."""

    title: str
    lead_lines: list[str]
    sections: list[tuple[str, str]]


def normalize_heading(value: str) -> str:
    """Normalize a section heading for deterministic lookup."""

    return " ".join(value.strip().lower().split())


def parse_markdown_sections(body: str) -> ParsedMarkdownNote:
    """Parse a Markdown note into title, lead-in lines, and H2 sections."""

    lines = body.rstrip().splitlines()
    title = ""
    start = 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        start = 1

    lead_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_header, current_lines
        if current_header is None:
            return
        sections.append((current_header, "\n".join(current_lines).strip()))
        current_header = None
        current_lines = []

    for line in lines[start:]:
        if line.startswith("## "):
            flush_current()
            current_header = line[3:].strip()
            current_lines = []
            continue
        if current_header is None:
            lead_lines.append(line.rstrip())
        else:
            current_lines.append(line.rstrip())

    flush_current()
    return ParsedMarkdownNote(title=title, lead_lines=lead_lines, sections=sections)


def section_content(body: str, *headings: str) -> str:
    """Return the first matching H2 section content."""

    parsed = parse_markdown_sections(body)
    wanted = {normalize_heading(item) for item in headings}
    for header, content in parsed.sections:
        if normalize_heading(header) in wanted:
            return content.strip()
    return ""


def merge_line_blocks(existing: str, new: str) -> str:
    """Merge two line-oriented blocks while preserving order."""

    merged: list[str] = []
    seen: set[str] = set()
    for raw in (*existing.splitlines(), *new.splitlines()):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line in seen:
            continue
        seen.add(line)
        merged.append(line)
    return "\n".join(merged)


def merge_paragraph_blocks(existing: str, new: str) -> str:
    """Merge paragraph blocks while preserving order."""

    def collect(text: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.strip():
                current.append(line.rstrip())
                continue
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        if current:
            blocks.append("\n".join(current).strip())
        return blocks

    merged: list[str] = []
    seen: set[str] = set()
    for block in (*collect(existing), *collect(new)):
        if not block or block in seen:
            continue
        seen.add(block)
        merged.append(block)
    return "\n\n".join(merged)


def render_markdown_note(
    title: str,
    lead_lines: list[str],
    ordered_sections: list[tuple[str, str]],
    extra_sections: list[tuple[str, str]],
) -> str:
    """Render a note body from ordered managed sections and preserved extra sections."""

    lines: list[str] = [f"# {title}", ""]
    if any(line.strip() for line in lead_lines):
        lines.extend(lead_lines)
        lines.append("")
    for header, content in [*ordered_sections, *extra_sections]:
        lines.append(f"## {header}")
        if content.strip():
            lines.extend(content.rstrip().splitlines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
