"""Small, fail-open renderer for Markdown sent through Discord.

Discord does not implement GFM tables.  This module only rewrites complete
table blocks outside fenced code, leaving all other Markdown untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from wcwidth import wcswidth

_DIVIDER = re.compile(r"^\s*:?-{3,}:?\s*$")
_STATUS_HEADERS = {"status", "result", "health", "severity", "priority"}
_STATUS_EMOJI = {
    "success": "✅", "successful": "✅", "passed": "✅", "healthy": "✅",
    "operational": "✅", "complete": "✅", "completed": "✅", "ready": "✅",
    "warning": "⚠️", "degraded": "⚠️", "pending": "⚠️",
    "failure": "❌", "failed": "❌", "error": "❌", "unhealthy": "❌",
    "critical": "🔴", "high": "🔴", "medium": "🟠", "low": "🟢",
}
_MAX_FIELDS = 25


@dataclass(frozen=True)
class DiscordMarkdown:
    """Discord-safe text plus optional, short status fields for an embed."""

    content: str
    fields: list[tuple[str, str]]


def _width(value: str) -> int:
    return max(wcswidth(value), 0)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().removeprefix("|").removesuffix("|").split("|")]


def _is_divider(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(_DIVIDER.match(cell) for cell in cells)


def _is_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (stripped.startswith("|") or stripped.count("|") >= 2)


def _render_table(header: list[str], rows: list[list[str]]) -> tuple[str, list[tuple[str, str]]]:
    """Render a complete table as a display-width-aligned Discord code block."""
    columns = max(len(header), *(len(row) for row in rows))
    normalized = [row + [""] * (columns - len(row)) for row in [header, *rows]]
    fields: list[tuple[str, str]] = []
    remove: set[int] = set()
    for index, name in enumerate(normalized[0]):
        key = name.strip().lower()
        values = [row[index].strip() for row in normalized[1:]]
        # A column is promoted only when every non-empty value is an explicit,
        # short status. This avoids reinterpreting arbitrary user content.
        if key in _STATUS_HEADERS and values and all(value.lower() in _STATUS_EMOJI for value in values):
            remove.add(index)
            fields.extend((name or "Status", f"{_STATUS_EMOJI[value.lower()]} {value}") for value in values)
    fields = fields[:_MAX_FIELDS]
    keep = [index for index in range(columns) if index not in remove]
    # Forum sends do not carry embeds. Keep a pseudo-table for an all-status
    # table so every delivery path retains visible content.
    if not keep:
        keep = list(range(columns))
    visible = [[row[index] for index in keep] for row in normalized]
    widths = [max(3, *(_width(row[index]) for row in visible)) for index in range(len(keep))]

    def row(cells: list[str]) -> str:
        padded = (cell + " " * max(0, widths[index] - _width(cell)) for index, cell in enumerate(cells))
        return "| " + " | ".join(padded) + " |"

    divider = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "```text\n" + "\n".join([row(visible[0]), divider, *(row(item) for item in visible[1:])]) + "\n```", fields


def render_discord_markdown(content: str) -> DiscordMarkdown:
    """Convert GFM table blocks while preserving fenced code and failing open."""
    if not content or "|" not in content:
        return DiscordMarkdown(content, [])
    try:
        lines = content.split("\n")
        rendered: list[str] = []
        fields: list[tuple[str, str]] = []
        index = 0
        in_fence = False
        while index < len(lines):
            line = lines[index]
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                rendered.append(line)
                index += 1
                continue
            if not in_fence and _is_row(line) and index + 1 < len(lines) and _is_divider(lines[index + 1]):
                header = _cells(line)
                rows: list[list[str]] = []
                end = index + 2
                while end < len(lines) and _is_row(lines[end]):
                    rows.append(_cells(lines[end]))
                    end += 1
                block, block_fields = _render_table(header, rows)
                if block:
                    rendered.append(block)
                fields.extend(block_fields[: max(0, _MAX_FIELDS - len(fields))])
                index = end
                continue
            rendered.append(line)
            index += 1
        return DiscordMarkdown("\n".join(rendered), fields)
    except Exception:
        # Formatting must never prevent a response from being delivered.
        return DiscordMarkdown(content, [])
