"""Clean-room ASCII canvas and docs-renderer prototype.

This module intentionally implements only behavior-level ideas captured from the
Spearhead source spike for qindapao/Nokse22 ascii-draw. It does not copy source
text or structure from that GPL project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import unicodedata

LEFT = 1
RIGHT = 2
UP = 4
DOWN = 8

_LINE_GLYPHS = {
    LEFT: "─",
    RIGHT: "─",
    UP: "│",
    DOWN: "│",
    LEFT | RIGHT: "─",
    UP | DOWN: "│",
    RIGHT | DOWN: "┌",
    LEFT | DOWN: "┐",
    RIGHT | UP: "└",
    LEFT | UP: "┘",
    LEFT | RIGHT | DOWN: "┬",
    LEFT | RIGHT | UP: "┴",
    RIGHT | UP | DOWN: "├",
    LEFT | UP | DOWN: "┤",
    LEFT | RIGHT | UP | DOWN: "┼",
}


@dataclass
class _Cell:
    char: str = " "
    mask: int = 0


def char_width(char: str) -> int:
    """Return display width for one Unicode code point.

    Policy: combining marks are zero-width; East Asian Wide, Full-width, and
    Ambiguous characters count as two columns; all other code points count as
    one. This is deterministic for snapshot docs even though terminals may vary
    on Ambiguous width.
    """

    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 1


def display_width(text: str) -> int:
    return sum(char_width(char) for char in text)


def _pad_display(text: str, width: int) -> str:
    padding = max(0, width - display_width(text))
    return f"{text}{' ' * padding}"


class Canvas:
    """Fixed-size character canvas addressed in display cells."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Canvas width and height must be positive")
        self.width = width
        self.height = height
        self._grid = [[_Cell() for _ in range(width)] for _ in range(height)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _set_text_cell(self, x: int, y: int, char: str) -> None:
        if not self._in_bounds(x, y):
            return
        self._grid[y][x] = _Cell(char=char, mask=0)

    def _add_line_mask(self, x: int, y: int, mask: int) -> None:
        if not self._in_bounds(x, y):
            return
        cell = self._grid[y][x]
        cell.mask |= mask
        cell.char = _LINE_GLYPHS.get(cell.mask, cell.char if cell.char != " " else "┼")

    def text(self, x: int, y: int, value: str) -> None:
        cursor = x
        for char in value:
            width = char_width(char)
            if width == 0:
                if self._in_bounds(cursor - 1, y):
                    self._grid[y][cursor - 1].char += char
                continue
            if cursor >= self.width:
                break
            self._set_text_cell(cursor, y, char)
            if width == 2:
                if cursor + 1 < self.width:
                    self._set_text_cell(cursor + 1, y, "")
                cursor += 2
            else:
                cursor += 1

    def horizontal_line(self, x1: int, y: int, x2: int) -> None:
        start, end = sorted((x1, x2))
        for x in range(start, end + 1):
            mask = 0
            if x > start:
                mask |= LEFT
            if x < end:
                mask |= RIGHT
            self._add_line_mask(x, y, mask)

    def vertical_line(self, x: int, y1: int, y2: int) -> None:
        start, end = sorted((y1, y2))
        for y in range(start, end + 1):
            mask = 0
            if y > start:
                mask |= UP
            if y < end:
                mask |= DOWN
            self._add_line_mask(x, y, mask)

    def rectangle(self, x: int, y: int, width: int, height: int) -> None:
        if width < 2 or height < 2:
            raise ValueError("Rectangle width and height must be at least 2")
        right = x + width - 1
        bottom = y + height - 1
        self.horizontal_line(x, y, right)
        self.horizontal_line(x, bottom, right)
        self.vertical_line(x, y, bottom)
        self.vertical_line(right, y, bottom)

    def arrow(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if y1 == y2:
            if x1 == x2:
                self._set_text_cell(x2, y2, "•")
                return
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2, step):
                mask = RIGHT if step > 0 else LEFT
                if x != x1:
                    mask |= LEFT if step > 0 else RIGHT
                self._add_line_mask(x, y1, mask)
            self._set_text_cell(x2, y2, ">" if step > 0 else "<")
            return
        if x1 == x2:
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2, step):
                mask = DOWN if step > 0 else UP
                if y != y1:
                    mask |= UP if step > 0 else DOWN
                self._add_line_mask(x1, y, mask)
            self._set_text_cell(x2, y2, "↓" if step > 0 else "↑")
            return
        corner_x = x2
        self.horizontal_line(x1, y1, corner_x)
        self.arrow(corner_x, y1, x2, y2)

    def render(self) -> str:
        return "\n".join("".join(cell.char for cell in row) for row in self._grid)

def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    all_rows = [list(headers), *[list(row) for row in rows]]
    if not headers:
        return ""
    widths = [0] * len(headers)
    for row in all_rows:
        for index, value in enumerate(row[: len(headers)]):
            widths[index] = max(widths[index], display_width(str(value)))
    inner_widths = [max(width + 2, 7) for width in widths]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * width for width in inner_widths) + right

    def row_line(row: Sequence[str]) -> str:
        cells = []
        for index, width in enumerate(widths):
            value = str(row[index]) if index < len(row) else ""
            cells.append(" " + _pad_display(value, inner_widths[index] - 2) + " ")
        return "│" + "│".join(cells) + "│"

    rendered = [border("┌", "┬", "┐"), row_line(headers), border("├", "┼", "┤")]
    rendered.extend(row_line(row) for row in rows)
    rendered.append(border("└", "┴", "┘"))
    return "\n".join(rendered)


def _tree_children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, Mapping):
        return [(str(key), value) for key, value in node.items()]
    if isinstance(node, (list, tuple)):
        return [(str(item), None) for item in node]
    return []


def render_tree(tree: Mapping[str, Any] | Sequence[Any]) -> str:
    roots = _tree_children(tree)
    lines: list[str] = []

    def walk(label: str, value: Any, prefix: str, is_last: bool, is_root: bool = False) -> None:
        if is_root:
            lines.append(label)
            child_prefix = ""
        else:
            connector = "└─ " if is_last else "├─ "
            lines.append(prefix + connector + label)
            child_prefix = prefix + ("   " if is_last else "│  ")
        children = _tree_children(value)
        for index, (child_label, child_value) in enumerate(children):
            walk(child_label, child_value, child_prefix, index == len(children) - 1)

    for index, (label, value) in enumerate(roots):
        walk(label, value, "", index == len(roots) - 1, is_root=True)
    return "\n".join(lines)


def render_title(text: str, *, figlet: bool = False) -> str:
    if not figlet:
        return text
    try:
        import pyfiglet  # type: ignore[import-not-found]
    except Exception:
        return text
    return pyfiglet.figlet_format(text).rstrip("\n")


__all__ = [
    "Canvas",
    "char_width",
    "display_width",
    "render_table",
    "render_tree",
    "render_title",
]
