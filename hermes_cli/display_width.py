"""Terminal-cell-aware formatting helpers for plain-text CLI tables.

Rich is already a required Hermes dependency and handles CJK, combining marks,
and emoji grapheme clusters.  Keeping this logic here prevents individual CLI
surfaces from drifting back to Python code-point padding (``:<N`` / slicing).
"""

from __future__ import annotations

from rich.cells import cell_len, chop_cells, set_cell_size


def cell_width(text: str) -> int:
    """Return the number of terminal cells occupied by ``text``."""
    return cell_len(text)


def truncate_cells(text: str, max_cells: int, *, ellipsis: str = "…") -> str:
    """Truncate ``text`` within a terminal-cell budget without splitting graphemes."""
    if max_cells <= 0:
        return ""
    if cell_len(text) <= max_cells:
        return text

    ellipsis_width = cell_len(ellipsis)
    if ellipsis_width >= max_cells:
        return set_cell_size(ellipsis, max_cells).rstrip()

    chunks = chop_cells(text, max_cells - ellipsis_width)
    prefix = chunks[0] if chunks else ""
    return prefix.rstrip() + ellipsis


def fit_cells(text: str, width: int, *, ellipsis: str = "…") -> str:
    """Truncate then right-pad ``text`` to exactly ``width`` terminal cells."""
    if width <= 0:
        return ""
    return set_cell_size(truncate_cells(text, width, ellipsis=ellipsis), width)


def format_columns(*columns: tuple[str, int | None], separator: str = " ") -> str:
    """Join columns after cell-aware padding; ``None`` leaves a column unbounded."""
    return separator.join(
        text if width is None else fit_cells(text, width)
        for text, width in columns
    )
