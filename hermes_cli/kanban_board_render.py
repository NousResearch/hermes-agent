"""Board-view rendering for the ``hermes kanban board`` subcommand.

Pure render layer: ``render_board`` takes a list of ``kanban_db.Task`` and
returns a ``rich.table.Table``. No database access, no stdout side effects.
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from hermes_cli import kanban_db as kb

# Canonical column order: (status, header label, Rich style). Covers every
# member of kanban_db.VALID_STATUSES so no task can be silently dropped.
COLUMNS: list[tuple[str, str, str]] = [
    ("triage",    "Triage",    "magenta"),
    ("todo",      "Todo",      "white"),
    ("ready",     "Ready",     "cyan"),
    ("running",   "Running",   "green"),
    ("scheduled", "Scheduled", "blue"),
    ("blocked",   "Blocked",   "red"),
    ("review",    "Review",    "yellow"),
    ("done",      "Done",      "dim green"),
    ("archived",  "Archived",  "bright_black"),
]

# Hidden unless show_all=True (usually noise at a glance).
_COLLAPSED = frozenset({"done", "archived"})

_STATUS_ICON = {
    "triage": "△", "todo": "◻", "ready": "▷", "running": "●",
    "scheduled": "⏱", "blocked": "⊘", "review": "⊙", "done": "✓", "archived": "▤",
}


def render_board(
    tasks: list[kb.Task],
    *,
    board_slug: str = "default",
    other_board_count: int = 0,
    show_all: bool = False,
    limit: Optional[int] = 5,
) -> Table:
    """Build a Rich Table view of the kanban board.

    Parameters
    ----------
    tasks:
        Tasks for the current board (already filtered by the caller).
    board_slug:
        Active board identifier, shown in the table title.
    other_board_count:
        Number of *other* boards; >0 appends a hint to the title.
    show_all:
        Include the collapsed ``done``/``archived`` columns.
    limit:
        Max tasks shown per column; ``None`` means unlimited. Overflow
        appends a ``… (+N hidden)`` line.
    """
    visible_columns = [
        (status, label, style)
        for status, label, style in COLUMNS
        if show_all or status not in _COLLAPSED
    ]

    # Title (+ optional multi-board hint).
    title = f"Board: {board_slug}"
    if other_board_count:
        plural = "s" if other_board_count != 1 else ""
        title += f"  (+{other_board_count} other board{plural}; 'hermes kanban boards list')"

    table = Table(title=title, title_style="bold", min_width=80)

    if not tasks:
        table.add_column("Board", style="dim", min_width=12)
        table.add_row("(no tasks)")
        return table

    # Group tasks into the visible columns. Hidden collapsed tasks are
    # dropped by design; truly-unknown statuses (shouldn't happen given
    # VALID_STATUSES) fall into todo so they are never lost.
    grouped: dict[str, list[kb.Task]] = {status: [] for status, _, _ in visible_columns}
    for t in tasks:
        if t.status in grouped:
            grouped[t.status].append(t)
        elif t.status in _COLLAPSED:
            continue  # hidden unless show_all
        else:
            grouped.setdefault("todo", []).append(t)

    for status, label, style in visible_columns:
        count = len(grouped[status])
        icon = _STATUS_ICON.get(status, "?")
        header = f"{icon} {label}" + (f" ({count})" if count else "")
        table.add_column(header, style=style, no_wrap=True, overflow="ellipsis")

    # Build each column's cell as a list of lines, then emit aligned rows.
    cells: list[list[str]] = []
    for status, _, _ in visible_columns:
        col_tasks = grouped[status]
        shown = col_tasks if limit is None else col_tasks[:limit]
        hidden = len(col_tasks) - len(shown)
        lines = [
            f"{t.id} {t.title}" + (f" [{t.assignee}]" if t.assignee else "")
            for t in shown
        ]
        if hidden > 0:
            lines.append(f"… (+{hidden} hidden)")
        cells.append(lines)

    max_rows = max((len(c) for c in cells), default=0)
    for i in range(max_rows):
        table.add_row(*[(c[i] if i < len(c) else "") for c in cells])

    return table


# Rough min characters a single status column needs to stay readable.
_MIN_COL_WIDTH = 16


def _needed_width(show_all: bool) -> int:
    """Approximate terminal width the columnar board needs to fit."""
    n = len([c for c in COLUMNS if show_all or c[0] not in _COLLAPSED])
    return n * _MIN_COL_WIDTH


def should_stack(width: int, *, show_all: bool = False) -> bool:
    """True when the columnar board won't fit ``width`` and the caller
    should fall back to the vertical stacked layout."""
    return width < _needed_width(show_all)


def render_board_stacked(
    tasks: list[kb.Task],
    *,
    board_slug: str = "default",
    other_board_count: int = 0,
    show_all: bool = False,
    limit: Optional[int] = 5,
) -> Group:
    """Vertical board layout for narrow terminals.

    Each non-empty status becomes a section header followed by its task
    lines, one per line. Fits any width (titles wrap instead of truncating),
    which makes it the readable choice over SSH from a phone. Same grouping
    and collapse rules as ``render_board``.
    """
    visible_statuses = [
        (status, label, style)
        for status, label, style in COLUMNS
        if show_all or status not in _COLLAPSED
    ]

    title = f"Board: {board_slug}"
    if other_board_count:
        plural = "s" if other_board_count != 1 else ""
        title += f"  (+{other_board_count} other board{plural}; 'hermes kanban boards list')"

    blocks: list = [Text(title, style="bold")]

    if not tasks:
        blocks.append(Text("(no tasks)", style="dim"))
        return Group(*blocks)

    grouped: dict[str, list[kb.Task]] = {status: [] for status, _, _ in visible_statuses}
    for t in tasks:
        if t.status in grouped:
            grouped[t.status].append(t)
        elif t.status in _COLLAPSED:
            continue  # hidden unless show_all
        else:
            grouped.setdefault("todo", []).append(t)

    any_section = False
    for status, label, style in visible_statuses:
        col_tasks = grouped[status]
        if not col_tasks:
            continue  # skip empty sections — vertical noise
        any_section = True
        icon = _STATUS_ICON.get(status, "?")
        section = Text()
        section.append(f"\n{icon} {label} ({len(col_tasks)})", style=f"bold {style}")
        shown = col_tasks if limit is None else col_tasks[:limit]
        hidden = len(col_tasks) - len(shown)
        for t in shown:
            line = f"\n  {t.id} {t.title}" + (f" [{t.assignee}]" if t.assignee else "")
            section.append(line)
        if hidden > 0:
            section.append(f"\n  … (+{hidden} hidden)", style="dim")
        blocks.append(section)

    if not any_section:
        # Everything was collapsed (e.g. only done/archived without show_all).
        blocks.append(Text("(no tasks)", style="dim"))

    return Group(*blocks)


def _render_to_string(renderable) -> str:
    """Render any Rich renderable (Table or Group) to a plain, uncolored
    string for tests. Width 200 keeps the wide table from truncating its
    headers, so substring assertions stay reliable."""
    console = Console(width=200, force_terminal=False, color_system=None)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()
