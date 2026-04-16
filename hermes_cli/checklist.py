"""Shared curses-based multi-select checklist for Hermes CLI.

Used by both ``hermes tools`` and ``hermes skills`` to present a
toggleable list of items.  Falls back to a numbered text UI when
curses is unavailable (Windows without curses, piped stdin, etc.).
"""

import sys
from typing import List, Set

from hermes_cli.colors import Colors, color


def curses_checklist(
    title: str,
    items: List[str],
    pre_selected: Set[int],
) -> Set[int]:
    """Multi-select checklist.  Returns set of **selected** indices.

    Args:
        title: Header text shown at the top of the checklist.
        items: Display labels for each row.
        pre_selected: Indices that start checked.

    Returns:
        The indices the user confirmed as checked.  On cancel (ESC/q),
        returns ``pre_selected`` unchanged.
    """
    # Safety: return defaults when stdin is not a terminal.
    if not sys.stdin.isatty():
        return set(pre_selected)

    try:
        import curses
        selected = set(pre_selected)
        result = [None]

        def _ui(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, 8, -1)  # dim gray
            cursor = 0
            scroll_offset = 0

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()

                # Header
                try:
                    hattr = curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
                    stdscr.addnstr(0, 0, title, max_x - 1, hattr)
                    stdscr.addnstr(
                        1, 0,
                        "  ↑↓ navigate  SPACE toggle  ENTER confirm  ESC cancel",
                        max_x - 1, curses.A_DIM,
                    )
                except curses.error:
                    pass

                # Scrollable item list
                visible_rows = max_y - 3
                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + visible_rows:
                    scroll_offset = cursor - visible_rows + 1

                for draw_i, i in enumerate(
                    range(scroll_offset, min(len(items), scroll_offset + visible_rows))
                ):
                    y = draw_i + 3
                    if y >= max_y - 1:
                        break
                    check = "✓" if i in selected else " "
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow} [{check}] {items[i]}"

                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord("k")):
                    cursor = (cursor - 1) % len(items)
                elif key in (curses.KEY_DOWN, ord("j")):
                    cursor = (cursor + 1) % len(items)
                elif key == ord(" "):
                    selected.symmetric_difference_update({cursor})
                elif key in (curses.KEY_ENTER, 10, 13):
                    result[0] = set(selected)
                    return
                elif key in (27, ord("q")):
                    result[0] = set(pre_selected)
                    return

        curses.wrapper(_ui)
        return result[0] if result[0] is not None else set(pre_selected)

    except Exception:
        pass  # fall through to numbered fallback

    # ── Numbered text fallback ────────────────────────────────────────────
    selected = set(pre_selected)
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

    while True:
        for i, label in enumerate(items):
            check = "✓" if i in selected else " "
            print(f"    {i + 1:3}. [{check}] {label}")
        print()

        try:
            raw = input(color("  Number to toggle, 's' to save, 'q' to cancel: ", Colors.DIM)).strip()
        except (KeyboardInterrupt, EOFError):
            return set(pre_selected)

        if raw.lower() == "s" or raw == "":
            return selected
        if raw.lower() == "q":
            return set(pre_selected)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                selected.symmetric_difference_update({idx})
        except ValueError:
            print(color("  Invalid input", Colors.DIM))
