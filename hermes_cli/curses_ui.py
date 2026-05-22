"""Shared curses-based UI components for Hermes CLI.

Used by `hermes tools` and `hermes skills` for interactive checklists.
Provides a curses multi-select with keyboard navigation, plus a
text-based numbered fallback for terminals without curses support.
"""
from dataclasses import dataclass
import sys
from typing import Callable, List, Optional, Set

from hermes_cli.colors import Colors, color


def _query_matches(label: str, query: str) -> bool:
    """Return True when every query token is a case-insensitive subsequence."""
    normalized = label.lower()
    tokens = query.lower().split()

    if not tokens:
        return True

    for token in tokens:
        pos = 0

        for ch in token:
            pos = normalized.find(ch, pos)

            if pos < 0:
                return False

            pos += 1

    return True


def _filter_indices(items: List[str], query: str) -> List[int]:
    """Return original item indices matching *query*, preserving list order."""
    q = query.strip()

    if not q:
        return list(range(len(items)))

    return [i for i, label in enumerate(items) if _query_matches(label, q)]


@dataclass
class _SearchState:
    """Mutable search state shared by curses picker loops."""

    active: bool = False
    query: str = ""


def _reconcile_cursor(filtered: List[int], cursor: int) -> tuple[int, int]:
    """Return ``(cursor, cursor_pos)`` inside the filtered index list."""
    if not filtered:
        return cursor, 0

    if cursor not in filtered:
        cursor = filtered[0]

    return cursor, filtered.index(cursor)


def _move_filtered_cursor(filtered: List[int], cursor: int, cursor_pos: int, delta: int) -> int:
    """Move through the filtered index list, wrapping like the legacy menus."""
    if not filtered:
        return cursor

    return filtered[(cursor_pos + delta) % len(filtered)]


def _scroll_for_cursor(scroll_offset: int, cursor_pos: int, visible_rows: int, total_rows: int) -> int:
    """Clamp scroll offset so the cursor remains visible."""
    visible_rows = max(1, visible_rows)

    if cursor_pos < scroll_offset:
        scroll_offset = cursor_pos
    elif cursor_pos >= scroll_offset + visible_rows:
        scroll_offset = cursor_pos - visible_rows + 1

    return max(0, min(scroll_offset, max(0, total_rows - visible_rows)))


def _handle_active_search_key(curses_mod, key: int, search: _SearchState) -> tuple[bool, bool, bool]:
    """Handle a key while the search prompt is active.

    Returns ``(handled, confirm, changed)``. Active search consumes query
    editing keys, but leaves navigation keys for the menu loop to handle.
    """
    if not search.active:
        return False, False, False

    if key == 27:
        search.active = False
        return True, False, False

    if key in (curses_mod.KEY_BACKSPACE, 127, 8):
        search.query = search.query[:-1]
        return True, False, True

    if key == 21:  # Ctrl+U
        search.query = ""
        return True, False, True

    if key in (curses_mod.KEY_ENTER, 10, 13):
        return True, True, False

    if 0 <= key < 256 and chr(key).isprintable():
        search.query += chr(key)
        return True, False, True

    return False, False, False


def flush_stdin() -> None:
    """Flush any stray bytes from the stdin input buffer.

    Must be called after ``curses.wrapper()`` (or any terminal-mode library
    like simple_term_menu) returns, **before** the next ``input()`` /
    ``getpass.getpass()`` call.  ``curses.endwin()`` restores the terminal
    but does NOT drain the OS input buffer — leftover escape-sequence bytes
    (from arrow keys, terminal mode-switch responses, or rapid keypresses)
    remain buffered and silently get consumed by the next ``input()`` call,
    corrupting user data (e.g. writing ``^[^[`` into .env files).

    On non-TTY stdin (piped, redirected) or Windows, this is a no-op.
    """
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def curses_checklist(
    title: str,
    items: List[str],
    selected: Set[int],
    *,
    cancel_returns: Set[int] | None = None,
    status_fn: Optional[Callable[[Set[int]], str]] = None,
    searchable: bool = False,
) -> Set[int]:
    """Curses multi-select checklist. Returns set of selected indices.

    Args:
        title: Header line displayed above the checklist.
        items: Display labels for each row.
        selected: Indices that start checked (pre-selected).
        cancel_returns: Returned on ESC/q. Defaults to the original *selected*.
        status_fn: Optional callback ``f(chosen_indices) -> str`` whose return
            value is rendered on the bottom row of the terminal.  Use this for
            live aggregate info (e.g. estimated token counts).
        searchable: When true, "/" opens a filter prompt. Selection and return
            values always use original item indices, not filtered row positions.
    """
    if cancel_returns is None:
        cancel_returns = set(selected)

    # Safety: curses and input() both hang or spin when stdin is not a
    # terminal (e.g. subprocess pipe).  Return defaults immediately.
    if not sys.stdin.isatty():
        return cancel_returns

    try:
        import curses
        chosen = set(selected)
        result_holder: list = [None]

        def _draw(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)  # dim gray
            cursor = 0
            scroll_offset = 0
            search = _SearchState()

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                filtered = _filter_indices(items, search.query) if searchable else list(range(len(items)))
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                # Reserve bottom row for status bar when status_fn provided
                footer_rows = 1 if status_fn else 0

                # Header
                try:
                    hattr = curses.A_BOLD
                    if curses.has_colors():
                        hattr |= curses.color_pair(2)
                    stdscr.addnstr(0, 0, title, max_x - 1, hattr)
                    if searchable and search.active:
                        help_text = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop search"
                    elif searchable and search.query:
                        help_text = (
                            f"  Filter: {search.query} ({len(filtered)}/{len(items)})  "
                            "↑↓ navigate  SPACE toggle  ENTER confirm  / edit  ESC/q cancel"
                        )
                    elif searchable:
                        help_text = "  ↑↓ navigate  SPACE toggle  ENTER confirm  / search  ESC/q cancel"
                    else:
                        help_text = "  ↑↓ navigate  SPACE toggle  ENTER confirm  ESC cancel"
                    stdscr.addnstr(1, 0, help_text, max_x - 1, curses.A_DIM)
                except curses.error:
                    pass

                # Scrollable item list
                visible_rows = max(1, max_y - 3 - footer_rows)
                scroll_offset = _scroll_for_cursor(scroll_offset, cursor_pos, visible_rows, len(filtered))

                if searchable and search.query and not filtered:
                    try:
                        stdscr.addnstr(3, 0, "  No matches", max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass

                for draw_i, filtered_pos in enumerate(
                    range(scroll_offset, min(len(filtered), scroll_offset + visible_rows))
                ):
                    i = filtered[filtered_pos]
                    y = draw_i + 3
                    if y >= max_y - 1 - footer_rows:
                        break
                    check = "✓" if i in chosen else " "
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

                # Status bar (bottom row, right-aligned)
                if status_fn:
                    try:
                        status_text = status_fn(chosen)
                        if status_text:
                            # Right-align on the bottom row
                            sx = max(0, max_x - len(status_text) - 1)
                            sattr = curses.A_DIM
                            if curses.has_colors():
                                sattr |= curses.color_pair(3)
                            stdscr.addnstr(max_y - 1, sx, status_text, max_x - sx - 1, sattr)
                    except curses.error:
                        pass

                stdscr.refresh()
                key = stdscr.getch()

                if searchable:
                    handled, confirm, changed = _handle_active_search_key(curses, key, search)
                    if handled:
                        if changed:
                            scroll_offset = 0
                        if confirm:
                            result_holder[0] = set(chosen)
                            return
                        continue

                if key in {curses.KEY_UP, ord("k")}:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, -1)
                elif key in {curses.KEY_DOWN, ord("j")}:
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, 1)
                elif key == ord(" "):
                    if filtered:
                        chosen.symmetric_difference_update({cursor})
                elif key in {curses.KEY_ENTER, 10, 13}:
                    result_holder[0] = set(chosen)
                    return
                elif searchable and key == ord("/"):
                    search.active = True
                elif key in {27, ord("q")}:
                    result_holder[0] = cancel_returns
                    return

        curses.wrapper(_draw)
        flush_stdin()
        return result_holder[0] if result_holder[0] is not None else cancel_returns

    except KeyboardInterrupt:
        return cancel_returns
    except Exception:
        return _numbered_fallback(title, items, selected, cancel_returns, status_fn)


def curses_radiolist(
    title: str,
    items: List[str],
    selected: int = 0,
    *,
    cancel_returns: int | None = None,
    description: str | None = None,
    searchable: bool = False,
) -> int:
    """Curses single-select radio list. Returns the selected index.

    Args:
        title: Header line displayed above the list.
        items: Display labels for each row.
        selected: Index that starts selected (pre-selected).
        cancel_returns: Returned on ESC/q. Defaults to the original *selected*.
        description: Optional multi-line text shown between the title and
            the item list.  Useful for context that should survive the
            curses screen clear.
        searchable: When true, "/" opens a filter prompt. The returned value
            is always the original item index, not the filtered row position.
    """
    if cancel_returns is None:
        cancel_returns = selected

    if not sys.stdin.isatty():
        return cancel_returns

    desc_lines: list[str] = []
    if description:
        desc_lines = description.splitlines()

    try:
        import curses
        result_holder: list = [None]

        def _draw(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
            cursor = selected
            scroll_offset = 0
            search = _SearchState()

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                filtered = _filter_indices(items, search.query) if searchable else list(range(len(items)))
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                row = 0

                # Header
                try:
                    hattr = curses.A_BOLD
                    if curses.has_colors():
                        hattr |= curses.color_pair(2)
                    stdscr.addnstr(row, 0, title, max_x - 1, hattr)
                    row += 1

                    # Description lines
                    for dline in desc_lines:
                        if row >= max_y - 1:
                            break
                        stdscr.addnstr(row, 0, dline, max_x - 1, curses.A_NORMAL)
                        row += 1

                    if searchable and search.active:
                        help_text = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop search"
                    elif searchable and search.query:
                        help_text = (
                            f"  Filter: {search.query} ({len(filtered)}/{len(items)})  "
                            "↑↓ navigate  ENTER/SPACE select  / edit  ESC cancel"
                        )
                    elif searchable:
                        help_text = "  \u2191\u2193 navigate  ENTER/SPACE select  / search  ESC cancel"
                    else:
                        help_text = "  \u2191\u2193 navigate  ENTER/SPACE select  ESC cancel"
                    stdscr.addnstr(row, 0, help_text, max_x - 1, curses.A_DIM)
                    row += 1
                except curses.error:
                    pass

                # Scrollable item list
                items_start = row + 1
                visible_rows = max(1, max_y - items_start - 1)
                scroll_offset = _scroll_for_cursor(scroll_offset, cursor_pos, visible_rows, len(filtered))

                if searchable and search.query and not filtered:
                    try:
                        stdscr.addnstr(items_start, 0, "  No matches", max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass

                for draw_i, filtered_pos in enumerate(range(scroll_offset, min(len(filtered), scroll_offset + visible_rows))):
                    i = filtered[filtered_pos]
                    y = draw_i + items_start
                    if y >= max_y - 1:
                        break
                    radio = "\u25cf" if i == selected else "\u25cb"
                    arrow = "\u2192" if i == cursor else " "
                    line = f" {arrow} ({radio}) {items[i]}"
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

                if searchable:
                    handled, confirm, changed = _handle_active_search_key(curses, key, search)
                    if handled:
                        if changed:
                            scroll_offset = 0
                        if confirm and filtered:
                            result_holder[0] = cursor
                            return
                        continue

                if key in (curses.KEY_UP, ord("k")):
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, -1)
                elif key in (curses.KEY_DOWN, ord("j")):
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, 1)
                elif key in (ord(" "), curses.KEY_ENTER, 10, 13):
                    if filtered:
                        result_holder[0] = cursor
                        return
                elif searchable and key == ord("/"):
                    search.active = True
                elif key in (27, ord("q")):
                    result_holder[0] = cancel_returns
                    return

        curses.wrapper(_draw)
        flush_stdin()
        return result_holder[0] if result_holder[0] is not None else cancel_returns

    except KeyboardInterrupt:
        return cancel_returns
    except Exception:
        return _radio_numbered_fallback(title, items, selected, cancel_returns)


def _radio_numbered_fallback(
    title: str,
    items: List[str],
    selected: int,
    cancel_returns: int,
) -> int:
    """Text-based numbered fallback for radio selection."""
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Select by number, Enter to confirm.\n", Colors.DIM))

    for i, label in enumerate(items):
        marker = color("(\u25cf)", Colors.GREEN) if i == selected else "(\u25cb)"
        print(f"  {marker} {i + 1:>2}. {label}")
    print()
    try:
        val = input(color(f"  Choice [default {selected + 1}]: ", Colors.DIM)).strip()
        if not val:
            return selected
        idx = int(val) - 1
        if 0 <= idx < len(items):
            return idx
        return selected
    except (ValueError, KeyboardInterrupt, EOFError):
        return cancel_returns


def curses_single_select(
    title: str,
    items: List[str],
    default_index: int = 0,
    *,
    cancel_label: str = "Cancel",
    searchable: bool = False,
) -> int | None:
    """Curses single-select menu. Returns selected index or None on cancel.

    Works inside prompt_toolkit because curses.wrapper() restores the terminal
    safely, unlike simple_term_menu which conflicts with /dev/tty. When
    searchable=True, "/" opens a filter prompt and selection still returns
    the original item index.
    """
    if not sys.stdin.isatty():
        return None

    try:
        import curses
        result_holder: list = [None]

        all_items = list(items) + [cancel_label]
        cancel_idx = len(items)

        def _draw(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
            cursor = min(default_index, len(all_items) - 1)
            scroll_offset = 0
            search = _SearchState()

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                filtered = _filter_indices(all_items, search.query) if searchable else list(range(len(all_items)))
                cursor, cursor_pos = _reconcile_cursor(filtered, cursor)

                try:
                    hattr = curses.A_BOLD
                    if curses.has_colors():
                        hattr |= curses.color_pair(2)
                    stdscr.addnstr(0, 0, title, max_x - 1, hattr)
                    if searchable and search.active:
                        help_text = f"  Search: {search.query}\u258e  BACKSPACE edit  Ctrl+U clear  ESC stop search"
                    elif searchable and search.query:
                        help_text = (
                            f"  Filter: {search.query} ({len(filtered)}/{len(all_items)})  "
                            "↑↓ navigate  ENTER confirm  / edit  ESC/q cancel"
                        )
                    elif searchable:
                        help_text = "  ↑↓ navigate  ENTER confirm  / search  ESC/q cancel"
                    else:
                        help_text = "  ↑↓ navigate  ENTER confirm  ESC/q cancel"
                    stdscr.addnstr(1, 0, help_text, max_x - 1, curses.A_DIM)
                except curses.error:
                    pass

                visible_rows = max(1, max_y - 3)
                scroll_offset = _scroll_for_cursor(scroll_offset, cursor_pos, visible_rows, len(filtered))

                if searchable and search.query and not filtered:
                    try:
                        stdscr.addnstr(3, 0, "  No matches", max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass

                for draw_i, filtered_pos in enumerate(range(scroll_offset, min(len(filtered), scroll_offset + visible_rows))):
                    i = filtered[filtered_pos]
                    y = draw_i + 3
                    if y >= max_y - 1:
                        break
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow} {all_items[i]}"
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

                if searchable:
                    handled, confirm, changed = _handle_active_search_key(curses, key, search)
                    if handled:
                        if changed:
                            scroll_offset = 0
                        if confirm and filtered:
                            result_holder[0] = cursor
                            return
                        continue

                if key in (curses.KEY_UP, ord("k")):
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, -1)
                elif key in (curses.KEY_DOWN, ord("j")):
                    cursor = _move_filtered_cursor(filtered, cursor, cursor_pos, 1)
                elif key in (curses.KEY_ENTER, 10, 13):
                    if filtered:
                        result_holder[0] = cursor
                        return
                elif searchable and key == ord("/"):
                    search.active = True
                elif key in (27, ord("q")):
                    result_holder[0] = None
                    return

        curses.wrapper(_draw)
        flush_stdin()
        if result_holder[0] is not None and result_holder[0] >= cancel_idx:
            return None
        return result_holder[0]

    except KeyboardInterrupt:
        return None
    except Exception:
        all_items = list(items) + [cancel_label]
        cancel_idx = len(items)
        return _numbered_single_fallback(title, all_items, cancel_idx)


def _numbered_single_fallback(
    title: str,
    items: List[str],
    cancel_idx: int,
) -> int | None:
    """Text-based numbered fallback for single-select."""
    print(f"\n  {title}\n")
    for i, label in enumerate(items, 1):
        print(f"  {i}. {label}")
    print()
    try:
        val = input(f"  Choice [1-{len(items)}]: ").strip()
        if not val:
            return None
        idx = int(val) - 1
        if 0 <= idx < len(items) and idx < cancel_idx:
            return idx
        if idx == cancel_idx:
            return None
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return None


def _numbered_fallback(
    title: str,
    items: List[str],
    selected: Set[int],
    cancel_returns: Set[int],
    status_fn: Optional[Callable[[Set[int]], str]] = None,
) -> Set[int]:
    """Text-based toggle fallback for terminals without curses."""
    chosen = set(selected)
    print(color(f"\n  {title}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

    while True:
        for i, label in enumerate(items):
            marker = color("[✓]", Colors.GREEN) if i in chosen else "[ ]"
            print(f"  {marker} {i + 1:>2}. {label}")
        if status_fn:
            status_text = status_fn(chosen)
            if status_text:
                print(color(f"\n  {status_text}", Colors.DIM))
        print()
        try:
            val = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
            if not val:
                break
            idx = int(val) - 1
            if 0 <= idx < len(items):
                chosen.symmetric_difference_update({idx})
        except (ValueError, KeyboardInterrupt, EOFError):
            return cancel_returns
        print()

    return chosen
