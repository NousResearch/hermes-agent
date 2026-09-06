"""Interactive picker for ``hermes sessions browse``: curses UI with live search filtering and ``d``
delete-with-confirmation; numbered-list fallback when curses is unavailable (Windows, etc.)."""

from typing import Optional

from hermes_cli.timefmt import relative_time as _relative_time


def _session_status_tag(status: Optional[str]) -> str:
    """Short fixed-width tag for a session lifecycle status."""
    return {"complete": "done", "interrupted": "intr", "error": "err", "empty": "empty"}.get(status or "", "-")


def _annotate_session_statuses(sessions: list, session_db) -> None:
    """Attach ``_status`` per row via one indexed lookup (never a transcript scan); on failure rows render '-'."""
    if session_db is None or not sessions:
        return
    try:
        statuses = session_db.session_lifecycle_statuses([s.get("id") for s in sessions])
    except Exception:
        return
    for s in sessions:
        s["_status"] = statuses.get(s.get("id"), "")


def _label(s: dict) -> str:
    """Title, else preview, else id."""
    return (s.get("title") or "").strip() or (s.get("preview") or "").strip() or s["id"]


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def _msgs_str(s: dict) -> str:
    msgs = s.get("message_count")
    return str(msgs) if isinstance(msgs, int) else "-"


def _match(s: dict, query: str) -> bool:
    """Case-insensitive substring match over title / preview / id / source."""
    q = query.lower()
    return (
        q in (s.get("title") or "").lower()
        or q in (s.get("preview") or "").lower()
        or q in s.get("id", "").lower()
        or q in (s.get("source") or "").lower()
    )


# Layout: [arrow 3] [title/preview flexible] [status 5] [msgs 5] [active 12] [src 6] [id 18]
_FIXED_COLS = 3 + 5 + 2 + 5 + 2 + 12 + 6 + 18 + 6


def _format_row(s: dict, max_x: int) -> str:
    name_width = max(20, max_x - _FIXED_COLS)
    sid = s["id"][:18]
    name = ((s.get("title") or "").strip() or (s.get("preview") or "").strip())[:name_width] or sid
    return (
        f"{name:<{name_width}}  {_session_status_tag(s.get('_status')):<5}  "
        f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active')):<10}  "
        f"{s.get('source', '')[:6]:<5} {sid}"
    )


class _CursesBrowser:
    """State + render loop for the curses picker. ``run`` is the wrapper target."""

    def __init__(self, curses, sessions, delete_fn):
        self.curses, self.sessions, self.delete_fn = curses, sessions, delete_fn  # delete_fn None => no delete
        self.result = None
        self.cursor = self.scroll = 0
        self.search = ""
        self.confirm_delete = None  # session dict pending y/n confirmation
        self.flash = ""  # one-frame notice (e.g. "Deleted.")
        self.filtered = list(sessions)

    def _pair(self, n, fallback=0):
        return self.curses.color_pair(n) if self.curses.has_colors() else fallback

    def _status_attr(self, status):
        pair = {"complete": 1, "interrupted": 2, "error": 5, "empty": 4}.get(status or "")
        return self._pair(pair, self.curses.A_NORMAL) if pair else self.curses.A_NORMAL

    def _put(self, stdscr, y, x, text, n, attr):
        try:
            stdscr.addnstr(y, x, text, n, attr)
        except self.curses.error:
            pass

    def _refilter(self, reset_cursor=True):
        self.filtered = [s for s in self.sessions if _match(s, self.search)] if self.search else list(self.sessions)
        if reset_cursor:
            self.cursor = self.scroll = 0

    def _draw(self, stdscr, max_y, max_x):
        c = self.curses
        if self.search:
            header, header_attr = f"  Browse sessions — filter: {self.search}█", c.A_BOLD | self._pair(3)
        else:
            header = "  Browse sessions — ↑↓ navigate  Enter select  Type to filter  Esc quit"
            header_attr = c.A_BOLD | self._pair(2)
        self._put(stdscr, 0, 0, header, max_x - 1, header_attr)
        name_width = max(20, max_x - _FIXED_COLS)
        col_header = (
            f"   {'Title / Preview':<{name_width}}  {'Stat':<5}  {'Msgs':>5}  {'Active':<10}  {'Src':<5} {'ID'}"
        )
        self._put(stdscr, 1, 0, col_header, max_x - 1, self._pair(4, c.A_DIM))
        visible_rows = max(max_y - 4, 1)  # header + col header + blank + footer
        filtered = self.filtered
        if not filtered:
            self._put(stdscr, 3, 0, "  No sessions match the filter.", max_x - 1, c.A_DIM)
        else:
            self.cursor = max(min(self.cursor, len(filtered) - 1), 0)
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            elif self.cursor >= self.scroll + visible_rows:
                self.scroll = self.cursor - visible_rows + 1
            tag_x = 3 + max(20, (max_x - 3) - _FIXED_COLS) + 2
            for draw_i, i in enumerate(range(self.scroll, min(len(filtered), self.scroll + visible_rows))):
                y = draw_i + 3
                if y >= max_y - 1:
                    break
                s, selected = filtered[i], i == self.cursor
                row = (" → " if selected else "   ") + _format_row(s, max_x - 3)
                try:
                    stdscr.addnstr(y, 0, row, max_x - 1, c.A_BOLD | self._pair(1) if selected else c.A_NORMAL)
                    if not selected and tag_x + 5 < max_x - 1:  # recolor the status tag in place
                        status = s.get("_status")
                        stdscr.addnstr(y, tag_x, f"{_session_status_tag(status):<5}", 5, self._status_attr(status))
                except c.error:
                    pass
        footer_attr = self._pair(4, c.A_DIM)
        if self.confirm_delete is not None:
            footer = f"  Delete session '{_clip(_label(self.confirm_delete), 40)}'? [y/N]"
            footer_attr = c.A_BOLD | self._pair(5)
        elif self.flash:
            footer = f"  {self.flash}"
            self.flash = ""
        else:
            footer = f"  {self.cursor + 1 if filtered else 0}/{len(filtered) or len(self.sessions)} sessions"
            if filtered and len(filtered) < len(self.sessions):
                footer += f" (filtered from {len(self.sessions)})"
            if self.delete_fn is not None and not self.search:
                footer += "   d delete"
        self._put(stdscr, max_y - 1, 0, footer, max_x - 1, footer_attr)

    def _handle_key(self, key) -> bool:
        """Apply one keypress; return True when the picker should exit."""
        c = self.curses
        if self.confirm_delete is not None:  # y/n confirmation mode — only an explicit 'y' deletes
            target, self.confirm_delete = self.confirm_delete, None
            if key not in {ord("y"), ord("Y")}:
                return False
            if not self.delete_fn(target["id"]):
                self.flash = "Delete failed."
                return False
            self.sessions[:] = [s for s in self.sessions if s["id"] != target["id"]]
            self._refilter(reset_cursor=False)
            self.flash = "Deleted."
            return not self.sessions
        if key in (c.KEY_UP, c.KEY_DOWN):
            if self.filtered:
                self.cursor = (self.cursor + (1 if key == c.KEY_DOWN else -1)) % len(self.filtered)
        elif key in {c.KEY_ENTER, 10, 13}:
            if self.filtered:
                self.result = self.filtered[self.cursor]["id"]
            return True
        elif key == 27 and not self.search:  # Esc: first clears the search, second exits
            return True
        elif key == 27:
            self.search = ""
            self._refilter()
        elif key in {c.KEY_BACKSPACE, 127, 8}:
            if self.search:
                self.search = self.search[:-1]
                self._refilter()
        elif key == ord("q") and not self.search:
            return True
        elif key == ord("d") and not self.search and self.delete_fn is not None and self.filtered:
            # 'd' deletes only when the filter is empty; mid-search it types into the query.
            self.confirm_delete = self.filtered[self.cursor]
        elif 32 <= key <= 126:
            self.search += chr(key)
            self._refilter()
        return False

    def run(self, stdscr):
        c = self.curses
        c.curs_set(0)
        if c.has_colors():
            c.start_color()
            c.use_default_colors()
            # 1 selected, 2 header, 3 search, 4 dim, 5 error/delete
            palette = (c.COLOR_GREEN, c.COLOR_YELLOW, c.COLOR_CYAN, 8 if c.COLORS > 8 else c.COLOR_WHITE, c.COLOR_RED)
            for n, color in enumerate(palette, 1):
                c.init_pair(n, color, -1)
        while True:
            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 5 or max_x < 40:
                try:
                    stdscr.addstr(0, 0, "Terminal too small")
                except c.error:
                    pass
                stdscr.refresh()
                stdscr.getch()
                return
            self._draw(stdscr, max_y, max_x)
            stdscr.refresh()
            if self._handle_key(stdscr.getch()):
                return


def _fallback_picker(sessions: list) -> Optional[str]:
    """Numbered list (Windows without curses, etc.). Same columns, no delete."""
    print("\n  Browse sessions  (enter number to resume, q to cancel)\n")
    for i, s in enumerate(sessions):
        print(
            f"  {i + 1:>3}. {_clip(_label(s), 50):<50}  {_session_status_tag(s.get('_status')):<5}  "
            f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active')):<10}  {s.get('source', '')[:6]}"
        )
    while True:
        try:
            val = input(f"\n  Select [1-{len(sessions)}]: ").strip()
            if not val or val.lower() in {"q", "quit", "exit"}:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
            print(f"  Invalid selection. Enter 1-{len(sessions)} or q to cancel.")
        except ValueError:
            print("  Invalid input. Enter a number or q to cancel.")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def _session_browse_picker(sessions: list, session_db=None) -> Optional[str]:
    """Curses session browser with live search; returns the selected session ID, or None if cancelled.

    With *session_db*: shows lifecycle status / message count per row, and ``d`` (while the filter is
    empty) prompts y/n and deletes via ``SessionDB.delete_session``.
    """
    if not sessions:
        print("No sessions found.")
        return None
    _annotate_session_statuses(sessions, session_db)

    def _delete_session(session_id: str) -> bool:
        try:
            from hermes_cli.sessions_cmd import get_hermes_home
            sessions_dir = get_hermes_home() / "sessions"
        except Exception:
            sessions_dir = None
        try:
            return bool(session_db.delete_session(session_id, sessions_dir=sessions_dir))
        except Exception:
            return False
    try:  # curses first; any failure (no curses module, odd terminal) falls back
        import curses
        browser = _CursesBrowser(curses, sessions, _delete_session if session_db is not None else None)
        curses.wrapper(browser.run)
        return browser.result
    except Exception:
        return _fallback_picker(sessions)
