"""Interactive picker for ``hermes sessions browse``: curses UI with live search filtering and ``d``
delete-with-confirmation; numbered-list fallback when curses is unavailable (Windows, etc.)."""

from typing import Optional
from wcwidth import wcwidth, wcswidth

from hermes_cli.timefmt import relative_time as _relative_time
from hermes_state_errors import SessionActiveWriteGuardError


class SessionBrowserUnavailable(RuntimeError):
    """The classic picker could not acquire a usable curses terminal."""


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
    """One printable line clipped by terminal cells, not Unicode code points."""
    text = "".join(" " if char.isspace() else char for char in text if char.isspace() or wcwidth(char) >= 0)
    if n <= 0:
        return ""
    if wcswidth(text) <= n:
        return text
    prefix = ""
    for char in text:
        if wcswidth(prefix + char) > n - 1:
            break
        prefix += char
    return prefix.rstrip("\u200d") + "…"


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
        f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active'), session_id=s['id']):<10}  "
        f"{s.get('source', '')[:6]:<5} {sid}"
    )


def _format_compact_row(s: dict, width: int) -> str:
    """Keep the label and activity visible; selected-row details carry the ID/preview."""
    age = _clip(_relative_time(s.get("last_active"), session_id=s["id"]), 12)
    label_width = width - wcswidth(age) - 2
    if label_width < 12:
        return _clip(_label(s), width)
    label = _clip(_label(s), label_width)
    return label + " " * (label_width - wcswidth(label) + 2) + age


class _CursesBrowser:
    """State + render loop for the curses picker. ``run`` is the wrapper target."""

    def __init__(self, curses, sessions, delete_fn, *, keyboard_navigation=False):
        self.curses, self.sessions, self.delete_fn = curses, sessions, delete_fn  # delete_fn None => no delete
        self.keyboard_navigation = keyboard_navigation
        self.search_editing = False
        self.page_size = 1
        self.result = None
        self.cursor = self.scroll = 0
        self.search = ""
        self.confirm_delete = None  # session dict pending y/n confirmation
        self.flash = ""  # one-frame notice (e.g. "Deleted.")
        self.filtered = list(sessions)
        self.row_numbers = {session["id"]: index for index, session in enumerate(sessions, 1)}

    def _pair(self, n, fallback=0):
        return self.curses.color_pair(n) if self.curses.has_colors() else fallback

    def _status_attr(self, status):
        pair = {"complete": 1, "interrupted": 2, "error": 5, "empty": 4}.get(status or "")
        return self._pair(pair, self.curses.A_NORMAL) if pair else self.curses.A_NORMAL

    def _put(self, stdscr, y, x, text, n, attr):
        try:
            stdscr.addstr(y, x, _clip(text, n), attr)
        except self.curses.error:
            pass

    def _refilter(self, reset_cursor=True):
        self.filtered = [s for s in self.sessions if _match(s, self.search)] if self.search else list(self.sessions)
        if reset_cursor:
            self.cursor = self.scroll = 0

    def _draw(self, stdscr, max_y, max_x):
        c = self.curses
        if self.keyboard_navigation:
            header = f"  Search: {self.search}█" if self.search_editing else "  Sessions  ↑↓/j/k move  / search"
            header_attr = c.A_BOLD | self._pair(2)
        elif self.search:
            header, header_attr = f"  Browse sessions — filter: {self.search}█", c.A_BOLD | self._pair(3)
        else:
            header = "  Browse sessions — ↑↓ navigate  Enter select  Type to filter  Esc quit"
            header_attr = c.A_BOLD | self._pair(2)
        self._put(stdscr, 0, 0, header, max_x - 1, header_attr)
        name_width = max(20, max_x - _FIXED_COLS)
        col_header = (
            f"   {'Title / Preview':<{name_width}}  {'Stat':<5}  {'Msgs':>5}  {'Active':<10}  {'Src':<5} {'ID'}"
        )
        if self.keyboard_navigation:
            col_header = "  Enter resume  PgUp/Dn  g/G  Esc/q cancel"
        dim_attr = c.A_DIM if self.keyboard_navigation else self._pair(4, c.A_DIM)
        self._put(stdscr, 1, 0, col_header, max_x - 1, dim_attr)
        visible_rows = max(max_y - (7 if self.keyboard_navigation else 4), 1)
        self.page_size = visible_rows
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
                prefix = " → " if selected else "   "
                if self.keyboard_navigation:
                    number = self.row_numbers[s["id"]]
                    prefix += f"{number:>{len(str(len(self.sessions)))}}  "
                row_text = (_format_compact_row(s, max_x - len(prefix) - 1) if self.keyboard_navigation
                            else _format_row(s, max_x - 3))
                row = prefix + row_text
                try:
                    self._put(stdscr, y, 0, row, max_x - 1, c.A_BOLD | self._pair(1) if selected else c.A_NORMAL)
                    if not self.keyboard_navigation and not selected and tag_x + 5 < max_x - 1:
                        status = s.get("_status")
                        stdscr.addnstr(y, tag_x, f"{_session_status_tag(status):<5}", 5, self._status_attr(status))
                except c.error:
                    pass
            if self.keyboard_navigation:
                selected = filtered[self.cursor]
                detail = f"  {selected['id']} · {_msgs_str(selected)} msgs · {selected.get('source') or 'cli'}"
                self._put(stdscr, max_y - 3, 0, detail, max_x - 1, dim_attr)
                self._put(stdscr, max_y - 2, 0, "  " + (selected.get("preview") or ""), max_x - 1, c.A_DIM)
        footer_attr = dim_attr
        if self.confirm_delete is not None:
            footer = f"  Delete session '{_clip(_label(self.confirm_delete), 40)}'? [y/N]"
            footer_attr = c.A_BOLD | self._pair(5)
        elif self.flash:
            footer = f"  {self.flash}"
            self.flash = ""
        else:
            footer = f"  {self.cursor + 1 if filtered else 0}/{len(filtered)} sessions"
            if filtered and len(filtered) < len(self.sessions):
                footer += f" (filtered from {len(self.sessions)})"
            if self.delete_fn is not None and not self.search:
                footer += "   d delete"
        self._put(stdscr, max_y - 1, 0, footer, max_x - 1, footer_attr)

    def _read_keyboard_key(self, stdscr):
        """Consume a whole terminal key before treating a lone Escape as cancellation."""
        from hermes_cli.curses_ui import _parse_csi_numbers, _parse_csi_u_key, _read_csi_tail

        key = stdscr.get_wch()
        if key != "\x1b":
            return key
        stdscr.timeout(60)
        try:
            introducer = stdscr.getch()
            if introducer == -1:
                return 27
            if introducer not in (ord("["), ord("O")):
                return -1
            raw, final = _read_csi_tail(stdscr)
        finally:
            stdscr.timeout(-1)
        enhanced = _parse_csi_u_key(raw)
        if enhanced is not None and enhanced[2] == 3:
            return -1  # release events must neither navigate nor cancel
        c = self.curses
        navigation = {ord("A"): c.KEY_UP, ord("B"): c.KEY_DOWN,
                      ord("H"): c.KEY_HOME, ord("F"): c.KEY_END}
        if final in navigation:
            return navigation[final]
        params = _parse_csi_numbers(raw)
        if final in (ord("u"), ord("~")) and len(params) >= 3 and params[0] == 27:
            codepoint, modifier = params[2], params[1]  # modifyOtherKeys
        elif final == ord("u") and enhanced is not None:
            codepoint, modifier, _ = enhanced
        elif final == ord("~"):
            return {1: c.KEY_HOME, 4: c.KEY_END, 5: c.KEY_PPAGE,
                    6: c.KEY_NPAGE, 7: c.KEY_HOME, 8: c.KEY_END}.get(params[0], -1)
        else:
            return -1
        bits = max(1, modifier) - 1
        if codepoint in (99, 67) and bits & 4:
            return 3
        if bits & 1 and 97 <= codepoint <= 122:
            codepoint -= 32  # CSI-u may report Shift+g as lowercase g + Shift
        if 0 <= codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF:
            return chr(codepoint)  # text stays distinct from integer curses KEY_* codes
        return -1

    def _keyboard_key(self, key):
        """Handle explicit search editing; None lets a shared navigation key through."""
        c = self.curses
        if isinstance(key, str):
            if self.search_editing and wcwidth(key) >= 0:
                self.search += key
                self._refilter()
            return False
        if key == 27 or (key == ord("q") and not self.search_editing):
            return True
        if self.search_editing:
            if key in {c.KEY_ENTER, 10, 13}:
                self.search_editing = False
                return None  # Enter always resumes the highlighted filtered row
            elif key in {c.KEY_BACKSPACE, 127, 8}:
                self.search = self.search[:-1]
                self._refilter()
            elif 32 <= key <= 126:
                self.search += chr(key)
                self._refilter()
            else:
                return None
            return False
        if key == ord("/"):
            self.search_editing = True
            return False
        if 32 <= key <= 126 and chr(key) not in "jkgG":
            return False
        return None

    def _navigate(self, key) -> bool:
        c = self.curses
        if key in (c.KEY_UP, c.KEY_DOWN):
            if self.filtered:
                self.cursor = (self.cursor + (1 if key == c.KEY_DOWN else -1)) % len(self.filtered)
        elif key in (c.KEY_PPAGE, c.KEY_NPAGE):
            delta = self.page_size if key == c.KEY_NPAGE else -self.page_size
            self.cursor = max(0, min(len(self.filtered) - 1, self.cursor + delta))
        elif key == c.KEY_HOME:
            self.cursor = 0
        elif key == c.KEY_END:
            self.cursor = max(0, len(self.filtered) - 1)
        else:
            return False
        return True

    def _handle_key(self, key) -> bool:
        """Apply one keypress; return True when the picker should exit."""
        c = self.curses
        if isinstance(key, str):
            if ord(key) >= 128:
                return bool(self._keyboard_key(key)) if self.keyboard_navigation else False
            key = ord(key)  # ASCII commands/control keys share the standalone dispatch
        if key == 3:
            return True
        if self.keyboard_navigation:
            handled = self._keyboard_key(key)
            if handled is not None:
                return handled
            key = {ord("j"): c.KEY_DOWN, ord("k"): c.KEY_UP,
                   ord("g"): c.KEY_HOME, ord("G"): c.KEY_END}.get(key, key)
        if self.confirm_delete is not None:  # y/n confirmation mode — only an explicit 'y' deletes
            target, self.confirm_delete = self.confirm_delete, None
            if key not in {ord("y"), ord("Y")}:
                return False
            try:
                ok = self.delete_fn(target["id"])
            except SessionActiveWriteGuardError:
                self.flash = "Session is active (a live turn owns it); try again when it finishes."
                return False
            if not ok:
                self.flash = "Delete failed."
                return False
            self.sessions[:] = [s for s in self.sessions if s["id"] != target["id"]]
            self._refilter(reset_cursor=False)
            self.flash = "Deleted."
            return not self.sessions
        if self._navigate(key):
            return False
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
            if max_y < (8 if self.keyboard_navigation else 5) or max_x < 40:
                try:
                    stdscr.addstr(0, 0, "Terminal too small")
                except c.error:
                    pass
                stdscr.refresh()
                stdscr.getch()
                return
            self._draw(stdscr, max_y, max_x)
            stdscr.refresh()
            key = self._read_keyboard_key(stdscr) if self.keyboard_navigation else stdscr.getch()
            if self._handle_key(key):
                return


def _fallback_picker(sessions: list) -> Optional[str]:
    """Numbered list (Windows without curses, etc.). Same columns, no delete."""
    print("\n  Browse sessions  (enter number to resume, q to cancel)\n")
    for i, s in enumerate(sessions):
        print(
            f"  {i + 1:>3}. {_clip(_label(s), 50):<50}  {_session_status_tag(s.get('_status')):<5}  "
            f"{_msgs_str(s):>5}  {_relative_time(s.get('last_active'), session_id=s['id']):<10}  {s.get('source', '')[:6]}"
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


def _session_browse_picker(sessions: list, session_db=None, *, keyboard_navigation=False) -> Optional[str]:
    """Curses session browser with live search; returns the selected session ID, or None if cancelled.

    With *session_db*: shows lifecycle status / message count per row, and ``d`` (while the filter is
    empty) prompts y/n and deletes via ``SessionDB.delete_session``. Keyboard mode raises
    ``SessionBrowserUnavailable`` when curses cannot run, never starting a numbered input prompt.
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
            return bool(session_db.delete_session(session_id, sessions_dir=sessions_dir, exclude_active_write_guards=True))
        except SessionActiveWriteGuardError:
            raise  # the browser tells the user the session is busy instead of a generic failure
        except Exception:
            return False
    try:  # curses first; any failure (no curses module, odd terminal) falls back
        import curses
        browser = _CursesBrowser(
            curses, sessions, _delete_session if session_db is not None else None,
            keyboard_navigation=keyboard_navigation)
        curses.wrapper(browser.run)
        return browser.result
    except KeyboardInterrupt:
        return None
    except Exception as exc:
        if keyboard_navigation:
            raise SessionBrowserUnavailable("Session picker unavailable") from exc
        return _fallback_picker(sessions)
    finally:
        if keyboard_navigation:
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()  # queued releases must not become input in the restored CLI prompt
