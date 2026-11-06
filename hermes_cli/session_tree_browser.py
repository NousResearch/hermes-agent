"""The classic CLI's ``/tree`` browser: the session family full-screen, with a preview pane.

Same rows, keys and preview as the Ink overlay (cells come from :mod:`hermes_cli.session_tree`),
drawn with curses: ↑/↓ move, ←/→ fold and unfold (← on a leaf jumps to its parent), Enter
switches, ``/`` filters titles, ``c`` / ``s`` show compressed segments and subagent runs, Esc or
``q`` closes. Every view change re-reads the family through ``SessionDB.session_family``, so the
fold rules are the shared builder's, not a second copy.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from agent.i18n import t
from hermes_cli.curses_ui import _addnstr
from hermes_cli.session_tree import (
    HERE_GLYPH, count_label, heaviest, hint_pairs, kind_label, load_family, row_glyph,
    messages_label, node_line, node_title, preview_lines, row_notes, tree_header, weight_bar,
)
from hermes_state_family import KIND_BRANCH, KIND_COMPRESSION, KIND_ROOT, KIND_SUBAGENT, SessionFamily

_RULE = "┄"
_PREVIEW_ROWS = 3
_LABEL_W = 9
_CHROME_ROWS = 3 + 1 + 1 + _PREVIEW_ROWS + 1 + 1  # header+rule+gap, gap, rule, preview, rule, keys
# Colour pair numbers (kept clear of curses_ui's shared 1-4); +_ACTIVE is the selected-row twin.
_ACCENT, _MUTED, _TEXT, _ROOT, _BRANCH, _SQUASH, _SUB = range(10, 17)
_ACTIVE = 10
_SELECTED_BG = 237
_KIND_PAIR = {KIND_ROOT: _ROOT, KIND_BRANCH: _BRANCH, KIND_COMPRESSION: _SQUASH, KIND_SUBAGENT: _SUB}


class _TreeBrowser:
    """State + render loop. ``run`` is the ``curses.wrapper`` target; ``result`` is the pick."""

    def __init__(self, curses: Any, db: Any, session_id: str, show_all: bool) -> None:
        self.curses, self.db, self.session_id = curses, db, session_id
        self.show_compressions = self.show_subagents = show_all
        self.folded: set[str] = set()
        self.query, self.searching = "", False
        self.cursor_id: Optional[str] = session_id
        self.scroll = 0
        self.rich = False
        self.result: Optional[str] = None
        self.rows: list[dict[str, Any]] = []
        self.family: SessionFamily = SessionFamily(nodes=[], root_id=session_id)
        self.reload()

    # ------------------------------------------------------------------ state
    def reload(self) -> None:
        self.family = load_family(self.db, self.session_id, show_compressions=self.show_compressions,
                                  show_subagents=self.show_subagents, folded=tuple(sorted(self.folded)))
        self.refilter()

    def refilter(self) -> None:
        needle = self.query.strip().lower()
        nodes = self.family.nodes
        self.rows = [n for n in nodes if not needle or needle in node_title(n).lower()]
        ids = [row["session_id"] for row in self.rows]
        if self.cursor_id not in ids:
            self.cursor_id = ids[0] if ids else None

    @property
    def cursor(self) -> int:
        ids = [row["session_id"] for row in self.rows]
        return ids.index(self.cursor_id) if self.cursor_id in ids else 0

    def move(self, delta: int) -> None:
        if self.rows:
            self.cursor_id = self.rows[max(0, min(len(self.rows) - 1, self.cursor + delta))]["session_id"]

    def fold(self, close: bool) -> None:
        if not self.rows:
            return
        row = self.rows[self.cursor]
        sid = row["session_id"]
        if close and row.get("has_children") and sid not in self.folded:
            self.folded.add(sid)
        elif close and row.get("parent_session_id"):
            parent = row["parent_session_id"]
            if any(r["session_id"] == parent for r in self.rows):
                self.cursor_id = parent
            return
        elif not close and sid in self.folded:
            self.folded.discard(sid)
        else:
            return
        self.reload()

    # ------------------------------------------------------------------ input
    def read_key(self, stdscr) -> str:
        """One key as a name: arrows decode from plain or escape-sequence form; bare Esc is ``esc``."""
        key = stdscr.getch()
        c = self.curses
        names = {c.KEY_UP: "up", c.KEY_DOWN: "down", c.KEY_LEFT: "left", c.KEY_RIGHT: "right",
                 c.KEY_ENTER: "enter", 10: "enter", 13: "enter", c.KEY_BACKSPACE: "backspace",
                 127: "backspace", 8: "backspace", c.KEY_PPAGE: "pageup", c.KEY_NPAGE: "pagedown"}
        if key in names:
            return names[key]
        if key == 27:
            stdscr.nodelay(True)
            try:
                seq = []
                while (nxt := stdscr.getch()) != -1 and len(seq) < 8:
                    seq.append(nxt)
            finally:
                stdscr.nodelay(False)
            if not seq:
                return "esc"
            return {ord("A"): "up", ord("B"): "down", ord("C"): "right", ord("D"): "left"}.get(seq[-1], "")
        return chr(key) if 0 < key < 0x110000 else ""

    def handle(self, key: str) -> bool:
        """Apply one key; ``True`` when the browser should close."""
        if self.searching:
            if key in ("esc",):
                self.query, self.searching = "", False
            elif key == "enter":
                self.searching = False
            elif key == "backspace":
                self.query = self.query[:-1]
            elif key in ("up", "down"):
                self.move(-1 if key == "up" else 1)
                return False
            elif len(key) == 1 and key.isprintable():
                self.query += key
            self.refilter()
            return False
        if key in ("esc", "q"):
            return True
        if key in ("up", "k"):
            self.move(-1)
        elif key in ("down", "j"):
            self.move(1)
        elif key in ("pageup", "pagedown"):
            self.move(-10 if key == "pageup" else 10)
        elif key in ("left", "right"):
            self.fold(close=key == "left")
        elif key == "/":
            self.searching = True
        elif key in ("c", "C"):
            self.show_compressions = not self.show_compressions
            self.reload()
        elif key in ("s", "S"):
            self.show_subagents = not self.show_subagents
            self.reload()
        elif key == "enter" and self.rows:
            row = self.rows[self.cursor]
            if not row.get("is_current"):
                self.result = row["session_id"]
                return True
        return False

    # ------------------------------------------------------------------ drawing
    def attr(self, pair: int, extra: int = 0, active: bool = False) -> int:
        """A colour pair (its selected-row twin when *active*) plus *extra* attributes."""
        c = self.curses
        if not c.has_colors():
            return extra | (c.A_REVERSE if active else 0)
        if active and not self.rich:
            return c.color_pair(pair) | c.A_REVERSE | extra
        return c.color_pair(pair + _ACTIVE if active else pair) | extra

    def init_colors(self) -> None:
        c = self.curses
        try:
            c.curs_set(0)
        except c.error:
            pass
        self.rich = False
        if not c.has_colors():
            return
        c.start_color()
        c.use_default_colors()
        self.rich = c.COLORS >= 256
        palette = ((_ACCENT, 178, c.COLOR_YELLOW), (_MUTED, 246, c.COLOR_WHITE), (_TEXT, 255, -1),
                   (_ROOT, 220, c.COLOR_YELLOW), (_BRANCH, 117, c.COLOR_CYAN),
                   (_SQUASH, 176, c.COLOR_MAGENTA), (_SUB, 114, c.COLOR_GREEN))
        for pair, rich_fg, basic_fg in palette:
            c.init_pair(pair, rich_fg if self.rich else basic_fg, -1)
            if self.rich:  # the selected row: same hue on a raised slate background
                c.init_pair(pair + _ACTIVE, rich_fg, _SELECTED_BG)

    def draw(self, stdscr) -> None:
        c = self.curses
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        width = max_x - 1
        muted = self.attr(_MUTED)
        # One block anchored to the bottom edge, like the TUI overlay above its status bar:
        # header, rule, rows, rule, preview card, rule, keys.
        list_rows = max(1, min(max(1, len(self.rows)), max_y - _CHROME_ROWS))
        top_y = max(0, max_y - _CHROME_ROWS - list_rows)
        header = f"  {HERE_GLYPH} {tree_header(self.family)}"
        count = f"{count_label(self.family)} "
        _addnstr(stdscr, top_y, 0, header, width - len(count) - 1, self.attr(_ACCENT, c.A_BOLD))
        _addnstr(stdscr, top_y, max(0, width - len(count)), count, len(count), muted)
        _addnstr(stdscr, top_y + 1, 1, _RULE * (width - 2), width - 2, muted)

        list_top = top_y + 3
        if self.cursor < self.scroll:
            self.scroll = self.cursor
        elif self.cursor >= self.scroll + list_rows:
            self.scroll = self.cursor - list_rows + 1
        heaviest_count = heaviest(self.rows)
        widest = max((len(r.get("guide") or "") + len(node_title(r)) for r in self.rows), default=0)
        title_w = max(24, min(widest + 4, width - 56))
        for offset, row in enumerate(self.rows[self.scroll:self.scroll + list_rows]):
            self.draw_row(stdscr, list_top + offset, row, self.scroll + offset == self.cursor,
                          width, title_w, heaviest_count)
        if not self.rows:
            _addnstr(stdscr, list_top, 3, t("cli.tree.no_matches"), width - 3, muted)

        rule_y = list_top + list_rows + 1
        _addnstr(stdscr, rule_y, 1, _RULE * (width - 2), width - 2, muted)
        if self.rows:
            self.draw_preview(stdscr, rule_y + 1, self.rows[self.cursor], width)
        hint_y = rule_y + _PREVIEW_ROWS + 1
        _addnstr(stdscr, hint_y, 1, _RULE * (width - 2), width - 2, muted)
        if self.searching:
            _addnstr(stdscr, hint_y + 1, 2, f"/ {self.query}▌", width - 2, self.attr(_ACCENT, c.A_BOLD))
        else:
            x = 2
            for key, action in hint_pairs():
                _addnstr(stdscr, hint_y + 1, x, key, width - x, self.attr(_ACCENT, c.A_BOLD))
                x += len(key) + 1
                _addnstr(stdscr, hint_y + 1, x, action, width - x, muted)
                x += len(action) + 3
        stdscr.refresh()

    def draw_preview(self, stdscr, y: int, row: dict[str, Any], width: int) -> None:
        c = self.curses
        (head, facts), *exchange = preview_lines(row)[:_PREVIEW_ROWS]
        glyph, _, name = head.partition(" ")
        _addnstr(stdscr, y, 3, glyph, 1, self.attr(_KIND_PAIR.get(row["kind"], _MUTED), c.A_BOLD))
        _addnstr(stdscr, y, 5, name, width - 5, self.attr(_TEXT, c.A_BOLD))
        _addnstr(stdscr, y, 5 + len(name) + 3, facts, max(0, width - len(name) - 8), self.attr(_MUTED))
        for i, (label, text) in enumerate(exchange, start=1):
            if label:
                mine = label == t("cli.tree.you")
                _addnstr(stdscr, y + i, 2, f"{label:>{_LABEL_W}} {'›' if mine else '‹'}", _LABEL_W + 2,
                         self.attr(_BRANCH if mine else _ACCENT, c.A_BOLD))
            _addnstr(stdscr, y + i, _LABEL_W + 6, text, width - _LABEL_W - 7,
                     self.attr(_TEXT if label else _MUTED))

    def draw_row(self, stdscr, y: int, row: dict[str, Any], active: bool, width: int,
                 title_w: int, heaviest_count: int) -> None:
        c = self.curses
        kind_pair = _KIND_PAIR.get(row["kind"], _MUTED)
        if active:
            _addnstr(stdscr, y, 0, " " * width, width, self.attr(_TEXT, active=True))
        x = 2
        guide = row.get("guide") or ""
        _addnstr(stdscr, y, x, guide, width - x, self.attr(_MUTED, active=active))
        x += len(guide)
        _addnstr(stdscr, y, x, row_glyph(row), 1, self.attr(kind_pair, c.A_BOLD, active))
        x += 2
        span = max(8, title_w - len(guide))
        title = node_title(row)
        title = title if len(title) <= span else title[: span - 1] + "…"
        bold = c.A_BOLD if active or row.get("is_current") else 0
        _addnstr(stdscr, y, x, title, span, self.attr(_TEXT, bold, active))
        x = 2 + title_w + 3
        count = int(row.get("message_count") or 0)
        cells = [(kind_label(row["kind"]), 13, kind_pair),
                 (f"{weight_bar(count, heaviest_count)} {messages_label(count):>8}", 13, _BRANCH),
                 (row.get("age") or "", 11, _MUTED)]
        for text, cell_w, pair in cells:
            _addnstr(stdscr, y, x, text, min(cell_w, max(0, width - x)), self.attr(pair, active=active))
            x += cell_w
        notes = row_notes(row)
        if notes and x < width:
            _addnstr(stdscr, y, x, "  ".join(notes), width - x, self.attr(_ACCENT, c.A_BOLD, active))

    def run(self, stdscr) -> None:
        self.init_colors()
        stdscr.keypad(True)
        while True:
            self.draw(stdscr)
            if self.handle(self.read_key(stdscr)):
                return


def browse_tree(db: Any, session_id: str, *, show_all: bool = False) -> Optional[str]:
    """Open the browser on the family around *session_id*; the picked session id, or ``None``."""
    os.environ.setdefault("ESCDELAY", "25")  # a bare Esc closes promptly instead of after 1s
    try:
        import curses
    except ImportError:
        return _numbered_pick(db, session_id, show_all)
    browser = _TreeBrowser(curses, db, session_id, show_all)
    if len(browser.rows) < 2:
        return None
    try:
        curses.wrapper(browser.run)
    except curses.error:
        return _numbered_pick(db, session_id, show_all)
    return browser.result


def _numbered_pick(db: Any, session_id: str, show_all: bool) -> Optional[str]:
    """No curses (Windows without the module, an odd terminal): the same rows as a numbered list."""
    from hermes_cli.curses_ui import _numbered_single_fallback

    family = load_family(db, session_id, show_all=show_all)
    top = heaviest(family.nodes)
    labels = [node_line(node, top) for node in family.nodes]
    choice = _numbered_single_fallback(tree_header(family), labels, len(labels))
    if choice is None or family.nodes[choice].get("is_current"):
        return None
    return family.nodes[choice]["session_id"]
