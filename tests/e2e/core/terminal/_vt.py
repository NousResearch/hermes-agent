"""A small, dependency-free VT100/xterm screen emulator for PTY transcript assertions.

A raw PTY byte dump shows frame HISTORY (every repaint, every spinner tick, every cursor-up
redraw), not what the user finally sees, so "the reply appears exactly once" can only be judged
on an emulated grid. This implements the subset prompt_toolkit (classic CLI) and Ink (TUI)
actually emit: cursor motion, erase, scroll regions, insert/delete, autowrap with pending-wrap,
the alternate screen, and wide characters. Lines that scroll off the top of the main screen go to
``history`` (the terminal's scrollback), exactly like xterm; ``ESC[3J`` clears it.

Everything unknown (SGR colours, OSC titles/hyperlinks, DEC private modes other than the alt
screen and autowrap, DCS/APC strings, kitty keyboard flags) is parsed and ignored so it can never
leak into the text grid.
"""

from __future__ import annotations

import unicodedata


def char_width(ch: str) -> int:
    if unicodedata.combining(ch) or ch in "\u200b\u200c\u200d\ufe0e\ufe0f":
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


class _Grid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows, self.cols = rows, cols
        self.lines = [self.blank() for _ in range(rows)]

    def blank(self) -> list[str]:
        return [" "] * self.cols


class Screen:
    def __init__(self, rows: int, cols: int, *, history_limit: int = 200_000) -> None:
        self.rows, self.cols = rows, cols
        self.history: list[str] = []
        self.history_limit = history_limit
        self._main = _Grid(rows, cols)
        self._alt = _Grid(rows, cols)
        self.grid = self._main
        self.alt_active = False
        self.x = self.y = 0
        self.top, self.bottom = 0, rows - 1
        self.autowrap = True
        self._pending_wrap = False
        self._saved = (0, 0)
        self._main_saved_cursor = (0, 0)
        self._state = "ground"
        self._params = ""
        self._pending_bytes = b""

    # -- public ---------------------------------------------------------------------------------

    def feed(self, data: bytes | str) -> None:
        if isinstance(data, bytes):
            data = self._pending_bytes + data
            # Keep an incomplete trailing UTF-8 sequence for the next chunk.
            cut = len(data)
            for back in range(1, min(4, len(data)) + 1):
                b = data[-back]
                if b & 0xC0 == 0x80:
                    continue
                if b & 0x80:
                    need = 2 if b & 0xE0 == 0xC0 else 3 if b & 0xF0 == 0xE0 else 4
                    if back < need:
                        cut = len(data) - back
                break
            self._pending_bytes = data[cut:]
            data = data[:cut].decode("utf-8", "replace")
        for ch in data:
            self._step(ch)

    def resize(self, rows: int, cols: int) -> None:
        """Mimic a terminal resize: truncate/pad lines; content above the new height scrolls into history."""
        for grid in (self._main, self._alt):
            lines = [ln[:cols] + [" "] * max(0, cols - len(ln)) for ln in grid.lines]
            if grid is self._main and len(lines) > rows:
                # xterm keeps the cursor row visible: push surplus top rows into scrollback.
                surplus = max(0, min(len(lines) - rows, self.y - rows + 1 if not self.alt_active else 0))
                surplus = max(surplus, 0)
                for ln in lines[:surplus]:
                    self._push_history("".join(ln))
                lines = lines[surplus:]
                if not self.alt_active:
                    self.y -= surplus
            lines = lines[:rows] + [[" "] * cols for _ in range(rows - len(lines[:rows]))]
            grid.lines, grid.rows, grid.cols = lines, rows, cols
        self.rows, self.cols = rows, cols
        self.top, self.bottom = 0, rows - 1
        self.x, self.y = min(self.x, cols - 1), max(0, min(self.y, rows - 1))
        self._pending_wrap = False

    def display(self) -> list[str]:
        return ["".join(c for c in ln if c is not None).rstrip() for ln in self.grid.lines]

    def main_display(self) -> list[str]:
        return ["".join(c for c in ln if c is not None).rstrip() for ln in self._main.lines]

    def transcript(self) -> list[str]:
        """Scrollback followed by the visible main screen (what a user can scroll through)."""
        return list(self.history) + self.main_display()

    # -- parser ---------------------------------------------------------------------------------

    def _step(self, ch: str) -> None:
        st = self._state
        if st == "ground":
            if ch == "\x1b":
                self._state = "esc"
            elif ch < " " or ch == "\x7f":
                self._control(ch)
            else:
                self._print(ch)
        elif st == "esc":
            self._state = "ground"
            if ch == "[":
                self._state, self._params = "csi", ""
            elif ch == "]":
                self._state = "osc"
            elif ch in "PX^_":
                self._state = "str"
            elif ch in "()*+#% ":
                self._state = "esc_skip1"
            elif ch == "7":
                self._saved = (self.x, self.y)
            elif ch == "8":
                self.x, self.y = self._saved
                self._pending_wrap = False
            elif ch == "M":
                self._reverse_index()
            elif ch == "D":
                self._linefeed()
            elif ch == "E":
                self.x = 0
                self._linefeed()
            elif ch == "c":
                self._full_reset()
        elif st == "esc_skip1":
            self._state = "ground"
        elif st == "csi":
            if "\x40" <= ch <= "\x7e":
                self._state = "ground"
                self._csi(self._params, ch)
            elif ch == "\x1b":
                self._state = "esc"
            else:
                self._params += ch
        elif st in ("osc", "str"):
            if ch == "\x07":
                self._state = "ground"
            elif ch == "\x1b":
                self._state = st + "_esc"
        elif st in ("osc_esc", "str_esc"):
            # ESC \ terminates; anything else stays inside the string.
            self._state = "ground" if ch == "\\" else st[:-4]

    def _control(self, ch: str) -> None:
        if ch == "\r":
            self.x = 0
            self._pending_wrap = False
        elif ch in "\n\x0b\x0c":
            self._linefeed()
        elif ch == "\b":
            self.x = max(0, self.x - 1)
            self._pending_wrap = False
        elif ch == "\t":
            self.x = min(self.cols - 1, (self.x // 8 + 1) * 8)

    def _print(self, ch: str) -> None:
        w = char_width(ch)
        if w == 0:
            return
        if self._pending_wrap:
            self.x = 0
            self._linefeed()
            self._pending_wrap = False
        if w == 2 and self.x == self.cols - 1:
            if self.autowrap:
                self.grid.lines[self.y][self.x] = " "
                self.x = 0
                self._linefeed()
            else:
                return
        line = self.grid.lines[self.y]
        line[self.x] = ch
        if w == 2:
            line[self.x + 1] = None  # type: ignore[call-overload]
        if self.x + w >= self.cols:
            self.x = self.cols - 1
            self._pending_wrap = self.autowrap
        else:
            self.x += w

    # -- movement / scrolling ------------------------------------------------------------------

    def _push_history(self, text: str) -> None:
        self.history.append(text.replace("\x00", "").rstrip())
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]

    def _scroll_up(self, n: int = 1) -> None:
        g = self.grid
        for _ in range(n):
            gone = g.lines.pop(self.top)
            if g is self._main and self.top == 0:
                self._push_history("".join(c for c in gone if c is not None))
            g.lines.insert(self.bottom, g.blank())

    def _scroll_down(self, n: int = 1) -> None:
        g = self.grid
        for _ in range(n):
            g.lines.pop(self.bottom)
            g.lines.insert(self.top, g.blank())

    def _linefeed(self) -> None:
        self._pending_wrap = False
        if self.y == self.bottom:
            self._scroll_up()
        elif self.y < self.rows - 1:
            self.y += 1

    def _reverse_index(self) -> None:
        if self.y == self.top:
            self._scroll_down()
        elif self.y > 0:
            self.y -= 1

    def _full_reset(self) -> None:
        self._main, self._alt = _Grid(self.rows, self.cols), _Grid(self.rows, self.cols)
        self.grid, self.alt_active = self._main, False
        self.x = self.y = 0
        self.top, self.bottom = 0, self.rows - 1
        self.autowrap, self._pending_wrap = True, False

    # -- CSI ------------------------------------------------------------------------------------

    def _csi(self, raw: str, final: str) -> None:
        private = raw[:1] if raw[:1] in "?<>=" else ""
        body = raw[len(private):]
        if any(c in body for c in " !\"#$%&'*+,-./"):  # intermediates (DECSCUSR etc.): ignore
            return
        try:
            args = [int(p) if p else 0 for p in body.split(";")] if body else []
        except ValueError:
            return  # sub-parameters (colon form) only appear in SGR, which we ignore anyway

        def a(i: int = 0, default: int = 1) -> int:
            v = args[i] if len(args) > i else 0
            return v or default

        if private == "?":
            if final in "hl":
                on = final == "h"
                for mode in args:
                    if mode == 7:
                        self.autowrap = on
                    elif mode in (47, 1047, 1049):
                        self._set_alt(on, save_cursor=mode == 1049)
            return
        if private:
            return
        self._pending_wrap = False if final not in "m" else self._pending_wrap
        g = self.grid
        if final == "A":
            self.y = max(self.top if self.y >= self.top else 0, self.y - a())
        elif final == "B" or final == "e":
            self.y = min(self.bottom if self.y <= self.bottom else self.rows - 1, self.y + a())
        elif final == "C" or final == "a":
            self.x = min(self.cols - 1, self.x + a())
        elif final == "D":
            self.x = max(0, self.x - a())
        elif final == "E":
            self.x, self.y = 0, min(self.rows - 1, self.y + a())
        elif final == "F":
            self.x, self.y = 0, max(0, self.y - a())
        elif final in "G`":
            self.x = min(self.cols - 1, a() - 1)
        elif final == "d":
            self.y = min(self.rows - 1, a() - 1)
        elif final in "Hf":
            self.y = min(self.rows - 1, a(0) - 1)
            self.x = min(self.cols - 1, a(1) - 1)
        elif final == "J":
            mode = args[0] if args else 0
            if mode == 0:
                g.lines[self.y][self.x:] = [" "] * (self.cols - self.x)
                for r in range(self.y + 1, self.rows):
                    g.lines[r] = g.blank()
            elif mode == 1:
                g.lines[self.y][: self.x + 1] = [" "] * (self.x + 1)
                for r in range(self.y):
                    g.lines[r] = g.blank()
            elif mode == 2:
                g.lines = [g.blank() for _ in range(self.rows)]
            elif mode == 3:
                self.history.clear()
        elif final == "K":
            mode = args[0] if args else 0
            line = g.lines[self.y]
            if mode == 0:
                line[self.x:] = [" "] * (self.cols - self.x)
            elif mode == 1:
                line[: self.x + 1] = [" "] * (self.x + 1)
            else:
                g.lines[self.y] = g.blank()
        elif final == "X":
            n = min(a(), self.cols - self.x)
            g.lines[self.y][self.x:self.x + n] = [" "] * n
        elif final == "P":
            line = g.lines[self.y]
            n = min(a(), self.cols - self.x)
            del line[self.x:self.x + n]
            line.extend([" "] * n)
        elif final == "@":
            line = g.lines[self.y]
            n = min(a(), self.cols - self.x)
            line[self.x:self.x] = [" "] * n
            del line[self.cols:]
        elif final in "LM":
            if self.top <= self.y <= self.bottom:
                for _ in range(min(a(), self.bottom - self.y + 1)):
                    if final == "L":
                        g.lines.pop(self.bottom)
                        g.lines.insert(self.y, g.blank())
                    else:
                        g.lines.pop(self.y)
                        g.lines.insert(self.bottom, g.blank())
                self.x = 0
        elif final == "S":
            self._scroll_up(a())
        elif final == "T":
            self._scroll_down(a())
        elif final == "r":
            top = (args[0] if args and args[0] else 1) - 1
            bottom = (args[1] if len(args) > 1 and args[1] else self.rows) - 1
            if 0 <= top < bottom < self.rows:
                self.top, self.bottom = top, bottom
                self.x = self.y = 0
        elif final == "s":
            self._saved = (self.x, self.y)
        elif final == "u":
            self.x, self.y = self._saved

    def _set_alt(self, on: bool, *, save_cursor: bool) -> None:
        if on == self.alt_active:
            return
        if on:
            if save_cursor:
                self._main_saved_cursor = (self.x, self.y)
            self._alt = _Grid(self.rows, self.cols)
            self.grid, self.alt_active = self._alt, True
        else:
            self.grid, self.alt_active = self._main, False
            if save_cursor:
                self.x, self.y = self._main_saved_cursor
        self.top, self.bottom = 0, self.rows - 1
        self._pending_wrap = False
