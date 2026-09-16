"""Exercise radiolist rendering and navigation with a bounded terminal surface."""
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import curses_ui


class Screen:
    def __init__(self, keys, heights):
        self.keys = iter(keys)
        self.heights = iter(heights)
        self.height = 24
        self.frames = []

    def clear(self):
        self.rows = {}

    def getmaxyx(self):
        self.height = next(self.heights, self.height)
        return self.height, 100

    def addnstr(self, y, x, text, n, attr):
        assert 0 <= y < self.height, f"off-screen write at row {y}"
        self.rows[y] = self.rows.get(y, "") + text[:n]

    def refresh(self):
        self.frames.append(dict(self.rows))

    def getch(self):
        return next(self.keys)


@pytest.fixture
def run_menu(monkeypatch):
    def run(keys, heights, *, description=None, selected=0):
        screen = Screen(keys, heights)
        curses = SimpleNamespace(
            error=type("CursesError", (Exception,), {}),
            A_BOLD=1, A_DIM=2, A_NORMAL=0,
            KEY_UP=259, KEY_DOWN=258, KEY_LEFT=260, KEY_ENTER=343,
            KEY_BACKSPACE=263,
            has_colors=lambda: False, curs_set=lambda value: None,
            wrapper=lambda draw: draw(screen),
        )
        monkeypatch.setitem(sys.modules, "curses", curses)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(curses_ui, "flush_stdin", lambda: None)
        result = curses_ui.curses_radiolist(
            "Select default model:", [f"model-{i:02}" for i in range(20)],
            selected=selected, description=description, searchable=True,
        )
        return result, screen.frames
    return run


LONG_DESCRIPTION = "\n".join(f"Unavailable model {i}" for i in range(40))


@pytest.mark.parametrize("height", [5, 8, 12, 24, 50])
def test_long_description_keeps_selected_choice_and_hint_visible(run_menu, height):
    result, frames = run_menu([10], [height], description=LONG_DESCRIPTION, selected=19)
    assert result == 19
    rows = list(frames[0].values())
    assert any("model-19" in row and "\u2192" in row for row in rows)
    assert any("/ search" in row for row in rows)
    choices = [row for row in rows if "model-" in row]
    assert len(choices) >= min(5, height - 4)
    if height < 12:
        # No room for any description row: it is dropped entirely, notice included.
        assert not any("Unavailable model" in row for row in rows)
        assert not any("more description lines" in row for row in rows)
    if 12 <= height < 50:
        assert any("more description lines" in row for row in rows)
    if height == 50:
        assert all(f"Unavailable model {i}" in rows for i in range(40))


def test_choice_scrolls_after_terminal_shrinks(run_menu):
    result, frames = run_menu([258] * 19 + [10], [50, 12], description=LONG_DESCRIPTION)
    assert result == 19
    for index, frame in enumerate(frames):
        assert any(f"model-{index:02}" in row and "\u2192" in row for row in frame.values())


def test_search_result_stays_visible_with_long_description(run_menu):
    result, frames = run_menu(
        [ord("/"), ord("1"), ord("9"), 10], [12], description=LONG_DESCRIPTION)
    assert result == 19
    assert any("Search: 19" in row for row in frames[-1].values())
    assert any("model-19" in row and "\u2192" in row for row in frames[-1].values())


@pytest.mark.parametrize("description", [None, "First line\nSecond line"])
def test_short_description_layout_is_unchanged(run_menu, description):
    result, frames = run_menu([10], [24], description=description)
    assert result == 0
    start = 3 + (2 if description else 0)
    assert "model-00" in frames[0][start]
    assert not any("more description lines" in row for row in frames[0].values())
