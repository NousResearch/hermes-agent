"""Regression: terminals reporting has_colors() with COLORS<=0 (Synology ssh) must not crash."""
from types import SimpleNamespace

from hermes_cli import curses_ui


class _Err(Exception):
    pass


def _fake(colors):
    pairs = []

    def init_pair(n, fg, bg):
        if fg > colors - 1:
            raise ValueError("Color number is greater than COLORS-1")
        pairs.append(n)

    return SimpleNamespace(
        error=_Err, COLORS=colors, COLOR_GREEN=2, COLOR_YELLOW=3, COLOR_WHITE=7,
        has_colors=lambda: True, start_color=lambda: None, use_default_colors=lambda: None,
        curs_set=lambda _n: None, init_pair=init_pair, pairs=pairs,
    )


def test_zero_colors_does_not_raise():
    c = _fake(0)
    curses_ui._init_colors(c, True)
    assert c.pairs == []


def test_normal_colors_inits_pairs():
    c = _fake(256)
    curses_ui._init_colors(c, True)
    assert c.pairs == [1, 2, 3]
