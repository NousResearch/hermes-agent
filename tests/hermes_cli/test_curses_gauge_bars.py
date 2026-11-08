"""Gauge colouring in curses/ANSI descriptions only touches real ``render_bar`` gauges."""

from hermes_cli.curses_ui import _gauge_bar_segments
from hermes_cli.model_picker_usage import render_bar


def test_real_gauges_are_styled():
    for fraction, style in ((0.0, "green"), (0.6, "yellow"), (0.9, "red")):
        bar = render_bar(fraction)
        assert _gauge_bar_segments(f"Today  {bar}  x") == [("Today  ", None), (bar, style), ("  x", None)]


def test_unrelated_brackets_stay_plain():
    for text in ("toggle [] here", "checkbox [ ] off", "a [||] b", "flag [x]", "[" + " " * 11 + "]"):
        assert _gauge_bar_segments(text) == [(text, None)]
