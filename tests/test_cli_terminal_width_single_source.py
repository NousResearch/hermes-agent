"""T2.2: one terminal-width source — the live prompt_toolkit app, shutil only as fallback.

RED premise: banner (:948), panel (:969), ChatConsole (:908), _terminal_columns (:423),
_scrollback_box_width (:537), _term_rows (:86) and cli_stream._terminal_columns each call
shutil.get_terminal_size() directly, so a prompt_toolkit app narrower than shutil's size
(Termux, resized terminals) overflows the banner/box chrome.
"""
import collections
import re

import prompt_toolkit.application

Size = collections.namedtuple("Size", "columns rows")


def _strip_markup(s: str) -> str:
    """Drop rich style markup so cells measure the rendered text, not the tags."""
    return re.sub(r"\[(?![/0-9])[^]]*\]|\[/]", "", s)


class _FakeOutput:
    def get_size(self):
        return Size(60, 20)


class _FakeApp:
    output = _FakeOutput()


def _patch(monkeypatch):
    monkeypatch.setattr(prompt_toolkit.application, "get_app", lambda: _FakeApp())
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "30")


def test_banner_uses_prompt_toolkit_width(monkeypatch):
    _patch(monkeypatch)
    from hermes_cli.cli_render import _build_compact_banner, _cells
    banner = _build_compact_banner()
    rows = [_strip_markup(line) for line in banner.splitlines() if line.strip()]
    assert max(_cells(line) for line in rows) == 60


def test_panel_box_fits_prompt_toolkit_width(monkeypatch):
    _patch(monkeypatch)
    from hermes_cli.cli_render import _panel_box_width
    box = _panel_box_width("title", ["body"], min_width=100, max_width=100)
    assert box + 2 <= 60


def test_scrollback_box_and_stream_width_follow_the_app(monkeypatch):
    _patch(monkeypatch)
    from hermes_cli.cli_render import _terminal_columns, _terminal_width_for_streaming
    from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin
    assert _terminal_columns() == 60
    assert _terminal_width_for_streaming() <= 60
    assert CLIStatusBarMixin._scrollback_box_width() == 60


def test_term_rows_follows_the_app(monkeypatch):
    _patch(monkeypatch)
    from hermes_cli.cli_tui_mixin import _term_rows
    assert _term_rows() == 20
