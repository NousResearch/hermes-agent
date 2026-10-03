"""Tests for the diff-color gate on non-tty output and NO_COLOR (#88920).

The inline diff renderer wrapped every line in skin truecolor and checked
nothing about where the output goes: Kanban dispatches workers with a plain
file handle as stdout, so ~31% of stored worker-log bytes were escape codes,
and the no-color.org NO_COLOR opt-out was ignored. Colors now require an
interactive sink (or an explicit set_diff_colors_forced(True) for the CLI's
prompt_toolkit sink, which sys.stdout probing cannot see); NO_COLOR and
TERM=dumb always win.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.display import (
    _diff_ansi,
    _render_inline_unified_diff,
    render_edit_diff_with_delta,
    set_diff_colors_forced,
)

_DIFF = "--- a/x.ts\n+++ b/x.ts\n@@ -1 +1 @@\n-old line\n+new line\n"


@pytest.fixture(autouse=True)
def _reset_diff_color_state(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    import agent.display as display

    display._diff_colors_cached = None
    display._diff_colors_forced = False
    yield
    display._diff_colors_cached = None
    display._diff_colors_forced = False


def _ansi_lines() -> int:
    return sum(1 for line in _render_inline_unified_diff(_DIFF) if "\033[" in line)


class TestDiffColorsGate:
    def test_non_tty_stdout_emits_no_ansi(self, monkeypatch):
        """The issue's core repro: a plain file stdout (Kanban worker logs)."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        assert _ansi_lines() == 0

    def test_tty_stdout_emits_ansi(self, monkeypatch):
        tty = MagicMock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", tty)
        assert _ansi_lines() == 4

    def test_no_color_wins_over_tty(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        tty = MagicMock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", tty)
        assert _ansi_lines() == 0

    def test_no_color_set_to_empty_string_still_wins(self, monkeypatch):
        """no-color.org: presence of the variable is the signal, not its value."""
        monkeypatch.setenv("NO_COLOR", "")
        tty = MagicMock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", tty)
        assert _ansi_lines() == 0

    def test_term_dumb_wins_over_tty(self, monkeypatch):
        monkeypatch.setenv("TERM", "dumb")
        tty = MagicMock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", tty)
        assert _ansi_lines() == 0

    def test_forced_beats_non_tty_but_not_no_color(self, monkeypatch):
        """The CLI's prompt_toolkit sink is not sys.stdout; forcing restores the
        interactive transcript's colors — but NO_COLOR still wins."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        set_diff_colors_forced(True)
        assert _ansi_lines() == 4

        monkeypatch.setenv("NO_COLOR", "1")
        import agent.display as display

        display._diff_colors_cached = None
        assert _ansi_lines() == 0


class TestDiffAnsiCache:
    def test_disabled_cache_returns_empty_prefixes(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        colors = _diff_ansi()
        assert set(colors) == {"file", "hunk", "minus", "plus", "dim"} | set(colors)
        assert all(value == "" for value in colors.values())
        # The empty cache must persist like the colored one did.
        assert _diff_ansi() is colors

    def test_wrap_helper_omits_reset_when_prefix_empty(self):
        from agent.display import _wrap_diff_ansi

        assert _wrap_diff_ansi("plus", "+text") == "+text"


class TestRenderedOutputContract:
    def test_non_tty_rendered_text_is_grep_clean(self, monkeypatch, capsys):
        """Issue consequence #2: grep must find diff lines without ANSI in the way."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        printer = MagicMock()
        render_edit_diff_with_delta(
            "patch", '{"diff": ' + repr(_DIFF).replace("'", '"') + "}", print_fn=printer,
        )
        printed = [call.args[0] for call in printer.call_args_list]
        assert printed, "the diff should still render, just without color"
        assert all("\033[" not in line for line in printed)
        assert any("+new line" in line for line in printed)

    def test_tty_render_keeps_colors(self, monkeypatch):
        tty = MagicMock()
        tty.isatty.return_value = True
        monkeypatch.setattr(sys, "stdout", tty)
        printer = MagicMock()
        render_edit_diff_with_delta(
            "patch", '{"diff": "--- a/x\\n+++ b/x\\n+new\\n"}', print_fn=printer,
        )
        printed = [call.args[0] for call in printer.call_args_list]
        assert any("\033[" in line for line in printed)
