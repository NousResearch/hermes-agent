"""Which terminals the classic CLI treats as re-wrapping their rows on a width change.

Regression for #157686: Zellij re-wraps on resize like tmux, but keeps the host terminal's TERM,
so only its pane environment (``ZELLIJ``) names it. Without it a shrink inside a Zellij pane with
a generic TERM erased and replayed the viewport into a terminal that had already re-wrapped it,
duplicating and cutting reply lines in Zellij's scrollback.
"""

from collections import deque

import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.cli_render import _MULTIPLEXER_ENV, _NO_REFLOW_ENV, _REFLOW_ENV, _terminal_reflows
from tests.hermes_cli.test_cli_force_redraw import _fake_app


@pytest.fixture
def terminal_env(monkeypatch):
    """Clears everything _terminal_reflows() reads, so the host running the tests cannot leak an
    answer; returns a setter for the variables a case needs."""
    for name in ("TERM", "TERM_PROGRAM", "LC_TERMINAL", *_MULTIPLEXER_ENV, *_NO_REFLOW_ENV, *_REFLOW_ENV):
        monkeypatch.delenv(name, raising=False)

    def set_env(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    return set_env


@pytest.mark.parametrize("env,expected", [
    # Zellij on a remote host over ssh, or in a host terminal with a forced generic TERM.
    ({"TERM": "xterm-256color", "ZELLIJ": "0"}, True),
    # Zellij started from an xterm inherits XTERM_VERSION; the CLI still talks to Zellij.
    ({"TERM": "xterm-256color", "ZELLIJ": "0", "XTERM_VERSION": "XTerm(390)"}, True),
    # A non-reflowing TERM still wins: st launched from a shell inside a Zellij pane.
    ({"TERM": "st-256color", "ZELLIJ": "0"}, False),
])
def test_zellij_pane_counts_as_reflowing(terminal_env, env, expected):
    terminal_env(**env)

    assert _terminal_reflows() is expected


def test_shrink_in_a_zellij_pane_only_erases_the_rewrapped_chrome(terminal_env, monkeypatch):
    """Zellij has re-wrapped the transcript and pushed the rows that grew into its scrollback:
    erasing the viewport and replaying the recorded output would show those lines twice. Only
    prompt_toolkit's erase moves, to the top of the chrome as re-wrapped at the new width."""
    import time

    from prompt_toolkit.data_structures import Size

    terminal_env(TERM="xterm-256color", ZELLIJ="0")
    app = _fake_app(rows=30, columns=90, chrome=5, painted=[0, 200, 50, 200, 30], cursor_y=4)
    out = app.renderer.output
    events = []
    out.erase_end_of_line.side_effect = lambda: events.append("erase_row")
    out.erase_screen.side_effect = lambda: events.append("clear")
    out.erase_down.side_effect = lambda: events.append("erase_down")
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY", deque("x" * 180 for _ in range(40)))
    monkeypatch.setattr(cli_mod, "_replay_output_history", lambda *a: events.append("replay"))
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {"display": {"cli_rebuild_scrollback_on_redraw": False}})
    cli_mod._set_chrome_floor(None)

    bare_cli = object.__new__(HermesCLI)
    bare_cli._status_bar_suppressed_after_resize = False
    bare_cli._last_resize_width = 200
    monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 90)
    monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
    app.renderer._last_size = Size(rows=30, columns=200)  # the chrome paint: at the old width
    out.get_size.return_value = Size(rows=30, columns=200)
    bare_cli._note_chrome_paint(app, True)
    bare_cli._resize_seen_at = time.monotonic() + 1.0
    out.get_size.return_value = Size(rows=30, columns=90)

    bare_cli._recover_after_resize(
        app, lambda: events.append(("original_resize", app.renderer._cursor_pos.y)))

    # Chrome rows of 0/200/50/200 cells re-wrapped at 90 columns: 1 + 3 + 1 + 3 above the cursor.
    assert events == [("original_resize", 1 + 3 + 1 + 3)]
