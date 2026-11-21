"""Colour depth of the classic CLI's prompt_toolkit app.

Regression for #157605: Zellij inside kitty paints no kitty Unicode placeholders, but it passes
24-bit colour through, so the CLI must stay truecolour there instead of dropping to 256 colours.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.styles import Style

from cli import HermesCLI

_GRAPHICS_ENV = ("KITTY_WINDOW_ID", "TERM", "TERM_PROGRAM", "WEZTERM_PANE", "ITERM_SESSION_ID", "ZELLIJ",
                 "TMUX", "STY", "PROMPT_TOOLKIT_COLOR_DEPTH")


def _build_app_depth(monkeypatch, env: dict[str, str]) -> ColorDepth:
    for key in _GRAPHICS_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cli = object.__new__(HermesCLI)
    cli._vim_mode = False
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        app = cli._tui_build_application(Layout(Window()), KeyBindings(), Style([]))
        return app.color_depth


@pytest.mark.parametrize(
    "env",
    [
        {"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1"},
        {"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1", "ZELLIJ": "0"},
        {"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty"},
    ],
    ids=["kitty", "zellij-in-kitty", "ghostty"],
)
def test_kitty_family_terminals_run_the_cli_in_truecolour(monkeypatch, env):
    assert _build_app_depth(monkeypatch, env) == ColorDepth.DEPTH_24_BIT
