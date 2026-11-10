"""The classic CLI starts its banner, and so its prompt, on the terminal's last row (#4412).

``show_banner`` begins with a screen clear, which puts the cursor back at the top: the scroll startup
did before it was undone, and a banner shorter than the terminal left the prompt mid-screen with
blank rows below it. ``/clear`` inside the running CLI cleared and printed from the top the same way.
Both now start from the last row and re-arm the chrome floor, the CLI's stand-in for the cursor
position report it has switched off, so each print keeps the footer on the last rows.
"""
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

ROWS, COLUMNS = 60, 100
BANNER_LINES = 20
CLEAR_SCREEN = "\x1b[2J"


def _row_after_last_clear(output: str) -> int:
    """The cursor's row at the end of ``output``, from its last screen clear (which homes it): each
    newline moves it down a row, and at the bottom the screen scrolls instead."""
    row = 0
    for character in output.rsplit(CLEAR_SCREEN, 1)[1]:
        if character == "\n":
            row = min(row + 1, ROWS - 1)
    return row


def _short_banner(**kwargs):
    for index in range(BANNER_LINES):
        kwargs["console"].print(f"banner line {index}")


@pytest.fixture
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a: os.terminal_size((COLUMNS, ROWS)))
    from cli import HermesCLI

    obj = HermesCLI.__new__(HermesCLI)
    obj.__dict__.update(
        model="test-model", provider="test", api_key="test", base_url="", _provider_source=None, agent=None,
        session_id=None, compact=False, enabled_toolsets=["hermes-core"], disabled_toolsets=[],
        console=Console(file=io.StringIO(), force_terminal=True, width=COLUMNS, height=ROWS))
    return obj


def test_the_startup_banner_starts_on_the_last_row(shell):
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.load_banner_snapshot", return_value=None), \
         patch("hermes_cli.banner.build_welcome_banner", _short_banner):
        shell.show_banner(from_last_row=True)

    assert _row_after_last_clear(shell.console.file.getvalue()) == ROWS - 1, \
        "a banner shorter than the terminal ends on its last row, where the prompt then starts"


def test_clear_inside_the_cli_starts_on_the_last_row(shell, monkeypatch):
    import cli as cli_mod
    from hermes_cli import cli_render

    monkeypatch.setattr(cli_render, "_CHROME_FLOOR", None)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a: None)
    rows_moved_to, resets = [], []
    shell._app = SimpleNamespace(
        output=SimpleNamespace(erase_screen=lambda: None, flush=lambda: None,
                               cursor_goto=lambda row=0, column=0: rows_moved_to.append(row)),
        renderer=SimpleNamespace(reset=lambda **_k: resets.append(True)))
    shell._confirm_destructive_slash = lambda *_a, **_k: "once"
    shell._print_random_tip = lambda: None
    shell.new_session = lambda silent=False: None
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.banner.build_welcome_banner", lambda **_kw: None):
        shell._cmd_clear("/clear")

    assert rows_moved_to == [ROWS], "the cleared screen's banner starts on the last row (CUP rows count from 1)"
    assert resets, "the erased paint is forgotten, so the first print starts on that row"
    assert cli_render._chrome_floor() is not None, "the footer's rows down to the bottom are known again"
