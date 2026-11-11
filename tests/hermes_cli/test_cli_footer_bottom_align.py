"""The classic CLI keeps its composer and status bar on the terminal's last rows.

prompt_toolkit does not shrink a non-full-screen app from one paint to the next: ``Renderer.render``
paints into ``max(rows below the cursor, the last frame's height, the preferred height)``, and only a
reset of the renderer (every print, a resize, Ctrl+L, ``/redraw``, ``/clear``) drops the last frame. So
when a panel in the live area closes (the slash-command menu, a clarify or approval panel), the root
container is handed more rows than its children ask for. ``HSplit``'s default ``JUSTIFY`` left them
blank at the bottom, under the composer or (modern layout) the status bar; bottom-aligned, they pad
the space above the composer, which stays on, or returns to, the terminal's last rows.
"""
from contextlib import contextmanager

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import walk
from prompt_toolkit.layout.screen import WritePosition
from prompt_toolkit.output import DummyOutput

SLASH_MENU_MAX_ROWS = 12
LAYOUTS = pytest.mark.parametrize("modern", [True, False], ids=["modern", "classic"])
WIDTHS = pytest.mark.parametrize("columns", [40, 80, 200])


@contextmanager
def _footer(monkeypatch, modern):
    """The CLI's real root container, built as ``display.cli_layout`` asks, with an app set as the
    current one (never run: a row clipped by the overflow path asks it for a repaint)."""
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    from cli import HermesCLI

    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture", base_url="http://127.0.0.1:1/v1")
    cli._modern_layout = modern
    cli._tui_init_run_state()
    kb = KeyBindings()
    layout, style = cli._tui_build_layout(kb)
    with create_pipe_input() as pipe:
        app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False,
                          input=pipe, output=DummyOutput())
        with set_app(app):
            yield layout


@LAYOUTS
@WIDTHS
@pytest.mark.parametrize("freed", [1, 6, SLASH_MENU_MAX_ROWS])
def test_rows_freed_by_a_closed_panel_pad_above_the_composer(monkeypatch, modern, columns, freed):
    with _footer(monkeypatch, modern) as layout:
        root = layout.container
        preferred = root.preferred_height(columns, 200).preferred
        fitted = root._divide_heights(WritePosition(0, 0, columns, preferred))
        region = preferred + freed
        stretched = root._divide_heights(WritePosition(0, 0, columns, region))

    assert sum(stretched) == region, "every row of the kept region belongs to a child; none is left blank below"
    assert stretched[0] - fitted[0] == freed, "the spare rows go to the top"
    assert stretched[1:] == fitted[1:], "the composer, its rules and the status bar keep their heights"


@LAYOUTS
@WIDTHS
def test_a_footer_one_row_short_clips_above_the_composer(monkeypatch, modern, columns):
    """The overflow path (#57393) under the bottom-aligned root: a row too few still paints the footer,
    and the row it loses is above the composer, never the composer or what sits below it."""
    with _footer(monkeypatch, modern) as layout:
        root = layout.container
        region = root.preferred_height(columns, 200).min - 1
        children = root._all_children
        minimums = [child.preferred_height(columns, region).min for child in children]
        composer = next(index for index, child in enumerate(children)
                        if any(window is layout.current_window for window in walk(child)))
        squeezed = root._divide_heights(WritePosition(0, 0, columns, region))

    assert squeezed is not None, "the footer is painted, not prompt_toolkit's 'Window too small' placeholder"
    assert sum(squeezed) == region
    assert squeezed[composer:] == minimums[composer:], "the composer and every row below it keep their rows"
