"""The classic CLI keeps its composer and status bar on the terminal's last rows.

prompt_toolkit never shrinks a non-full-screen app while it runs: ``Renderer.render`` paints into
``max(rows below the cursor, the last frame's height, the preferred height)``. So when a panel in
the live area closes (the slash-command menu, a clarify or approval panel), the root container
is handed more rows than its children ask for. Top-aligned, the spare rows were left blank under
the status bar and the composer jumped up by the closed panel's height; bottom-aligned, they pad
the space above the composer and the composer stays put.
"""
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.screen import WritePosition
from prompt_toolkit.output import DummyOutput

COLUMNS = 120
ROWS_FREED_BY_THE_CLOSED_PANEL = 12  # the slash-command menu's max height


def test_rows_freed_by_a_closed_panel_pad_above_the_composer(monkeypatch):
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    from cli import HermesCLI

    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture", base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    kb = KeyBindings()
    layout, style = cli._tui_build_layout(kb)
    root = layout.container

    with create_pipe_input() as pipe:
        app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False,
                          input=pipe, output=DummyOutput())
        with set_app(app):
            preferred = root.preferred_height(COLUMNS, 200).preferred
            fitted = root._divide_heights(WritePosition(0, 0, COLUMNS, preferred))
            region = preferred + ROWS_FREED_BY_THE_CLOSED_PANEL
            stretched = root._divide_heights(WritePosition(0, 0, COLUMNS, region))

    assert sum(stretched) == region, "every row of the kept region belongs to a child; none is left blank below"
    assert stretched[0] - fitted[0] == ROWS_FREED_BY_THE_CLOSED_PANEL, "the spare rows go to the top"
    assert stretched[1:] == fitted[1:], "the composer, its rules and the status bar keep their heights"
