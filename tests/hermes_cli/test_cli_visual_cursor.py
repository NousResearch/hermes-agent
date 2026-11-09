"""Real prompt_toolkit key dispatch must navigate wrapped drafts before history."""

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Size
from prompt_toolkit.widgets import TextArea

from cli import HermesCLI


def _navigate(text, position, keys, *, mode="auto", columns=12):
    async def run():
        class Terminal(DummyOutput):
            def get_size(self):
                return Size(rows=24, columns=columns)

        cli = HermesCLI.__new__(HermesCLI)
        cli.config = {"display": {"arrow_key_mode": mode}}
        area = TextArea(text=text, prompt="🧪 ", multiline=True, wrap_lines=True)
        area.buffer.cursor_position = position
        bindings = KeyBindings()
        bindings.add("up")(cli._tui_history_up)
        bindings.add("down")(cli._tui_history_down)

        @bindings.add("c-q")
        def finish(event):
            event.app.exit(result=(area.buffer.text, area.buffer.cursor_position))

        with create_pipe_input() as pipe:
            app = Application(layout=Layout(area), key_bindings=bindings, input=pipe, output=Terminal())
            sent = False

            def send_keys(_app):
                nonlocal sent
                if not sent:
                    sent = True
                    pipe.send_text(keys + "\x11")

            app.after_render += send_keys
            return await asyncio.wait_for(app.run_async(), timeout=5)

    return asyncio.run(run())


@pytest.mark.parametrize(("text", "position", "keys", "expected"), [
    ("abcdefghijklmnopqrstuvwx", 20, "\x1b[A", 8),
    ("abcdefghijklmnopqrstuvwx", 8, "\x1b[B", 20),
    ("abcdefghijklmnopqrstuvwx", 20, "\x1b[A\x1b[B", 20),
    ("abcdefghij\nx\nabcdefghij", 20, "\x1b[A\x1b[A\x1b[B\x1b[B", 20),
])
def test_arrows_move_between_visual_rows_without_replacing_draft(text, position, keys, expected):
    assert _navigate(text, position, keys) == (text, expected)


def test_cursor_mode_keeps_arrow_keys_inside_the_draft():
    assert _navigate("draft", 0, "\x1b[A", mode="cursor") == ("draft", 0)


def test_paste_replaces_selected_text_without_copying_or_submitting():
    from prompt_toolkit.buffer import Buffer

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"paste_collapse_threshold": 5}
    buffer = Buffer()
    buffer.text = "before OLD after"
    buffer.cursor_position = 7
    buffer.start_selection()
    buffer.cursor_position = 10
    cli._tui_handle_paste(SimpleNamespace(data="new\r\ntext", current_buffer=buffer))
    assert buffer.text == "before new\ntext after"
    assert buffer.selection_state is None
