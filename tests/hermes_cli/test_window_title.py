"""Window titles follow the active CLI session and release ownership on exit."""

import asyncio
import io
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.data_structures import Size

from hermes_cli.window_title import WindowTitle, install_window_title


class Terminal(io.StringIO):
    def isatty(self):
        return True


def output_for(stream):
    return Vt100_Output(stream, lambda: Size(rows=24, columns=80), term="xterm-256color")


def test_title_follows_session_and_restores_previous_title():
    stream = Terminal()
    label = [""]
    title = WindowTitle(output_for(stream), lambda: label[0], cwd="/tmp/project")
    title.refresh()
    label[0] = "Release prep"
    title.refresh()
    title.refresh()
    label[0] = "Other session"
    title.refresh()
    title.close()
    title.close()
    assert stream.getvalue() == (
        "\x1b[22;2t\x1b]2;Hermes · project\x07"
        "\x1b]2;Hermes · Release prep\x07"
        "\x1b]2;Hermes · Other session\x07"
        "\x1b]2;\x07\x1b[23;2t"
    )


def test_untrusted_title_is_single_line_bounded_and_control_free():
    stream = Terminal()
    title = WindowTitle(output_for(stream), lambda: "hello\nworld\x07\x1b\x9d" + "x" * 300, cwd="/")
    title.refresh()
    payload = stream.getvalue().split("\x1b]2;", 1)[1][:-1]
    assert payload.startswith("Hermes · hello world")
    assert len(payload) <= 120
    assert all(ord(c) >= 32 and not 127 <= ord(c) <= 159 for c in payload)


@pytest.mark.parametrize("enabled,tty,term", [(False, True, "xterm"), (True, False, "xterm"), (True, True, "dumb")])
def test_disabled_or_redirected_runs_leave_terminal_alone(monkeypatch, enabled, tty, term):
    monkeypatch.setattr("sys.stdout", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setenv("TERM", term)
    app = SimpleNamespace()
    assert install_window_title(app, lambda: "secret", enabled=enabled) is None


def test_application_polls_title_without_user_input_and_restores_on_exit(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = Terminal()
    monkeypatch.setattr("sys.stdout", stream)

    async def run():
        with create_pipe_input() as input_stream:
            app = Application(layout=Layout(Window()), input=input_stream, output=output_for(stream))
            label = ["First session"]
            updated = asyncio.Event()

            def get_title():
                if label[0] == "Renamed session":
                    updated.set()
                return label[0]

            title = install_window_title(app, get_title)
            title.poll_interval = 0.01

            async def rename_and_exit():
                label[0] = "Renamed session"
                await asyncio.wait_for(updated.wait(), 5)
                app.exit()

            app.pre_run_callables.append(lambda: app.create_background_task(rename_and_exit()))
            try:
                await app.run_async()
            finally:
                title.close()
            assert "\x1b]2;Hermes · Renamed session\x07" in stream.getvalue()
            assert stream.getvalue().endswith("\x1b]2;\x07\x1b[23;2t")

    asyncio.run(run())


def test_broken_terminal_does_not_break_the_cli():
    class Broken(Terminal):
        def write(self, text):
            raise OSError("terminal closed")

    title = WindowTitle(output_for(Broken()), lambda: "session", cwd="/tmp/project")
    title.refresh()
    title.close()


def test_cli_application_installs_session_title_handler(monkeypatch):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
    import cli

    stream = Terminal()
    output = output_for(stream)
    monkeypatch.setattr("sys.stdout", stream)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(cli, "_select_classic_cli_pt_output", lambda stream: output)
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"terminal_title": True}})
    owner = SimpleNamespace(_vim_mode=False, _get_status_bar_session_title=lambda: "Saved session")
    with create_pipe_input() as input_stream, create_app_session(input=input_stream, output=output):
        app = CLITuiRuntimeMixin._tui_build_application(
            owner, Layout(Window()), KeyBindings(), Style([]),
        )
        app.before_render.fire()
        owner._window_title.close()
    assert "\x1b]2;Hermes · Saved session\x07" in stream.getvalue()
    assert stream.getvalue().endswith("\x1b]2;\x07\x1b[23;2t")
