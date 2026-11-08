"""Right / Ctrl+E / Ctrl+F accept the ghost-text suggestion in the classic CLI (#151352).

PromptSession loads prompt_toolkit's auto-suggest accept bindings itself; the REPL builds a bare
Application, so they have to be added there. Driven through the real Application built by
``_tui_build_application``, prompt_toolkit's input parser and its key processor."""

import asyncio

import pytest

from tests.waiting import async_wait_until

HISTORY_ENTRY = "okay let's continue with our setup"


@pytest.fixture
def cli(monkeypatch):
    import cli as cli_module
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    # Keep the app on the pipe's DummyOutput rather than a CPR-guarded stdout wrapper.
    monkeypatch.setattr(cli_module, "_select_classic_cli_pt_output", lambda stdout: None)
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    cli._invalidate = lambda *a, **k: None
    return cli


def _drive(cli, script):
    """Run the REPL's own Application on a pipe; ``script(pipe, editor)`` is an async callback."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    editor = cli._tui_build_input_area()
    cli._input_area = editor
    editor.buffer.history.append_string(HISTORY_ENTRY)

    async def run():
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            app = cli._tui_build_application(Layout(editor), cli._tui_build_key_bindings(), None)
            cli._app = app
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                await script(pipe, editor)
            finally:
                app.exit()
                await task
    asyncio.run(run())


async def _until(predicate):
    await async_wait_until(predicate, timeout=3.0, interval=0.01)


@pytest.mark.parametrize("key", ["\x1b[C", "\x05", "\x06"], ids=["right", "ctrl-e", "ctrl-f"])
def test_accept_key_inserts_the_ghost_suggestion(cli, key):
    async def script(pipe, editor):
        pipe.send_text("okay let")
        await _until(lambda: editor.buffer.suggestion is not None)
        assert editor.buffer.suggestion.text == "'s continue with our setup"
        pipe.send_text(key)
        await _until(lambda: editor.text == HISTORY_ENTRY)
        assert editor.buffer.cursor_position == len(HISTORY_ENTRY)
    _drive(cli, script)


def test_right_still_moves_the_cursor_inside_the_draft(cli):
    async def script(pipe, editor):
        pipe.send_text("okay let")
        await _until(lambda: editor.buffer.suggestion is not None)
        pipe.send_text("\x1b[D\x1b[D")  # two Left: the suggestion only applies at the end
        await _until(lambda: editor.buffer.cursor_position == len("okay l"))
        pipe.send_text("\x1b[C")
        await _until(lambda: editor.buffer.cursor_position == len("okay le"))
        assert editor.text == "okay let"
    _drive(cli, script)
