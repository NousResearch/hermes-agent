"""Backtrack keys driven through prompt_toolkit's real input parser and key processor.

A terminal delivers Esc and the next key in one read when they are typed quickly, and the parser
turns that pair into an Alt chord. The gesture must still open browse and apply the key, while Alt
keys keep working whenever no prime is pending."""

import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.fixture
def cli(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture",
                    base_url="http://127.0.0.1:1/v1")
    cli._tui_init_run_state()
    cli.conversation_history = [
        {"role": "user", "content": "ALPHA"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "BETA"}, {"role": "assistant", "content": "b1"},
        {"role": "user", "content": "GAMMA"}, {"role": "assistant", "content": "g1"}]
    cli._invalidate = lambda *a, **k: None
    return cli


async def _until(predicate, timeout=3.0):
    await async_wait_until(predicate, timeout=timeout, interval=0.01)


def _drive(cli, script):
    """Run a real Application on a pipe; ``script(pipe, editor)`` is an async callback."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    editor = cli._tui_build_input_area()
    cli._input_area = editor

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            app.key_processor.after_key_press += cli._backtrack_after_key_press
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


def _selected(cli):
    return cli._backtrack_selected() and cli._backtrack_selected()[1]


@pytest.mark.parametrize(("prime", "payload"), [
    (False, "\x1b\x1b\x1b[D"),  # the whole gesture in one read
    (True, "\x1b\x1b[D"),  # primed, then Esc + ← faster than the escape timeout
    (True, "\x1b\x1bOD"),
    (True, "\x1bh"),
])
def test_esc_esc_left_typed_fast_opens_browse_on_the_older_prompt(cli, prime, payload):
    async def script(pipe, editor):
        if prime:
            pipe.send_text("\x1b")
            await _until(lambda: cli._backtrack_primed == "hint")
        pipe.send_text(payload)
        await _until(lambda: _selected(cli) == "BETA")
        assert editor.text == "" and editor.buffer.cursor_position == 0
    _drive(cli, script)


def test_prime_then_fast_esc_enter_prefills_the_latest_prompt_without_a_newline(cli):
    async def script(pipe, editor):
        pipe.send_text("\x1b")
        await _until(lambda: cli._backtrack_primed == "hint")  # lone Esc released by the timeout
        pipe.send_text("\x1b\r")
        await _until(lambda: cli._backtrack_edit is not None)
        assert editor.text == "GAMMA" and cli._backtrack_state is None
        assert cli._backtrack_edit["ordinal"] == 2
    _drive(cli, script)


def test_slow_esc_esc_still_opens_browse_and_real_esc_and_q_close_it(cli):
    async def script(pipe, editor):
        for close in ("\x1b", "q"):
            pipe.send_text("\x1b")
            await _until(lambda: cli._backtrack_primed == "hint")
            pipe.send_text("\x1b")
            await _until(lambda: cli._backtrack_state is not None)
            pipe.send_text("\x1b[D")
            await _until(lambda: _selected(cli) == "BETA")
            pipe.send_text(close)
            await _until(lambda: cli._backtrack_state is None)
            assert editor.text == "" and cli._backtrack_edit is None
    _drive(cli, script)


@pytest.mark.parametrize("draft", ["", "draft"])
def test_alt_enter_inserts_a_newline_when_nothing_is_primed(cli, draft):
    async def script(pipe, editor):
        editor.buffer.text = draft
        editor.buffer.cursor_position = len(draft)
        pipe.send_text("\x1b\r")
        await _until(lambda: editor.text == draft + "\n")
        assert cli._backtrack_state is None and cli._backtrack_primed is None
    _drive(cli, script)


@pytest.mark.parametrize("gap", [True, False])
def test_esc_esc_on_an_erased_edit_cancels_it_and_never_opens_browse(cli, gap):
    """The edit's prefill erased, Esc Esc (slow or in one read) disarms it: the next message is plain."""
    async def script(pipe, editor):
        cli._backtrack_begin_edit(1, "BETA")
        editor.buffer.text = ""
        if gap:
            pipe.send_text("\x1b")
            await _until(lambda: cli._backtrack_primed is not None)
            pipe.send_text("\x1b")
        else:
            pipe.send_text("\x1b\x1b")
        await _until(lambda: cli._backtrack_edit is None)
        assert cli._backtrack_state is None
    cli._prefill_input_buffer = lambda text: None
    _drive(cli, script)
