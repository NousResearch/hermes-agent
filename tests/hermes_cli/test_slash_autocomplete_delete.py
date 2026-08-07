"""Behavioral regression tests for classic CLI completion after deletion."""

import asyncio
from collections.abc import Callable

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.widgets import TextArea

from cli import _refresh_cli_input_completions_after_deletion
from hermes_cli.commands import SlashCommandAutoSuggest, SlashCommandCompleter


async def _wait_for(
    predicate: Callable[[], bool],
    timeout: float = 2.0,
    describe: Callable[[], object] | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    state = describe() if describe else None
    raise AssertionError(
        f"prompt_toolkit state did not update before timeout: {state!r}"
    )


def _completion_texts(input_area: TextArea) -> list[str]:
    state = input_area.buffer.complete_state
    return [completion.text for completion in state.completions] if state else []


async def _exercise_deletion(delete_input: bytes, *, move_left: bool) -> None:
    completer = SlashCommandCompleter()
    input_area = TextArea(
        multiline=True,
        completer=ThreadedCompleter(completer),
        complete_while_typing=True,
        auto_suggest=SlashCommandAutoSuggest(
            history_suggest=AutoSuggestFromHistory(),
            completer=completer,
        ),
    )
    previous_text_length = [0]

    def refresh_after_deletion(buffer) -> None:
        current_length = len(buffer.text)
        chars_added = current_length - previous_text_length[0]
        previous_text_length[0] = current_length
        _refresh_cli_input_completions_after_deletion(buffer, chars_added)

    input_area.buffer.on_text_changed += refresh_after_deletion

    with create_pipe_input() as pipe_input:
        app = Application(
            layout=Layout(input_area),
            input=pipe_input,
            output=DummyOutput(),
        )
        app_task = asyncio.create_task(app.run_async())
        try:
            await asyncio.sleep(0.05)
            pipe_input.send_text("/reas")
            await _wait_for(
                lambda: input_area.buffer.suggestion is not None
                and input_area.buffer.suggestion.text == "oning"
                and "reasoning" in _completion_texts(input_area)
            )

            if move_left:
                pipe_input.send_bytes(b"\x1b[D")
                await _wait_for(lambda: input_area.buffer.cursor_position == 4)
            pipe_input.send_bytes(delete_input)

            await _wait_for(
                lambda: input_area.text == "/rea"
                and input_area.buffer.suggestion is not None
                and input_area.buffer.suggestion.text == "soning"
                and "reasoning" in _completion_texts(input_area),
                describe=lambda: (
                    input_area.text,
                    input_area.buffer.cursor_position,
                    input_area.buffer.suggestion,
                    _completion_texts(input_area),
                ),
            )
        finally:
            if app.is_running:
                app.exit()
            await app_task


@pytest.mark.parametrize(
    ("delete_input", "move_left"),
    [
        pytest.param(b"\x7f", False, id="backspace"),
        pytest.param(b"\x1b[3~", True, id="forward-delete"),
    ],
)
def test_slash_completion_refreshes_after_deletion(
    delete_input: bytes,
    move_left: bool,
) -> None:
    asyncio.run(_exercise_deletion(delete_input, move_left=move_left))
