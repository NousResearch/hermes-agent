"""Regression proofs through production key bindings and prompt_toolkit rendering."""
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Size
from prompt_toolkit.widgets import TextArea

from cli import HermesCLI


def make_cli(questions):
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    for name in ('sudo', 'secret', 'approval', 'slash_confirm', 'model_picker',
                 'command_palette', 'auq', 'todo_panel'):
        setattr(cli, f'_{name}_state', None)
    cli._todo_panel_state = SimpleNamespace(expanded=False)
    cli._prompt_stash = SimpleNamespace(panel_open=False)
    cli._clarify_freetext = False
    cli._clarify_prefill = ''
    cli._clarify_multi_base = None
    cli._persist_prompt_summary = Mock()
    cli._clarify_state = dict(questions=questions, answers={}, answer_meta={},
                             active=0, reviewing=False, response_queue=queue.Queue())
    cli._clarify_batch_set_active(cli._clarify_state, 0)
    return cli


def question(i, choices=None, multi=False):
    return dict(qid=f'q{i}', question=f'Question {i}: choose a value',
                choices=choices, multi_select=multi)


def press(cli, key, text=''):
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.buffer import Buffer
    kb = cli._tui_build_key_bindings()
    keys = {'enter': Keys.ControlM, 'tab': Keys.ControlI, 's-tab': Keys.BackTab}
    binding = [b for b in kb.get_bindings_for_keys((keys.get(key, key),)) if b.filter()][-1]
    buf = Buffer()
    buf.text = text
    app = SimpleNamespace(current_buffer=buf, invalidate=lambda: None)
    binding.handler(SimpleNamespace(app=app, current_buffer=buf))
    return buf.text


def test_enter_after_last_answer_submits_through_real_builder():
    cli = make_cli([question(0, ['red', 'blue']), question(1, ['small', 'large'])])
    state = cli._clarify_state
    press(cli, 'enter')
    press(cli, 'enter')
    assert state['reviewing'] and state['response_queue'].empty()
    press(cli, 'enter')
    assert not state['response_queue'].empty(), 'final Enter never resolves callback queue'
    assert state['response_queue'].get_nowait() == {'q0': 'red', 'q1': 'small'}


def test_tab_from_open_ended_preserves_draft_and_returns():
    cli = make_cli([question(0), question(1, ['yes', 'no'])])
    async def run():
        assert press(cli, 'tab', 'unfinished custom answer') == ''
        assert cli._clarify_state['active'] == 1
        assert press(cli, 's-tab') == 'unfinished custom answer'
        assert cli._clarify_freetext
    asyncio.run(run())


def test_small_panel_only_renders_active_question(monkeypatch):
    import hermes_cli.cli_tui_mixin as tui
    monkeypatch.setattr(tui, '_term_rows', lambda: 16)
    cli = make_cli([question(i, ['alpha', 'beta']) for i in range(4)])
    text = ''.join(fragment[1] for fragment in cli._get_clarify_display_fragments())
    assert 'Question 0:' in text
    assert 'Question 1:' not in text, 'inactive question bodies overflow panel'
    assert len(text.splitlines()) <= 10


def test_recommended_choice_is_first_and_selected():
    cli = make_cli([question(0, ['slow', 'fast (Recommended)'])])
    assert cli._clarify_state['choices'] == ['fast (Recommended)', 'slow']
    assert cli._clarify_state['selected'] == 0


def test_unsaved_choice_and_multiselect_survive_tabs():
    cli = make_cli([question(0, ['a', 'b'], True), question(1, ['yes', 'no'])])
    press(cli, ' ')
    cli._clarify_state['selected'] = 1
    press(cli, 'tab')
    press(cli, 's-tab')
    assert cli._clarify_state['selected_indices'] == {0}
    assert cli._clarify_state['selected'] == 1


def test_number_choice_replaces_older_tab_draft():
    cli = make_cli([question(0, ['a', 'b']), question(1, ['yes', 'no'])])
    press(cli, 'tab')
    press(cli, 's-tab')
    press(cli, '2')
    press(cli, 's-tab')
    assert cli._clarify_state['answers']['q0'] == 'b'
    assert cli._clarify_state['selected'] == 1


def test_review_requires_enter_not_a_number_shortcut():
    cli = make_cli([question(0, ['a', 'b'])])
    state = cli._clarify_state
    press(cli, 'enter')
    press(cli, '2')
    assert state['response_queue'].empty()
    assert state['answers'] == {'q0': 'a'}


class SmallOutput(DummyOutput):
    def get_size(self):
        return Size(rows=16, columns=60)


def test_terminal_parser_multi_question_submit():
    """Real input parser, event loop and render: not calls to state helpers."""
    async def run():
        cli = make_cli([question(0, ['red', 'blue']), question(1, ['small', 'large'])])
        state = cli._clarify_state
        kb = cli._tui_build_key_bindings()
        composer = TextArea(height=1)
        layout = Layout(HSplit([
            cli._tui_overlay_widget(cli._get_clarify_display_fragments, '_clarify_state'),
            composer,
        ]), focused_element=composer)
        with create_pipe_input() as pipe:
            app = Application(layout=layout, key_bindings=kb, input=pipe,
                              output=SmallOutput(), full_screen=True)
            task = asyncio.create_task(app.run_async())
            try:
                await asyncio.sleep(.05)
                pipe.send_text('\r')
                await asyncio.sleep(.05)
                assert state['active'] == 1
                pipe.send_text('\x1b[Z')
                await asyncio.sleep(.05)
                assert state['active'] == 0 and state['answers']['q0'] == 'red'
                pipe.send_text('\t\r')
                await asyncio.sleep(.05)
                assert state['reviewing']
                screen = app.renderer.last_rendered_screen
                visible = '\n'.join(''.join(screen.data_buffer[y][x].char for x in range(60)) for y in range(16))
                assert 'Submit' in visible, visible
                pipe.send_text('\r')
                await asyncio.sleep(.05)
                assert not state['response_queue'].empty(), 'terminal submission hung'
            finally:
                app.exit()
                await task
    asyncio.run(run())
