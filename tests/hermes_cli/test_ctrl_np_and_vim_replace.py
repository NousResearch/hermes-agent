"""Regression coverage for classic list navigation and Vi mode presentation."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import DummyInput
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import DummyOutput

from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli import curses_ui


class Harness(CLITuiMixin):
    def __init__(self):
        self.config = {}
        for name in ('clarify', 'approval', 'slash_confirm', 'sudo', 'secret',
                     'connection', 'model_picker', 'command_palette'):
            setattr(self, f'_{name}_state', None)
        self._clarify_freetext = False
        self._prompt_stash = Stash()
        self._tui_input = SimpleNamespace(buffer=Buffer())

    def _tui_voice_record_key_sequence(self):
        return ('c-b',)


class Stash:
    panel_open = False

    def __len__(self):
        return 2


def active(kb, key):
    return [b for b in kb.get_bindings_for_keys((key,)) if b.filter()]


@pytest.mark.parametrize('state', ['clarify', 'connection', 'approval', 'slash_confirm',
                                   'model_picker', 'command_palette', 'stash'])
@pytest.mark.parametrize('key,arrow', [(Keys.ControlP, Keys.Up), (Keys.ControlN, Keys.Down)])
def test_overlay_aliases(state, key, arrow):
    owner = Harness()
    if state == 'stash':
        owner._prompt_stash.panel_open = True
    else:
        setattr(owner, f'_{state}_state', {'stage': 'provider'})
    kb = owner._tui_build_key_bindings()
    assert active(kb, key)[-1].handler == active(kb, arrow)[-1].handler


@pytest.mark.parametrize('mode', [EditingMode.VI, EditingMode.EMACS])
def test_real_key_processor_completion(mode):
    async def exercise():
        owner = Harness()
        buffer = owner._tui_input.buffer
        buffer.set_document(Document('a', 1))
        buffer.complete_state = CompletionState(buffer.document, [Completion('alpha', -1), Completion('alpine', -1)])
        app = Application(layout=Layout(Window(BufferControl(buffer))),
                          key_bindings=owner._tui_build_key_bindings(), editing_mode=mode,
                          input=DummyInput(), output=DummyOutput())
        with set_app(app):
            for key, index in [(Keys.ControlN, 0), (Keys.ControlN, 1), (Keys.ControlP, 0)]:
                app.key_processor.feed(KeyPress(key))
                app.key_processor.process_keys()
                assert buffer.complete_state.complete_index == index
                assert owner._command_palette_state is None
    asyncio.run(exercise())


@pytest.mark.parametrize('key,expected', [(16, curses_ui.NAV_UP), (14, curses_ui.NAV_DOWN)])
@pytest.mark.parametrize('searching', [False, True])
def test_curses_plain(key, expected, searching):
    import curses
    search = curses_ui._SearchState()
    search.active = searching
    search.query = 'keep'
    assert curses_ui._route_key(curses, Mock(), key, search, True) == (expected, False)
    assert search.query == 'keep'


@pytest.mark.parametrize('letter,expected', [('p', curses_ui.NAV_UP), ('P', curses_ui.NAV_UP),
                                           ('n', curses_ui.NAV_DOWN), ('N', curses_ui.NAV_DOWN)])
@pytest.mark.parametrize('modifier', [5, 69, 133])
def test_curses_enhanced(letter, expected, modifier):
    assert curses_ui._enhanced_key_action(ord(letter), modifier) == expected
    for rejected in (1, 6, 7, 8, 9, 13):
        assert curses_ui._enhanced_key_action(ord(letter), rejected) == curses_ui.NAV_NONE


@pytest.mark.parametrize('letter,expected', [('p', curses_ui.NAV_UP), ('P', curses_ui.NAV_UP),
                                           ('n', curses_ui.NAV_DOWN), ('N', curses_ui.NAV_DOWN)])
def test_enhanced_search_routing(letter, expected):
    import curses
    screen = Mock()
    screen.getch.side_effect = map(ord, f'[{ord(letter)};5u')
    search = curses_ui._SearchState()
    search.active, search.query = True, 'keep'
    assert curses_ui._route_key(curses, screen, 27, search, True) == (expected, False)
    assert search.active and search.query == 'keep'


def test_monitor_aliases():
    from hermes_cli.cli_subagent_monitor import build_monitor_application
    monitor = Mock(detail=False, roster=[], selected=None)
    app = build_monitor_application(monitor, input=DummyInput(), output=DummyOutput())
    for key, arrow in [(Keys.ControlP, Keys.Up), (Keys.ControlN, Keys.Down)]:
        assert active(app.key_bindings, key)[-1].handler == active(app.key_bindings, arrow)[-1].handler
        monitor.detail = True
        assert not active(app.key_bindings, key)
        monitor.detail = False


def test_gateway_cursor_import_fallback(monkeypatch):
    import builtins
    from hermes_cli import config, gateway_chat_view
    monkeypatch.setattr(config, 'load_config_readonly', lambda: {'display': {'vim_mode': True}})
    original_import = builtins.__import__
    def without_modal(name, *args, **kwargs):
        if name == 'prompt_toolkit.cursor_shapes':
            raise ImportError('old prompt_toolkit')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_modal)
    assert asyncio.run(gateway_chat_view._composer_editing_options()) == {'editing_mode': EditingMode.VI}


@pytest.mark.parametrize('vim', [True, False])
def test_gateway_uses_readonly_config_off_loop(monkeypatch, vim):
    from hermes_cli import config, gateway_chat_view
    import prompt_toolkit
    import threading
    main_thread = threading.get_ident()
    def load():
        assert threading.get_ident() != main_thread
        return {'display': {'vim_mode': vim}}
    monkeypatch.setattr(config, 'load_config_readonly', load)
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr(prompt_toolkit, 'PromptSession', factory)
    async def exercise():
        view = gateway_chat_view.GatewayChatView(SimpleNamespace(events=asyncio.Queue()),
                                                {'stored_session_id': 'test'})
        async def read(*args):
            raise EOFError
        view._read_line = read
        assert await view.run() == 0
    asyncio.run(exercise())
    assert factory.call_args.kwargs['editing_mode'] == (EditingMode.VI if vim else EditingMode.EMACS)
    if vim:
        from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
        assert isinstance(factory.call_args.kwargs['cursor'], ModalCursorShapeConfig)


@pytest.mark.parametrize('malformed', [{'display': False}, {'display': None}, {'display': 'vim'}, None, []])
def test_gateway_ignores_non_mapping_display_config(monkeypatch, malformed):
    from hermes_cli import config, gateway_chat_view
    monkeypatch.setattr(config, 'load_config_readonly', lambda: malformed)
    assert asyncio.run(gateway_chat_view._composer_editing_options()) == {'editing_mode': EditingMode.EMACS}
