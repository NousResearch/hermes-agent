"""Classic-CLI detach key (``display.background_key``, default Ctrl+]) and ``/detach``.

The key is driven through a real prompt_toolkit Application fed raw bytes, so the test covers
what a terminal actually delivers (Ctrl+] = 0x1d) instead of calling the handler directly.
Mirrors tests/hermes_cli/test_subagent_shortcut.py.
"""
import asyncio

import pytest


def _cli(monkeypatch):
    from cli import HermesCLI
    monkeypatch.setenv('HERMES_DEFER_AGENT_STARTUP', '1')
    cli = HermesCLI(model='fixture', provider='openai-compat', api_key='fixture',
                    base_url='http://127.0.0.1:1/v1')
    cli._tui_init_run_state()
    return cli


class _FakeAgent:
    def __init__(self, n=1):
        self.calls = 0
        self.n = n

    def detach_foreground(self):
        self.calls += 1
        return self.n


def _config(monkeypatch, display=None, voice=None):
    cfg = {'display': display or {}, 'voice': voice or {}}
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: cfg)


@pytest.mark.parametrize(('raw', 'expected'), [
    ('ctrl+]', ('c-]',)),
    ('ctrl+\\', ('c-\\',)),
    ('alt+x', ('escape', 'x')),
    ('Control+]', ('c-]',)),
])
def test_background_key_parsing(monkeypatch, raw, expected):
    cli = _cli(monkeypatch)
    _config(monkeypatch, display={'background_key': raw})
    assert cli._tui_background_key_sequence() == expected


def test_background_key_default_is_ctrl_bracket(monkeypatch):
    cli = _cli(monkeypatch)
    _config(monkeypatch)
    assert cli._tui_background_key_sequence() == ('c-]',)


@pytest.mark.parametrize('off', ['off', 'none', '', None, False])
def test_background_key_can_be_disabled(monkeypatch, off):
    cli = _cli(monkeypatch)
    _config(monkeypatch, display={'background_key': off})
    assert cli._tui_background_key_sequence() == ()


def test_background_key_collision_with_voice_disables_detach_key(monkeypatch, caplog):
    """Voice shipped first: on a clash the detach key is dropped (with a warning), voice keeps it."""
    cli = _cli(monkeypatch)
    _config(monkeypatch, display={'background_key': 'ctrl+b'}, voice={'record_key': 'ctrl+b'})
    with caplog.at_level('WARNING'):
        assert cli._tui_background_key_sequence() == ()
    assert 'collides with voice.record_key' in caplog.text


@pytest.mark.parametrize('raw', ['ctrl+spcae', 'ctrl+ctrl+b', 'garbage', 'super+]', 'ctrl+c', ']'])
def test_malformed_background_key_is_rejected_not_reported_as_voice_collision(monkeypatch, caplog, raw):
    """A typo must not fall back to ctrl+b (the voice default) and masquerade as a collision with a
    voice.record_key the user never set: the warning names the bad value instead."""
    cli = _cli(monkeypatch)
    _config(monkeypatch, display={'background_key': raw})
    with caplog.at_level('WARNING'):
        assert cli._tui_background_key_sequence() == ()
    assert 'not a valid ctrl+<key>/alt+<key> binding' in caplog.text
    assert 'collides with voice.record_key' not in caplog.text


def test_background_key_survives_voice_fallback_for_a_typod_voice_key(monkeypatch):
    """The other direction: a typo'd voice.record_key falls back to ctrl+b, which does not touch a
    valid detach key."""
    cli = _cli(monkeypatch)
    _config(monkeypatch, display={'background_key': 'ctrl+]'}, voice={'record_key': 'ctrl+spcae'})
    assert cli._tui_background_key_sequence() == ('c-]',)


def test_detach_key_fires_only_while_a_turn_runs(monkeypatch):
    from prompt_toolkit.application import Application
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    cli = _cli(monkeypatch)
    _config(monkeypatch)
    editor = cli._tui_build_input_area()
    cli._input_area = editor
    agent = _FakeAgent()
    cli.agent = agent

    async def run():
        with create_pipe_input() as pipe:
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                bindings = app.key_bindings
                # Idle: the binding's filter is off, so Ctrl+] keeps its readline meaning.
                cli._agent_running = False
                assert not any(b.filter() for b in bindings.get_bindings_for_keys(('c-]',))
                               if b.handler == cli._tui_handle_detach_key)
                # Busy: raw 0x1d (what the terminal sends for Ctrl+]) reaches detach_foreground AT
                # ONCE — emacs' 2-key ``c-] <char>`` must not make it wait ``timeoutlen``.
                cli._agent_running = True
                editor.buffer.document = Document('keep my draft', 4)
                app.timeoutlen = 30  # without eager=True the test would hang here, not pass late
                painted.clear()
                pipe.send_text('\x1d')
                await asyncio.wait_for(painted.wait(), 3)
                for _ in range(50):
                    if agent.calls:
                        break
                    await asyncio.sleep(0.01)
                assert agent.calls == 1
                assert (editor.text, editor.buffer.cursor_position) == ('keep my draft', 4)
            finally:
                cli._agent_running = False
                app.exit()
                await task
    asyncio.run(run())


def test_detach_command_reports_nothing_to_detach(monkeypatch, capsys):
    cli = _cli(monkeypatch)
    cli.agent = _FakeAgent(n=0)
    cli._agent_running = True
    assert cli._detach_foreground() == 0
    cli._agent_running = False
    assert cli._detach_foreground() == 0
    assert cli.agent.calls == 1  # idle path never reaches the agent


def test_detach_is_dispatched_inline_while_busy():
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli._agent_running = True
    assert cli._should_handle_background_command_inline('/detach') is True
    cli._agent_running = False
    assert cli._should_handle_background_command_inline('/detach') is False


def test_detach_slash_command_detaches_the_running_command(monkeypatch, capsys):
    """``/detach`` through the real slash dispatcher reaches ``agent.detach_foreground()``."""
    cli = _cli(monkeypatch)
    agent = _FakeAgent(n=1)
    cli.agent = agent
    cli._agent_running = True
    try:
        cli.process_command('/detach')
    finally:
        cli._agent_running = False
    assert agent.calls == 1
