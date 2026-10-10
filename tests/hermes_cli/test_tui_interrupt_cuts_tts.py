"""Ctrl+C / Ctrl+Q must silence the speaker, not only stop the turn.

The CLI killed the player process on an interrupt but left a live streaming pipeline
draining the rest of the sentence into the speaker, so the voice kept talking after the
user asked it to stop. Both keys now cut playback through the shared ``_tui_cut_tts``
helper; the rest of each key's behaviour (Ctrl+C's double-press force exit, Ctrl+Q
leaving the palette) is untouched.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest


def _make_cli(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv('HERMES_DEFER_AGENT_STARTUP', '1')
    cli = HermesCLI(model='fixture', provider='openai-compat', api_key='fixture',
                    base_url='http://127.0.0.1:1/v1')
    cli._tui_init_run_state()
    return cli


def _interrupt_keys(cli, monkeypatch, keys, *, audio_active, tts_pending):
    """Drive real key presses through the TUI app and return the recorded side effects."""
    import hermes_cli.cli_tui_mixin as tui
    import tools.tts_streaming as tts
    import tools.voice_mode as vm
    from prompt_toolkit.application import Application
    from prompt_toolkit.document import Document
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.output import DummyOutput

    stopped, marked = [], []
    monkeypatch.setattr(vm, 'is_audio_output_active', lambda: bool(audio_active))
    monkeypatch.setattr(vm, 'stop_playback', lambda: stopped.append(True))
    monkeypatch.setattr(tts, 'mark_speech_interrupted', lambda: marked.append(True))
    monkeypatch.setattr(tui, 'request_hard_interrupt', lambda agent: None)

    async def run():
        with create_pipe_input() as pipe:
            editor = cli._tui_build_input_area()
            cli._input_area = editor
            app = Application(layout=Layout(editor), key_bindings=cli._tui_build_key_bindings(),
                              input=pipe, output=DummyOutput())
            painted = asyncio.Event()
            app.after_render += lambda _: painted.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(painted.wait(), 3)
            try:
                for key in keys:
                    cli._agent_running = True
                    cli.agent = SimpleNamespace(interrupt=lambda: None)
                    if tts_pending:
                        cli._voice_tts_done.clear()
                    else:
                        cli._voice_tts_done.set()
                    cli._voice_tts_stop = threading.Event()
                    editor.buffer.document = Document('keep my draft', 5)
                    painted.clear()
                    pipe.send_text(key)
                    await asyncio.wait_for(painted.wait(), 3)
            finally:
                app.exit()
                await task

    asyncio.run(run())
    return stopped, marked


@pytest.mark.parametrize('key', ['\x03', '\x11'])  # Ctrl+C, Ctrl+Q
def test_interrupt_key_cuts_pending_tts(key, monkeypatch):
    """A user interrupting mid-speech gets silence: player stopped, pipeline stop event set
    and the superseded reply marked so it does not resume."""
    cli = _make_cli(monkeypatch)

    stopped, marked = _interrupt_keys(cli, monkeypatch, [key], audio_active=True, tts_pending=True)

    assert stopped, 'stop_playback() was never called'
    assert marked, 'mark_speech_interrupted() was never called'
    assert cli._voice_tts_done.is_set(), 'the pending TTS pipeline was left running'
    assert cli._voice_tts_stop.is_set(), "the streaming pipeline's stop event was not set"


@pytest.mark.parametrize('key', ['\x03', '\x11'])  # Ctrl+C, Ctrl+Q
def test_interrupt_key_is_a_noop_in_silence(key, monkeypatch):
    """Nothing playing (no audio active, pipeline finished): the interrupt key must not touch
    the audio subsystem at all — no player to kill, no stale interrupt flag."""
    cli = _make_cli(monkeypatch)
    cli._voice_tts_done.set()

    stopped, marked = _interrupt_keys(cli, monkeypatch, [key], audio_active=False, tts_pending=False)

    assert not stopped
    assert not marked
