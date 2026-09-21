"""Real PTY driver: shared clarify tool -> callback queue -> production keys/render -> result.

Run with pytest; HERMES_CLARIFY_EVIDENCE optionally preserves ANSI/screen/result artifacts.
"""
import asyncio
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

import pytest

fcntl = pytest.importorskip('fcntl')
pty = pytest.importorskip('pty')
termios = pytest.importorskip('termios')

ROOT = Path(__file__).resolve().parents[2]


def child(directory, scenario):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_clarify_terminal_panel import make_cli, question
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout, HSplit
    from prompt_toolkit.widgets import TextArea
    from tools.clarify_tool import clarify_tool
    from tools.ask_user_questions_tool import ask_user_questions_tool
    import cli as cli_module

    async def run():
        cli = make_cli([question(0, ['unused'])])
        cli._clarify_state = None
        cli._clarify_deadline = None
        cli._voice_recording = False
        cli._voice_lock = __import__('threading').Lock()
        cli._agent_running = False
        cli._ring_bell = lambda **kwargs: None
        cli._restore_modal_input_snapshot = lambda: None
        cli_module.CLI_CONFIG['clarify'] = {'timeout': 1 if scenario == 'timeout' else 30}
        composer = TextArea(height=1)
        app = Application(layout=Layout(HSplit([
            cli._tui_overlay_widget(cli._get_clarify_display_fragments, '_clarify_state'),
            composer,
        ]), focused_element=composer), key_bindings=cli._tui_build_key_bindings(), full_screen=True)
        cli._paint_now = app.invalidate
        def snapshot(_):
            screen = app.renderer.last_rendered_screen
            if screen is None:
                return
            size = app.output.get_size()
            visible = '\n'.join(''.join(screen.data_buffer[y][x].char for x in range(size.columns))
                                for y in range(size.rows))
            state = cli._clarify_state or {}
            data = dict(pid=os.getpid(), screen=visible, active=state.get('active'), reviewing=state.get('reviewing'),
                        answers=state.get('answers'), freetext=cli._clarify_freetext, text=composer.text)
            temp = directory / 'frame.tmp'
            temp.write_text(json.dumps(data))
            temp.replace(directory / 'frame.json')
        app.after_render += snapshot
        task = asyncio.create_task(app.run_async())
        await asyncio.sleep(.05)
        def call_tool():
            if scenario == 'auq':
                return ask_user_questions_tool([
                    {'question': 'Deployment?', 'options': [{'label': 'cloud'}, {'label': 'local', 'recommended': True}]},
                    {'question': 'Database?', 'options': [{'label': 'sqlite', 'recommended': True}, {'label': 'postgres'}]},
                ], clarify_callback=cli._clarify_callback)
            return clarify_tool(question='', questions=[
                {'question': 'Colour?', 'choices': ['red', 'blue']},
                {'question': 'Features?', 'choices': ['A', 'B'], 'multi_select': True},
                {'question': 'Describe the special requirement.'},
            ], callback=cli._clarify_callback)
        try:
            result = await asyncio.to_thread(call_tool)
            (directory / 'result.json').write_text(result)
        finally:
            app.exit()
            await task
    asyncio.run(run())


@pytest.mark.parametrize('scenario', ['mixed', 'auq', 'cancel', 'timeout'])
@pytest.mark.parametrize('viewport', [(16, 60), (12, 40)])
def test_real_pty_shared_question_lane(tmp_path, scenario, viewport):
    rows, columns = viewport
    directory = Path(os.environ.get('HERMES_CLARIFY_EVIDENCE', tmp_path)) / f'{scenario}-{rows}x{columns}'
    directory.mkdir(parents=True, exist_ok=True)
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, columns, 0, 0))
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'), TERM='xterm-256color')
    _test_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    command = [sys.executable, '-c',
               f'import runpy; m=runpy.run_path({_test_path!r}); '
               f'm["child"](__import__("pathlib").Path({str(directory)!r}), {scenario!r})']
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    transcript = bytearray()
    frames = []
    def wait_for(predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], .02)
            if ready:
                try:
                    transcript.extend(os.read(master, 65536))
                except OSError:
                    pass
            frame_path = directory / 'frame.json'
            if frame_path.exists():
                frame = json.loads(frame_path.read_text())
                if frame.get('pid') == proc.pid and predicate(frame):
                    frames.append(frame)
                    return frame
            if proc.poll() is not None:
                break
        pytest.fail(f'PTY did not reach expected screen, exit={proc.poll()}: {transcript.decode(errors="replace")[-4000:]}')
    def send(keys):
        os.write(master, keys.encode())
    try:
        first = wait_for(lambda f: f['active'] == 0)
        assert ('Colour?' if scenario != 'auq' else 'Deployment?') in first['screen']
        assert 'Features?' not in first['screen']
        if scenario == 'timeout':
            pass
        elif scenario == 'auq':
            assert '1. local (Recommended)' in first['screen']
            send('\r')
            wait_for(lambda f: f['active'] == 1)
            send('\r')
            wait_for(lambda f: f['reviewing'])
            send('\r')
        else:
            send('\r')
            wait_for(lambda f: f['active'] == 1)
            if scenario == 'cancel':
                send('\x03')
            else:
                # Check A and Other using actual arrows/Space, enter inline text.
                send(' \x1b[B\x1b[B \r')
                wait_for(lambda f: f['freetext'])
                send('custom feature\t')
                wait_for(lambda f: f['active'] == 2 and f['text'] == '')
                send('open draft\x1b[Z')
                wait_for(lambda f: f['active'] == 1 and f['text'] == 'custom feature')
                send('\r')
                wait_for(lambda f: f['active'] == 2 and f['text'] == 'open draft')
                send('\r')
                review = wait_for(lambda f: f['reviewing'])
                assert 'Submit all answers?' in review['screen']
                # Revisit the first question and change an earlier answer.
                send('\t')
                wait_for(lambda f: f['active'] == 0 and not f['reviewing'])
                send('\x1b[B\r')
                wait_for(lambda f: f['reviewing'])
                send('\r')
        # Drain while awaiting exit (PTY buffers must not block the renderer).
        deadline = time.monotonic() + 12
        while proc.poll() is None and time.monotonic() < deadline:
            if select.select([master], [], [], .02)[0]:
                try:
                    transcript.extend(os.read(master, 65536))
                except OSError:
                    break
        assert proc.wait(timeout=2) == 0, transcript.decode(errors='replace')[-4000:]
        result = json.loads((directory / 'result.json').read_text())
        responses = result['responses']
        if scenario == 'mixed':
            assert [r['user_response'] for r in responses] == ['blue', ['A', 'custom feature'], 'open draft']
        elif scenario == 'auq':
            assert [a['answer'] for a in result['answers']] == ['local', 'sqlite']
        elif scenario == 'cancel':
            assert result['cancelled'] is True
            assert responses[0]['user_response'] == 'red'
            assert responses[1]['status'] == 'cancelled'
        else:
            assert result['timed_out'] is True
            assert all(r['status'] == 'timed_out' for r in responses)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
        (directory / 'terminal.ansi').write_bytes(transcript)
        (directory / 'screens.json').write_text(json.dumps(frames, indent=2))
