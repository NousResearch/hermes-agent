"""A local session's ``/voice tts`` must not make the owner synthesize audio.

Ink sends ``slash.exec`` ``voice tts``. The messaging handler sets that chat's
voice mode to ``all`` and every later reply is synthesized, then handed to
``LocalSessionAdapter``. It inherits the base ``send_voice``, which only posts
a media-fallback notice through the adapter's no-op ``send``, so the audio is
dropped. Nothing plays, the provider is still billed, and Ink's ``/voice off``
does not clear the owner's mode.

The gateway daemon is real (``python -m gateway.run``). The model and the TTS
endpoint are one loopback HTTP server that records every request path.
"""
import asyncio
import json
import signal
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


class RecordingModel(Model):
    def do_POST(self):
        self.server.paths.append(self.path)
        if self.path.endswith('/audio/speech'):
            self.rfile.read(int(self.headers['Content-Length']))
            payload = b'ID3' + b'\0' * 512
            self.send_response(200)
            self.send_header('Content-Type', 'audio/mpeg')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_POST()


def test_local_voice_tts_is_refused_and_other_voice_forms_stay():
    """``/voice tts`` is a named refusal. ``/voice on|off|status`` stay controls."""
    from gateway.session_commands import command_verdict

    assert command_verdict('voice', 'tts') == 'refused'
    assert command_verdict('voice', ' TTS ') == 'refused'
    for arg in ('', 'on', 'off', 'status', 'channel', 'join', 'leave'):
        assert command_verdict('voice', arg) == 'control', arg


def test_local_voice_refusal_reads_the_argument_as_the_handler_does():
    """Only the argument the handler stores as mode ``all`` is refused. Any other text, an
    unbalanced quote included, still reaches the handler instead of failing the request."""
    from gateway.session_commands import command_verdict

    for arg in ("it's", '"tts', 'tts now'):
        assert command_verdict('voice', arg) == 'control', arg


@pytest.mark.platforms("linux")  # the daemon is stopped with SIGINT, which raises on Windows
def test_local_session_voice_tts_never_synthesizes_unplayable_audio(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), RecordingModel)
    peer.requests, peer.paths = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'voice-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'tts': {'provider': 'openai', 'openai': {'base_url': url, 'api_key': 'loopback-only'}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    out = {}

    async def probe(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='voice', source='tui', toolsets=[], cwd=str(work))
            sid = created['result']['session_id']
            out['first'] = await rpc(ws, 'slash.exec', session_id=sid, command='voice tts')
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='hello', text='HELLO')
            deadline = time.monotonic() + 45
            snap = None
            while time.monotonic() < deadline:
                snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
                if (not snap['running'] and not snap['pending']
                        and 'RECOVERY_ACK_HELLO' in json.dumps(snap['messages'])):
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError(snap)
            # Voice synthesis is awaited before the turn settles, so the paths are final here.
            out['paths_after_turn'] = list(peer.paths)
            out['second'] = await rpc(ws, 'slash.exec', session_id=sid, command='voice tts')
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(probe(desc))
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
    speech = [p for p in out['paths_after_turn'] if p.endswith('/audio/speech')]
    # No client of a local session can play an owner-side voice reply.
    assert not speech, ('owner synthesized TTS for a TUI session reply that nothing plays', out)
    # The owner names the refusal instead of acknowledging a mode it cannot deliver.
    for reply in (out['first'], out['second']):
        assert reply.get('error', {}).get('message') == 'unsupported_command', (
            '/voice tts claims a voice mode nothing delivers', out)
