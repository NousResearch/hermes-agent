"""A managed worker applies the same per-turn approval facts as the in-process turn: the admission's
one-shot flags (``chat -q`` / ``-z``) and the route's current YOLO bypass, including its revocation."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


class Model(BaseHTTPRequestHandler):
    """Asks for one dangerous terminal call per user turn, then answers once the tool result lands."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        last = (body.get('messages') or [{'role': 'system'}])[-1]
        if last['role'] == 'user':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f"call-{len(body['messages'])}",
                'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': self.server.command, 'timeout': 10})}}]}
        else:
            message = {'role': 'assistant', 'content': 'TURN_DONE'}
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
        for index, call in enumerate(message.get('tool_calls', [])):
            call['index'] = index
        frame = {'id': 'flags', 'model': 'flags-model', 'choices': [choice],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _run(tmp_path, scenario):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    target = home / 'delete-me'
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.command = 'rm -rf ' + str(target)
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'flags-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'approvals': {'mode': 'manual', 'single_query_mode': 'deny'}, 'platform_toolsets': {'cli': ['terminal']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def status(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall()

    async def turn(ws, sid, request_id, **flags):
        """Submit one turn; returns 'prompted' if it parked an approval, else 'settled'."""
        target.mkdir(exist_ok=True)
        submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=request_id, text='DELETE', **flags)
        assert 'result' in submitted, submitted
        async with asyncio.timeout(90):
            while status(request_id) != [('terminal',)]:
                snapshot = await rpc(ws, 'session.resume', session_id=sid)
                prompt = next((p for p in snapshot['result']['prompts'] if p['kind'] == 'approval'), None)
                if prompt:
                    await rpc(ws, 'approval.respond', session_id=sid, prompt_id=prompt['prompt_id'],
                              execution_generation=prompt['execution_generation'], choice='deny')
                    while status(request_id) != [('terminal',)]:
                        await asyncio.sleep(.05)
                    return 'prompted'
                await asyncio.sleep(.1)
        return 'settled'

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='flags', source='cli', cwd=str(home),
                                model='flags-model', provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=['terminal'], ignore_rules=True)
            assert 'result' in created, created
            await scenario(ws, created['result']['session_id'], turn, target)
    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()


@pytest.mark.platforms("linux")
def test_one_shot_managed_turns_never_park_an_approval(tmp_path):
    async def scenario(ws, sid, turn, target):
        # `-z`: nobody can answer, so the classic one-shot contract auto-approves.
        assert await turn(ws, sid, 'unattended', finite=True, unattended=True) == 'settled'
        assert not target.exists()
        # `chat -q`: approvals.single_query_mode (deny here) decides instantly, never a parked prompt.
        assert await turn(ws, sid, 'single-query', finite=True) == 'settled'
        assert target.exists()
    _run(tmp_path, scenario)


@pytest.mark.platforms("linux")
def test_managed_turn_follows_the_sessions_current_yolo(tmp_path):
    async def scenario(ws, sid, turn, target):
        on = await rpc(ws, 'config.set', session_id=sid, key='yolo', value='1')
        assert on['result']['value'] == '1', on
        assert await turn(ws, sid, 'yolo-on') == 'settled'
        assert not target.exists()
        # A revocation reaches the next child; it is not a frozen launch flag.
        off = await rpc(ws, 'config.set', session_id=sid, key='yolo', value='0')
        assert off['result']['value'] == '0', off
        assert await turn(ws, sid, 'yolo-off') == 'prompted'
        assert target.exists()
    _run(tmp_path, scenario)
