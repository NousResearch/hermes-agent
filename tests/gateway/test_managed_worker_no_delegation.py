"""A managed worker turn never offers ``delegate_task``: its child would need an owner-registered
execution this single-turn worker cannot reserve, so the tool's only outcome would be a refusal
(real daemon, real managed worker, loopback model)."""
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
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.offered.append(json.dumps(body.get('tools') or []))
        message = {'role': 'assistant', 'content': 'WORKER_DONE'}
        frame = {'id': 'tools', 'model': 'm', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        if body.get('stream'):
            payload, kind = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        else:
            frame['choices'][0]['message'] = frame['choices'][0].pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms("linux")
def test_managed_worker_turn_does_not_offer_delegate_task(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.offered = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': ['delegation', 'todo']}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def rows(sql):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql).fetchall()

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            session = await rpc(ws, 'session.create', request_id='tools', source='cli', cwd=str(home), model='m',
                                provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=['delegation', 'todo'], ignore_rules=True)
            sid = session['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='run', text='hello')
            assert 'result' in submitted, submitted
            async with asyncio.timeout(90):
                while rows("SELECT status FROM session_admissions WHERE request_id='run'") != [('terminal',)]:
                    await asyncio.sleep(.1)
    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()
    assert peer.offered, 'the managed worker never reached the model'
    assert all('todo_list' in tools for tools in peer.offered), peer.offered  # the turn kept its other tools
    assert not any('delegate_task' in tools for tools in peer.offered), peer.offered
