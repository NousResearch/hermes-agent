"""A managed worker builds its agent with the session's frozen checkpoint limits, not only the
enable flag (andrexibiza 27 / P3.b, kshitijk4poor): ``max_snapshots`` reaches the store's trim."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


class Model(BaseHTTPRequestHandler):
    """One write_file call per user turn, then a plain answer once the tool result lands."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        last = (body.get('messages') or [{'role': 'system'}])[-1]
        if last['role'] == 'user':
            turn = sum(m['role'] == 'user' for m in body['messages'])
            args = {'path': str(self.server.project / f'turn{turn}.txt'), 'content': f'turn {turn}\n'}
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': f'call-{turn}',
                       'type': 'function', 'function': {'name': 'write_file', 'arguments': json.dumps(args)}}]}
        else:
            message = {'role': 'assistant', 'content': 'WROTE'}
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}
        frame = {'id': 'cp', 'model': 'cp-model', 'choices': [choice],
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


@pytest.mark.platforms("linux")
def test_managed_worker_trims_checkpoints_to_the_configured_max_snapshots(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, project = tmp_path / 'state', tmp_path / 'user', tmp_path / 'project'
    home.mkdir(mode=0o700)
    user.mkdir()
    project.mkdir()
    (project / 'seed.txt').write_text('seed\n')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.project = project
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'cp-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['file']},
        'checkpoints': {'enabled': True, 'max_snapshots': 1}}))
    env = {**child_env(), **{k: os.environ[k] for k in ('TIRITH_ENABLED',) if k in os.environ}}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def status(request_id):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT status FROM session_admissions WHERE request_id=?', (request_id,)).fetchall()

    async def exercise(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='cp', source='cli', cwd=str(project),
                                model='cp-model', provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=['file'], ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            for request_id in ('first', 'second'):
                submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=request_id, text='WRITE')
                assert 'result' in submitted, submitted
                async with asyncio.timeout(90):
                    while status(request_id) != [('terminal',)]:
                        await asyncio.sleep(.05)
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute('SELECT COUNT(*) FROM worker_executions WHERE session_id=?', (sid,)).fetchone()[0]
    try:
        with daemon(root, home, env, barrier=False) as (_owner, desc):
            workers = asyncio.run(exercise(desc))
    finally:
        peer.shutdown()
        peer.server_close()
    assert workers == 2 and (project / 'turn1.txt').exists() and (project / 'turn2.txt').exists()
    store = home / 'checkpoints' / 'store'
    refs = subprocess.run(['git', '--git-dir', str(store), 'for-each-ref', '--format=%(refname)', 'refs/hermes'],
                          capture_output=True, text=True, check=True).stdout.split()
    assert len(refs) == 1, refs
    log = subprocess.run(['git', '--git-dir', str(store), 'log', '--format=%s', refs[0]],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    # Both turns checkpointed before writing (default limit 20 would keep both); the frozen limit keeps one.
    assert log == ['before write_file'], log
