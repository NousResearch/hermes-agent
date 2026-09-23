"""Tool-enabled ordinary-daemon proof, distinct from unregistered ledger hooks."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        tools = [message for message in messages if message.get('role') == 'tool']
        if tools:
            message = {'role': 'assistant', 'content': 'WORKER_TOOL_LEDGER_OK'}
            finish = 'stop'
        else:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'inert-tool', 'index': 0, 'type': 'function', 'function': {'name': 'terminal',
                'arguments': json.dumps({'command': 'printf WORKER_TOOL_LEDGER_OK'})}}]}
            finish = 'tool_calls'
        frame = {'id': 'tool-reply', 'model': 'worker-model',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}]}
        kind = 'application/json'
        if body.get('stream'):
            frame['choices'][0]['delta'] = frame['choices'][0].pop('message')
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps(frame).encode()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.linux_only
def test_real_tool_enabled_worker_never_opens_canonical_database(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'worker-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    async def create(desc):
        async with websocket(home, desc) as ws:
            response = await rpc(ws, 'session.create', request_id='worker-session', source='cli',
                                 model='worker-model', toolsets=['terminal'])
            assert 'result' in response, response
            return response['result']['session_id']
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid = asyncio.run(create(desc))
            receipt = tmp_path / 'receipt.json'
            worker = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/ledger_tool_worker.py')],
                cwd=root, env=env, input=json.dumps({'home': str(home), 'url': url, 'session_id': sid,
                                                   'receipt': str(receipt)}) + '\n',
                text=True, capture_output=True, timeout=90)
            assert worker.returncode == 0, worker.stdout + worker.stderr
            proof = json.loads(receipt.read_text())
            assert not proof['failed'] and proof['result'] == 'WORKER_TOOL_LEDGER_OK', proof
            assert proof['opens'] == [] and proof['fds'] == [], proof
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                tools = db.execute("SELECT content FROM messages WHERE session_id=? AND role='tool'", (sid,)).fetchall()
                assert len(tools) == 1 and 'WORKER_TOOL_LEDGER_OK' in tools[0][0], tools
                assert db.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0] == 0
            assert len(peer.requests) == 2
            assert any(tool['function']['name'] == 'terminal' for tool in peer.requests[0]['tools'])
            print(json.dumps(proof | {'inference_requests': len(peer.requests), 'tool_rows': tools}))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
