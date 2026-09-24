"""Unmocked worker AIAgent / loopback inference / ordinary owner transport."""
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
        if body.get('messages'):
            self.server.requests.append(body)
        message = {'role': 'assistant', 'content': 'RECOVERY_ACK_WORKER_INFERENCE'}
        usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        frame = {'id': 'worker-reply', 'model': 'worker-model', 'usage': usage,
                 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]}
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
@pytest.mark.parametrize('probe_constructors', [False, True])
def test_real_agent_worker_persists_context_usage_and_releases_lease(tmp_path, probe_constructors):
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
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    async def create(desc):
        async with websocket(home, desc) as ws:
            response = await rpc(ws, 'session.create', request_id='worker-session', source='cli',
                                 model='worker-model', toolsets=[])
            assert 'result' in response, response
            return response['result']['session_id']
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid = asyncio.run(create(desc))
            receipt = tmp_path / 'receipt.json'
            result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/agent_persistence_worker.py')],
                cwd=root, env=env, input=json.dumps({'home': str(home), 'url': url, 'session_id': sid,
                                                   'receipt': str(receipt), 'probe_constructors': probe_constructors}) + '\n',
                text=True, capture_output=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            proof = json.loads(receipt.read_text())
            assert not proof['failed'], proof
            assert 'RECOVERY_ACK_WORKER_INFERENCE' in proof['result'], proof
            if probe_constructors:
                assert proof['refusals']['child'] == 'worker_child_registration_required', proof['refusals']
                assert proof['refusals']['cron'] == 'worker_cron_registration_required', proof['refusals']
                assert proof['refusals']['ledger'] == 'worker_delegation_ledger_unavailable', proof['refusals']
                success, doc, final_response, error = proof['refusals']['script_result']
                assert success and error is None and 'SCRIPT_ONLY_RESULT' in doc
                assert final_response.strip() == 'SCRIPT_ONLY_RESULT'
            assert proof['opens'] == [] and proof['fds'] == [], json.dumps(proof['opens'], indent=2)
            assert proof['context']['system_prompt'], proof
            assert proof['context']['input_tokens'] == 10, proof
            assert proof['context']['output_tokens'] == 5, proof
            assert proof['finished']['status'] == 'terminal'
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                assert db.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0] == 0
                assert db.execute('SELECT input_tokens,output_tokens FROM sessions WHERE id=?', (sid,)).fetchone() == (10, 5)
                rows = db.execute('SELECT role,content FROM messages WHERE session_id=? ORDER BY id', (sid,)).fetchall()
                assert any(role == 'user' and 'WORKER_INFERENCE' in content for role, content in rows)
                assert any(role == 'assistant' and 'RECOVERY_ACK_WORKER_INFERENCE' in content for role, content in rows)
            assert len(peer.requests) == 1, peer.requests
            print(json.dumps({'result': proof['result'], 'opens': proof['opens'], 'fds': proof['fds'],
                              'input_tokens': proof['context']['input_tokens'],
                              'output_tokens': proof['context']['output_tokens'],
                              'prompt_persisted': bool(proof['context']['system_prompt']),
                              'finished': proof['finished'], 'refusals': proof['refusals'],
                              'rows': rows, 'inference_requests': len(peer.requests)}, default=str))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
