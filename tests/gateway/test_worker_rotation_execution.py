"""Real compression, real registered worker, real ordinary owner transport."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_worker_agent_execution import Model


@pytest.mark.linux_only
def test_full_agent_worker_rotation_continues_inference_without_canonical_opens(tmp_path):
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
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'worker-model', 'base_url': url}},
        'platform_toolsets': {'cli': []}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    async def create(desc, key):
        async with websocket(home, desc) as ws:
            response = await rpc(ws, 'session.create', request_id=key, source='cli', model='worker-model', toolsets=[])
            assert 'result' in response, response
            return response['result']['session_id']
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid = asyncio.run(create(desc, 'compressed'))
            foreign = asyncio.run(create(desc, 'foreign'))
            receipt = tmp_path / 'receipt.json'
            result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/rotation_persistence_worker.py')],
                cwd=root, env=env, input=json.dumps(dict(home=str(home), url=url, session_id=sid,
                    foreign=foreign, receipt=str(receipt))) + '\n', text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr
            proof = json.loads(receipt.read_text())
            assert proof['opens'] == proof['fds'] == [], json.dumps(proof['opens'], indent=2)
            assert proof['compressed_count'] < proof['input_count']
            assert len(peer.requests) == 2, 'summary and continuation must both run'
            assert 'RECOVERY_ACK_WORKER_INFERENCE' in proof['result']
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                assert db.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0] == 0
                assignment = db.execute('SELECT session_id,status FROM worker_executions').fetchone()
                assert assignment == (proof['after'], 'terminal')
                assert db.execute('SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1', (proof['after'],)).fetchone()[0] == len(proof['history'])
            print(json.dumps(dict(before=proof['before'], after=proof['after'], opens=proof['opens'],
                fds=proof['fds'], negatives=proof['negatives'], requests=len(peer.requests),
                compressed_count=proof['compressed_count'], input_count=proof['input_count'],
                persisted_rows=len(proof['history']))))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
