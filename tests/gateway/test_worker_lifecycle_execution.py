"""Ordinary owner and worker exercise lifecycle parity without worker SQLite."""
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
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_worker_agent_execution import Model


@pytest.mark.linux_only
def test_worker_lifecycle_then_owner_successor_refuses_late_mutations(tmp_path):
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
            ids = []
            for name in ('owned', 'foreign'):
                response = await rpc(ws, 'session.create', request_id=name, source='cli',
                                     model='worker-model', toolsets=[])
                assert 'result' in response, response
                ids.append(response['result']['session_id'])
            return ids

    def rows(sid):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return dict(db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone())

    async def successor(desc, sid):
        async with websocket(home, desc) as ws:
            resumed = await rpc(ws, 'session.resume', session_id=sid)
            assert 'result' in resumed, resumed
            response = await rpc(ws, 'prompt.submit', session_id=sid, input_id='successor', text='SUCCESSOR')
            assert 'result' in response, response
            async with asyncio.timeout(30):
                while True:
                    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                        status = db.execute('SELECT status FROM session_admissions WHERE request_id=?', ('successor',)).fetchone()
                    if status and status[0] == 'terminal':
                        return
                    await asyncio.sleep(.05)
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid, foreign = asyncio.run(create(desc))
            ready, receipt = tmp_path / 'ready.json', tmp_path / 'receipt.json'
            with (tmp_path / 'worker.log').open('w+') as log:
                proc = subprocess.Popen([sys.executable, str(root / 'tests/gateway/fixtures/lifecycle_persistence_worker.py')],
                    cwd=root, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
                try:
                    proc.stdin.write(json.dumps(dict(home=str(home), url=url, session_id=sid, foreign=foreign,
                                                     ready=str(ready), receipt=str(receipt))) + '\n')
                    proc.stdin.flush()
                    deadline = time.monotonic() + 60
                    while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                        time.sleep(.05)
                    log.seek(0)
                    assert ready.exists(), log.read()
                    before = rows(sid)
                    assert before['end_reason'] == 'agent_close' and before['title'] == 'worker-title'
                    asyncio.run(successor(desc, sid))
                    settled = rows(sid)
                    assert settled['runtime_generation'] > before['runtime_generation']
                    proc.stdin.write('successor-settled\n')
                    proc.stdin.flush()
                    proc.wait(timeout=30)
                    log.seek(0)
                    assert proc.returncode == 0, log.read()
                    proof = json.loads(receipt.read_text())
                    assert proof['opens'] == [] and proof['fds'] == [], proof
                    assert set(proof['foreign'].values()) == {'permission_denied'}, proof
                    assert len(proof['foreign']) == 3
                    assert set(proof['late'].values()) == {'stale_generation'}, proof
                    assert len(proof['late']) == 4
                    assert proof['context']['input_tokens'] == 10 and proof['context']['output_tokens'] == 5
                    assert proof['finished']['status'] == 'terminal'
                    after = rows(sid)
                    for key in ('title', 'end_reason', 'ended_at', 'billing_provider', 'runtime_generation'):
                        assert after[key] == settled[key]
                    assert rows(foreign)['title'] is None and rows(foreign)['ended_at'] is None
                    assert len(peer.requests) == 2
                    print(json.dumps(dict(worker=proof, before_generation=before['runtime_generation'],
                                          successor_generation=settled['runtime_generation'],
                                          inference_requests=len(peer.requests)), default=str))
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)
                    proc.stdin.close()
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
