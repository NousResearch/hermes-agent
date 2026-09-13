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
def test_rotated_agent_survives_owner_kill_and_explicit_adoption(tmp_path):
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
    worker = None
    def wait_file(path):
        import time
        deadline = time.monotonic() + 40
        while not path.exists():
            assert worker.poll() is None, 'worker exited before publishing receipt'
            assert time.monotonic() < deadline, 'worker receipt deadline'
            time.sleep(0.05)
    try:
        with (tmp_path / 'worker.log').open('w+') as log:
            ready, outage, receipt = (tmp_path / name for name in ('ready.json', 'outage', 'receipt.json'))
            with daemon(root, home, env, barrier=False) as (owner, desc):
                sid = asyncio.run(create(desc, 'compressed'))
                worker = subprocess.Popen([sys.executable, str(root / 'tests/gateway/fixtures/rotation_persistence_worker.py')],
                    cwd=root, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log, text=True)
                def send(value):
                    worker.stdin.write(value + '\n')
                    worker.stdin.flush()
                send(json.dumps(dict(home=str(home), url=url, session_id=sid, restart=True,
                    ready=str(ready), outage=str(outage), receipt=str(receipt))))
                wait_file(ready)
                first = json.loads(ready.read_text())
                assert first['session_id'] != sid
                assert len(peer.requests) == 1
                owner.kill()
                owner.wait(timeout=10)
                send('outage')
                wait_file(outage)
            with daemon(root, home, env, barrier=False):
                send('adopt')
                worker.wait(timeout=60)
                log.seek(0)
                assert worker.returncode == 0, log.read()
                proof = json.loads(receipt.read_text())
                assert proof['opens'] == proof['fds'] == [], proof['opens']
                assert proof['restart']['pid'] == worker.pid
                assert proof['restart']['after_epoch'] > first['epoch']
                assert len(peer.requests) == 2
                assert 'RECOVERY_ACK_WORKER_INFERENCE' in proof['result']
                with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                    assert db.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0] == 0
                    assert db.execute('SELECT session_id,status FROM worker_executions').fetchone() == (proof['after'], 'terminal')
                    assert db.execute('SELECT input_tokens,output_tokens FROM sessions WHERE id=?', (proof['after'],)).fetchone() == (10, 5)
                    assert db.execute('SELECT COUNT(*) FROM worker_receipts WHERE sequence=?',
                        (proof['restart']['pending_sequence'],)).fetchone()[0] == 1
                print(json.dumps(dict(restart=proof['restart'], before=proof['before'], after=proof['after'],
                    opens=proof['opens'], fds=proof['fds'], requests=len(peer.requests))))
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
            worker.stdin.close()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
