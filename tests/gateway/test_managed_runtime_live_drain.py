"""A real managed child persists its final receipt while the ordinary daemon drains."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.gateway.test_normal_runtime_boot import control


def test_managed_turn_finishes_inside_configured_restart_drain(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'managed_workers': True},
        'agent': {'restart_after_turn_timeout': 0, 'restart_drain_timeout': 20},
        'model': {'provider': 'custom', 'default': 'drain-fixture', 'base_url': url},
        'platform_toolsets': {'cli': []}, 'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    admission_id = None
    async def exercise(process, descriptor):
        nonlocal admission_id
        async with websocket(home, descriptor) as ws:
            created = await rpc(ws, 'session.create', request_id='drain-session', source='cli', cwd=str(home),
                model='drain-fixture', provider='custom', base_url=url, api_key='loopback-only',
                toolsets=[], ignore_rules=True)
            assert 'result' in created, created
            submitted = await rpc(ws, 'prompt.submit', session_id=created['result']['session_id'],
                                  input_id='drain-input', text='BLOCK_STARTED', finite=True)
            assert 'result' in submitted, submitted
            admission_id = submitted['result']['admission_id']
            assert await asyncio.to_thread(peer.blocked.wait, 30)
            accepted = await asyncio.to_thread(control, home, 'pause-for-update')
            assert accepted['pausing'] is True, accepted
            async with asyncio.timeout(15):
                while (await asyncio.to_thread(control, home, 'identify'))['state'] != 'draining':
                    await asyncio.sleep(.05)
            peer.release.set()
            await asyncio.to_thread(process.wait, 50)
    try:
        with daemon(root, home, env, barrier=False) as (process, descriptor):
            asyncio.run(exercise(process, descriptor))
        with closing(sqlite3.connect(home / 'state.db')) as db:
            row = db.execute('SELECT status,outcome FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
            if row != ('terminal', 'completed'):
                for log in [home / 'restart.log', *(home / 'logs').glob('*.log')]:
                    print(log.name, '\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-100:]))
            assert row == ('terminal', 'completed'), (row, (home / 'restart.log').read_text())
            assert db.execute('SELECT status FROM worker_executions').fetchone()[0] == 'terminal'
        assert len(peer.requests) == 1
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
