"""A live worker recovers its original finish after an authenticated owner handoff."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until, wait_until


@pytest.mark.platforms('linux')
def test_finish_survives_reattach_after_epoch_refusal(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True, 'detached_workers': True},
        'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def rows(sql, *params):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, params).fetchall()

    def admissions():
        return rows('SELECT admission_id,status,outcome,owner_epoch FROM session_admissions WHERE request_id=?',
                    'epoch-turn')

    async def start(desc):
        async with websocket(home, desc) as ws:
            result = await rpc(ws, 'session.create', request_id='epoch-session', source='cli', cwd=str(home),
                               model='m', provider='custom', base_url=url, api_key='loopback-only',
                               toolsets=[], ignore_rules=True)
            sid = result['result']['session_id']
            accepted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='epoch-turn', text='EPOCH_TURN')
            assert accepted['result']['status'] == 'queued', accepted
            await async_wait_until((home / 'finish-held.json').exists, timeout=80, what='finish held before RPC')
            return sid

    async def finish(desc, sid, request):
        async with websocket(home, desc) as ws:
            (home / 'successor-observed').touch()
            await async_wait_until((home / 'server-refusal.json').exists, timeout=35, what='native epoch refusal')
            await async_wait_until((home / 'client-refusal.json').exists, timeout=35, what='worker saw refusal')
            assert not rows('SELECT 1 FROM worker_receipts WHERE execution_id=? AND sequence=?',
                            request['execution_id'], request['sequence'])
            (home / 'release-eof').touch()
            await async_wait_until(lambda: admissions()[0][1] in {'unknown', 'terminal'},
                                   timeout=60, what='original admission settled')
            response = await rpc(ws, 'session.resume', session_id=sid)
            return response['result']['messages']

    try:
        with daemon(root, home, env, barrier=True, fixture='detached_epoch_handoff_daemon.py') as (old, first):
            sid = asyncio.run(start(first))
            request = json.loads((home / 'finish-held.json').read_text())
            [(admission_id, status, _, epoch)] = admissions()
            assert status == 'started' and epoch == first['authority_epoch'] == request['epoch']
            queries = (request['execution_id'], request['sequence'])
            assert not rows('SELECT 1 FROM worker_receipts WHERE execution_id=? AND sequence=?', *queries)
            old.kill()
            assert old.wait(timeout=10) < 0
        wait_until((home / 'eof-held').exists, timeout=20, what='old EOF withheld before channel updates')
        with daemon(root, home, env, barrier=True, fixture='detached_epoch_handoff_daemon.py') as (new, successor):
            next_epoch = successor['authority_epoch']
            assert new.pid != old.pid and next_epoch > epoch
            assert admissions() == [(admission_id, 'started', None, next_epoch)]
            messages = asyncio.run(finish(successor, sid, request))
            refusal = json.loads((home / 'server-refusal.json').read_text())
            assert refusal == request | {'error': 'stale_epoch'}
            observed = json.loads((home / 'client-refusal.json').read_text())
            assert observed == {'error': 'stale_epoch', 'epoch': epoch}
            assert admissions() == [(admission_id, 'terminal', 'completed', next_epoch)]
            assert json.loads((home / 'server-applied.json').read_text()) == request | {'epoch': next_epoch}
            assert rows('SELECT sequence FROM worker_receipts WHERE execution_id=? AND sequence=?',
                        *queries) == [(request['sequence'],)]
            assert [(m['role'], m['content']) for m in messages if m['role'] == 'assistant'] == [
                ('assistant', 'RECOVERY_ACK_EPOCH_TURN')]
            assert sum(any(m.get('content') == 'EPOCH_TURN' for m in call['messages'])
                       for call in peer.requests) == 1
    finally:
        (home / 'release-eof').touch()
        (home / 'successor-observed').touch()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
