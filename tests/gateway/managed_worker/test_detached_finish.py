"""A live detachable worker retains its turn on either side of the finish commit."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


class ModelPeer(ThreadingHTTPServer):
    requests: list[dict]


@pytest.mark.platforms('linux')
@pytest.mark.parametrize(('committed', 'worker_dies'), [(False, False), (True, False), (True, True)],
                         ids=['before-commit', 'after-commit', 'dead-worker'])
def test_same_admission_finishes_across_owner_loss(tmp_path, committed, worker_dies):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ModelPeer(('127.0.0.1', 0), Model)
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
    marker = home / 'before-finish.json'
    if committed:
        (home / 'commit-before-loss').touch()

    def rows(sql, *params):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, params).fetchall()

    def admissions():
        return rows('SELECT admission_id, status, outcome, owner_epoch FROM session_admissions WHERE request_id=?',
                    'finish-turn')

    def receipts(finish):
        return rows('SELECT sequence FROM worker_receipts WHERE execution_id=? AND sequence=?',
                    finish['execution_id'], finish['sequence'])

    async def send_turn(desc):
        async with websocket(home, desc) as ws:
            response = await rpc(ws, 'session.create', request_id='finish-session', source='cli', cwd=str(home),
                                 model='m', provider='custom', base_url=url, api_key='loopback-only',
                                 toolsets=[], ignore_rules=True)
            sid = response['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='finish-turn', text='FINISH_TURN')
            assert submitted['result']['status'] == 'queued', submitted
            await async_wait_until(marker.exists, timeout=50, what='the worker to send execution.finish')
            return sid

    async def receive_result(desc, sid):
        async with websocket(home, desc) as ws:
            await async_wait_until(lambda: any(r[1] in ('terminal', 'unknown') for r in admissions()),
                                   timeout=45, what='the original admission to settle')
            return (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']

    try:
        with daemon(root, home, env, barrier=True, fixture='detached_finish_daemon.py') as (old, first):
            sid = asyncio.run(send_turn(first))
            finish = json.loads(marker.read_text())
            [(admission_id, status, _, epoch)] = admissions()
            record_key = 'gateway.detached_worker.v1.' + finish['execution_id']
            record = rows('SELECT value FROM state_meta WHERE key=?', record_key)
            assert status == 'started' and epoch == finish['epoch'] == first['authority_epoch']
            if committed:
                assert receipts(finish) == [(finish['sequence'],)]
                assert rows('SELECT status FROM worker_executions WHERE execution_id=?', finish['execution_id']) == [('terminal',)]
            else:
                [(raw,)] = record
                assert (json.loads(raw)['pid'], json.loads(raw)['admission_id']) == (finish['pid'], admission_id)
                assert not receipts(finish), 'the old owner committed the finish before the fault'
            if worker_dies:
                os.kill(finish['pid'], signal.SIGKILL)
            old.kill()
            assert old.wait(timeout=10) < 0
        with daemon(root, home, env, barrier=True, fixture='detached_finish_daemon.py') as (successor, next_owner):
            assert successor.pid != old.pid and next_owner['authority_epoch'] > epoch
            messages = asyncio.run(receive_result(next_owner, sid))
            [[got_id, got_status, outcome, owner_epoch]] = admissions()
            if worker_dies:
                assert (got_id, got_status, outcome) == (admission_id, 'unknown', None)
                assert not (home / 'after-finish.json').exists()
            else:
                assert (got_id, got_status, outcome) == (admission_id, 'terminal', 'completed')
                assert owner_epoch == next_owner['authority_epoch']
                assert json.loads((home / 'after-finish.json').read_text()) == {**finish, 'epoch': owner_epoch}
                assert [(m['role'], m['content']) for m in messages if m['role'] == 'assistant'] == [
                    ('assistant', 'RECOVERY_ACK_FINISH_TURN')]
            assert receipts(finish) == [(finish['sequence'],)]
            assert not rows('SELECT value FROM state_meta WHERE key=?', record_key)
            assert rows('SELECT status FROM worker_executions WHERE execution_id=?', finish['execution_id']) == [('terminal',)]

            assert sum(any(m.get('content') == 'FINISH_TURN' for m in call['messages']) for call in peer.requests) == 1
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
