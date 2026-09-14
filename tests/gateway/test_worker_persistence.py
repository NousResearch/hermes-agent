"""Separate worker, ordinary daemon, production private control and authenticated WS."""
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket


@pytest.mark.linux_only
def test_worker_survives_owner_restart_without_repeating_tool(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'inert', 'base_url': 'http://127.0.0.1:9/v1'},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL='http://127.0.0.1:9/v1')
    async def create(desc):
        async with websocket(home, desc) as ws:
            response = await rpc(ws, 'session.create', request_id='worker-session', source='cli')
            assert 'result' in response, response
            return response['result']['session_id']
    with (tmp_path / 'worker.log').open('w+') as log:
        worker = subprocess.Popen([sys.executable, str(root / 'tests/gateway/fixtures/persistence_worker.py')],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        def command(op, **params):
            worker.stdin.write(json.dumps({'op': op, **params}) + '\n')
            worker.stdin.flush()
            import select
            assert select.select([worker.stdout], [], [], 30)[0], 'worker response timeout'
            raw = worker.stdout.readline()
            log.flush(); log.seek(0)
            assert raw, log.read()
            result = json.loads(raw)
            assert 'fatal' not in result, result
            return result
        try:
            with daemon(root, home, env, barrier=False) as (owner, desc):
                sid = asyncio.run(create(desc))
                first = command('start', home=str(home), session_id=sid)
                assert first['registered'], first
                assert first['rows'] == 3 and first['lost_ack_replayed']
                assert first['wrong_session'] == 'permission_denied'
                assert first['wrong_profile'] == 'profile_mismatch'
                assert first['wrong_generation'] == 'stale_generation'
                assert first['conflict'] == 'admission_conflict'
                assert first['outbox_full'] == 'outbox_full'
                assert first['writable_canonical_fds'] == []
                assert first['canonical_opens'] == []
                with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                    assert db.execute('SELECT COUNT(*) FROM messages WHERE session_id=?', (sid,)).fetchone()[0] == 3
                    assert db.execute('SELECT input_tokens FROM sessions WHERE id=?', (sid,)).fetchone()[0] == 11
                owner.kill(); owner.wait(timeout=10)
                outage = command('outage')
                assert outage['pending'] == 1 and outage['canonical_opens'] == []
            with daemon(root, home, env, barrier=False):
                second = command('adopt')
                assert second['stale'] == 'stale_epoch'
                assert second['epoch'] > first['epoch']
                assert second['writable_canonical_fds'] == []
                assert second['canonical_opens'] == []
                assert second['pid'] == first['pid'] == worker.pid
                with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                    assert db.execute("SELECT COUNT(*) FROM messages WHERE content='tool-marker'").fetchone()[0] == 1
                    assert db.execute('SELECT input_tokens FROM sessions WHERE id=?', (sid,)).fetchone()[0] == 11
                    assert db.execute("SELECT input_tokens FROM session_model_usage WHERE task='compression'").fetchone()[0] == 5
                    assert db.execute("SELECT status FROM worker_executions WHERE execution_id='owned-worker'").fetchone()[0] == 'terminal'
                assert (home / 'tool-marker').read_text() == 'once'
            print(json.dumps({'first': first, 'second': second}))
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
            worker.stdin.close(); worker.stdout.close()
