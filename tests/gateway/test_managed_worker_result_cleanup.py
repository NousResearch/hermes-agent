"""A known native worker answer survives cleanup uncertainty before its result frame."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
import sqlite3
import threading

import pytest

from tests.gateway.test_managed_finished_stop_fence import Model, ROOT, alive
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
def test_known_worker_result_survives_cleanup_unknown_and_native_retry(tmp_path, monkeypatch):
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.follower_alive, peer.effect_processes = [], []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'finish-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
        'approvals': {'mode': 'off'}, 'platform_toolsets': {'cli': []}}))
    env = {**child_env(), 'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
           'PYTHONPATH': str(ROOT), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': url}
    jobs = None

    def query(sql, args=()):
        with closing(sqlite3.connect(home / 'state.db')) as db:
            return db.execute(sql, args).fetchall()

    async def exercise(desc):
        nonlocal jobs
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='cleanup-result', source='cli', cwd=str(home),
                model='finish-model', provider='custom', base_url=url, api_key='loopback-only', toolsets=[], ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='FOLLOWER first')
            await async_wait_until(lambda: (home / 'known-result.json').exists(), timeout=30)
            assert json.loads((home / 'known-result.json').read_text())['final_response'] == 'FOLLOWER_RESULT'
            jobs = json.loads((home / 'jobs.json').read_text())
            own, foreign = jobs['owned'], jobs['foreign']
            assert own['owner_task_id'] == foreign['owner_task_id'] == sid
            assert own['session_key'] == foreign['session_key'] and foreign['parent_session_id'] == 'foreign-parent'
            assert own['parent_session_id'] == sid and not alive(own) and alive(foreign)
            genuine_bytes = (home / 'genuine-result.bin').read_bytes()
            genuine = json.loads(genuine_bytes)
            assert genuine['exit_code'] == 7 and genuine['result_complete'] is True
            assert genuine['output'] == 'physical original output\n'
            assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='follower', text='FOLLOWER queued')
            snapshot = await rpc(ws, 'session.resume', session_id=sid)
            generation = snapshot['result']['execution_generation']
            stopped = await rpc(ws, 'session.interrupt', session_id=sid, execution_generation=generation)
            assert 'result' in stopped, stopped
            await async_wait_until(lambda: query("SELECT status FROM session_admissions WHERE request_id='first'")
                                   != [('started',)], timeout=30)
            assert (home / 'cleanup-refused').exists()
            admission = query("SELECT admission_id,status,outcome FROM session_admissions WHERE request_id='first'")[0]
            saved = query('SELECT value FROM state_meta WHERE key=?', ('gateway.admission.result.v1.' + admission[0],))
            assert saved and json.loads(saved[0][0])['result']['final_response'] == 'FOLLOWER_RESULT', (admission, saved)
            assert admission[1:] == ('terminal', 'completed')
            marker = ('gateway.process_stop.v1:' + sid,)
            assert query('SELECT value FROM state_meta WHERE key=?', marker)
            assert query("SELECT status FROM session_admissions WHERE request_id='follower'") == [('queued',)]
            assert peer.follower_alive == [False] and alive(foreign)
            result_path = home / 'logs' / 'process-results' / (own['id'] + '.json')
            assert result_path.read_text() == '{unreadable native retained ownership'
            # Restore only the exact bytes produced by this actual original job.
            result_path.write_bytes(genuine_bytes)
            retried = await rpc(ws, 'session.interrupt', session_id=sid, execution_generation=generation)
            assert 'result' in retried, retried
            await async_wait_until(lambda: query("SELECT status,outcome FROM session_admissions WHERE request_id='follower'")
                                   == [('terminal', 'completed')], timeout=30)
            assert peer.follower_alive == [False, False] and alive(foreign)
            assert not query('SELECT value FROM state_meta WHERE key=?', marker)
            retained = json.loads(result_path.read_text())
            assert retained['exit_code'] == 7 and retained['output'] == genuine['output'] and retained['result_complete']
            assert query("SELECT status,outcome FROM session_admissions WHERE request_id='first'") == [('terminal', 'completed')]
            assert len(query("SELECT admission_id FROM session_admissions WHERE request_id IN ('first','follower')")) == 2

    try:
        fixture = str(ROOT / 'tests/gateway/fixtures/managed_result_cleanup_owner.py')
        with daemon(ROOT, home, env, barrier=True, fixture=fixture) as (_proc, desc):
            try:
                asyncio.run(exercise(desc))
            finally:
                if (home / 'genuine-result.bin').exists() and jobs is not None:
                    path = home / 'logs' / 'process-results' / (jobs['owned']['id'] + '.json')
                    path.write_bytes((home / 'genuine-result.bin').read_bytes())
                if jobs is not None:
                    from tools import process_registry as module
                    monkeypatch.setenv('HERMES_HOME', str(home))
                    monkeypatch.setattr(module, 'CHECKPOINT_PATH', home / 'processes.json')
                    cleanup = module.ProcessRegistry()
                    cleanup.recover_from_checkpoint(owns_entry=lambda row: row['session_id'] == jobs['foreign']['id'])
                    result = cleanup.kill_process(jobs['foreign']['id'])
                    assert result['status'] in {'already_exited', 'killed'}, result
                    assert not alive(jobs['foreign'])
    finally:
        peer.shutdown()
        peer.server_close()
