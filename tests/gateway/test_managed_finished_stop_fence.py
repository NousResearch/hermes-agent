"""A late Stop cannot trust a worker's finished frame produced before that Stop."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import sqlite3
import sys
import threading

import psutil
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until

ROOT = Path(__file__).resolve().parents[2]


def alive(entry):
    from tools.process_registry import ProcessRegistry
    try:
        process = psutil.Process(entry['pid'])
        return (process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                and ProcessRegistry._safe_host_start_time(process.pid) == entry['host_start_time'])
    except psutil.NoSuchProcess:
        return False


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages') or []
        latest = next((str(row.get('content')) for row in reversed(messages) if row['role'] == 'user'), '')
        if 'FOLLOWER' in latest:
            self.server.follower_alive.append(any(alive(entry) for entry in self.server.effect_processes))
        spawn = bool(messages) and not any(row['role'] == 'tool' for row in messages) and 'FOLLOWER' not in latest
        message = {'role': 'assistant', 'content': 'FOLLOWER_RESULT' if 'FOLLOWER' in latest else 'FINISH_RESULT'}
        if spawn:
            command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(90)'])
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'index': 0,
                'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({
                    'command': command, 'background': True, 'persist_on_release': True,
                    'notify_on_complete': False})}}]}
        choice = {'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if spawn else 'stop'}
        frame = {'id': 'fixture', 'model': 'finish-model', 'choices': [choice],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        if body.get('stream'):
            payload = ('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            choice['message'] = choice.pop('delta')
            payload, kind = json.dumps(frame).encode(), 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('refuse', [False, True], ids=['verified-cleanup', 'refused-cleanup'])
def test_late_stop_verifies_processes_after_real_worker_exit(tmp_path, monkeypatch, refuse):
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
        'auxiliary': {'title_generation': {'enabled': False}}, 'approvals': {'mode': 'off'},
        'terminal': {'cwd': str(home)}, 'platform_toolsets': {'cli': ['terminal']}}))
    env = {**child_env(), 'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
           'PYTHONPATH': str(ROOT), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': url}
    evidence, child = {'termination_refused': refuse}, None

    def query(sql, args=()):
        with closing(sqlite3.connect(home / 'state.db')) as db:
            return db.execute(sql, args).fetchall()

    async def exercise(desc):
        nonlocal child
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='finished-stop', source='cli', cwd=str(home),
                model='finish-model', provider='custom', base_url=url, api_key='loopback-only',
                toolsets=['terminal'], ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='SPAWN')
            await async_wait_until(lambda: (home / 'parent-held-finished.json').exists(), timeout=90,
                                   what='actual finished frame reaches parent')
            worker_pid = json.loads((home / 'parent-held-finished.json').read_text())['worker_pid']
            def worker_gone():
                try:
                    return psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    return True
            await async_wait_until(worker_gone, timeout=20, what='real worker exits')
            entries = json.loads((home / 'processes.json').read_text())
            owned = [entry for entry in entries if entry['owner_task_id'] == sid]
            assert len(owned) == 1, entries
            child = owned[0]
            assert child['persist_on_release'] and alive(child)
            from tools.process_registry import ProcessRegistry
            peer.effect_processes = [child, *({'pid': process.pid,
                'host_start_time': ProcessRegistry._safe_host_start_time(process.pid)}
                for process in psutil.Process(child['pid']).children(recursive=True))]
            evidence.update(worker_pid=worker_pid, process_id=child['session_id'])
            assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id='follower', text='FOLLOWER')
            snapshot = await rpc(ws, 'session.resume', session_id=sid)
            if refuse:
                (home / 'refuse-cleanup').touch()
            response = await rpc(ws, 'session.interrupt', session_id=sid,
                                 execution_generation=snapshot['result']['execution_generation'])
            assert 'result' in response, response  # Request ACK precedes physical cleanup verification.
            marker = ('gateway.process_stop.v1:' + sid,)
            assert query('SELECT value FROM state_meta WHERE key=?', marker)
            (home / 'parent-release-finished').touch()
            await async_wait_until(lambda: query('SELECT status FROM session_admissions WHERE request_id=?',
                ('first',)) != [('started',)], timeout=30, what='finished-frame reconciliation')
            states = dict(query('SELECT request_id,status FROM session_admissions'))
            if refuse:
                assert states == {'first': 'terminal', 'follower': 'queued'} and alive(child), states
                with closing(sqlite3.connect(home / 'state.db')) as db:
                    admission_id = db.execute("SELECT admission_id FROM session_admissions WHERE request_id='first'").fetchone()[0]
                    saved = db.execute('SELECT value FROM state_meta WHERE key=?',
                        ('gateway.admission.result.v1.' + admission_id,)).fetchone()
                assert saved and json.loads(saved[0])['result']['final_response'] == 'FINISH_RESULT'
                assert query('SELECT value FROM state_meta WHERE key=?', marker) and peer.follower_alive == []
                (home / 'refuse-cleanup').unlink()
                snapshot = await rpc(ws, 'session.resume', session_id=sid)
                if query('SELECT value FROM state_meta WHERE key=?', marker):
                    response = await rpc(ws, 'session.interrupt', session_id=sid,
                        execution_generation=snapshot['result']['execution_generation'])
                    if response.get('error', {}).get('data', {}).get('reason') == 'stale_generation':
                        assert not query('SELECT value FROM state_meta WHERE key=?', marker)
                        assert not any(alive(entry) for entry in peer.effect_processes)
                    else:
                        assert 'result' in response, response
            await async_wait_until(lambda: query('SELECT status FROM session_admissions WHERE request_id=?',
                ('follower',)) == [('terminal',)], timeout=30, what='human follower completes after native cleanup')
            assert peer.follower_alive == [False]
            assert not alive(child), 'native Stop cleared its fence with physical work still alive'
            assert not any(alive(entry) for entry in peer.effect_processes), 'a captured job descendant survived Stop'
            assert not query('SELECT value FROM state_meta WHERE key=?', marker)
            evidence.update(child_alive=False, follower_saw_live_process=peer.follower_alive)

    try:
        fixture = str(ROOT / 'tests/gateway/fixtures/managed_finished_stop_probe.py')
        with daemon(ROOT, home, env, barrier=True, fixture=fixture) as (_owner, desc):
            try:
                asyncio.run(exercise(desc))
            finally:
                (home / 'parent-release-finished').touch()
                (home / 'refuse-cleanup').unlink(missing_ok=True)
                if child is not None and alive(child):
                    from tools import process_registry as module
                    monkeypatch.setattr(module, 'CHECKPOINT_PATH', home / 'processes.json')
                    cleanup = module.ProcessRegistry()
                    cleanup.recover_from_checkpoint(owns_entry=lambda entry: entry['session_id'] == child['session_id'])
                    cleanup.kill_process(child['session_id'])
    finally:
        peer.shutdown()
        peer.server_close()
        (home / 'managed-finished-stop-receipt.json').write_text(json.dumps(evidence, indent=2))
