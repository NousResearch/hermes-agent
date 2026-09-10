"""Admission-backed automatic compression over the ordinary owner and HTTP model."""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


class OverflowModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        payload = json.dumps({'data': [{'id': n} for n in ('original', 'switched', 'summary')]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        if body.get('model') == 'original' and len(json.dumps(body.get('messages', []))) > 50000:
            payload = json.dumps({'error': {'message': 'maximum context length exceeded',
                                         'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}).encode()
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if body.get('model') == 'original' and 'SUMMARY_RETAINED_FACTS' in json.dumps(body.get('messages', [])):
            self.server.blocked.set()
            self.server.release.wait(60)
        if 'BLOCK_STARTED' in str(text):
            self.server.blocked.set()
            self.server.release.wait(60)
        if body.get('model') == 'summary':
            reply = 'SUMMARY_RETAINED_FACTS'
        else:
            reply = 'ACK_' + str(text)[:40] + ' ' + ('historical detail ' * 700)
        message = {'role': 'assistant', 'content': reply}
        payload = json.dumps({'id': 'local', 'model': body.get('model'), 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.mark.parametrize('ending', ['finish', 'kill'])
def test_automatic_worker_rotation_retains_retry_identity_and_fifo(tmp_path, ending):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), OverflowModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url, 'context_length': 64000},
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
        'compression': {'protect_first_n': 1, 'protect_last_n': 2, 'threshold_tokens': 12000, 'threshold': 0.5, 'in_place': False},
        'platform_toolsets': {'cli': []}}))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 64000 if name == 'original' else 1000000,
                                                 'output': 1000}}
        for name in ('original', 'summary')}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def rows(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, args)]

    replay_params = {}

    async def exercise(desc, owner):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                result = await rpc(ws, method, **params)
                assert 'result' in result, result
                return result['result']

            sid = (await call('session.create', source='cli', request_id='create', model='original',
                              provider='custom', base_url=url, toolsets=[], cwd=str(home)))['session_id']
            for i in range(12):
                params = dict(session_id=sid, input_id=f'pressure-{i}', text='HISTORY_' + str(i) + ' preserve these facts ' * 100)
                accepted = await call('prompt.submit', **params)
                async with asyncio.timeout(40):
                    while True:
                        admission = rows('SELECT * FROM session_admissions WHERE admission_id=?', (accepted['admission_id'],))[0]
                        children = rows('SELECT id FROM sessions WHERE parent_session_id=?', (sid,))
                        if children or admission['status'] in ('terminal', 'unknown'):
                            break
                        await asyncio.sleep(.03)
                if not children:
                    assert admission['status'] == 'terminal', admission
                    continue
                try:
                    assert await asyncio.to_thread(peer.blocked.wait, 20), 'no post-summary inference'
                    retried = await call('prompt.submit', **params)
                    assert retried['admission_id'] == accepted['admission_id']
                    follower = await call('prompt.submit', session_id=sid, input_id='follower', text='FOLLOWER')
                    pending = rows('SELECT admission_id,target_session_id,status FROM session_admissions WHERE status!=\'terminal\'')
                    assert pending == [
                        {'admission_id': accepted['admission_id'], 'target_session_id': sid, 'status': 'started'},
                        {'admission_id': follower['admission_id'], 'target_session_id': sid, 'status': 'queued'}]
                    worker = rows('SELECT session_id FROM worker_executions WHERE execution_id=?',
                                  ('admission-worker:' + accepted['admission_id'],))[0]
                    assert worker['session_id'] == children[0]['id'] != sid
                    assert sum(request.get('model') == 'summary' for request in peer.requests) == 1
                    replay_params.update(params)
                    if ending == 'kill':
                        owner.kill()
                        await asyncio.to_thread(owner.wait, 10)
                        return
                    peer.release.set()
                    async with asyncio.timeout(45):
                        while rows("SELECT status FROM session_admissions WHERE request_id='follower'")[0]['status'] != 'terminal':
                            await asyncio.sleep(.03)
                    assert rows("SELECT status FROM session_admissions WHERE status!='terminal'") == []
                    assert rows("SELECT status FROM worker_executions WHERE status!='terminal'") == []
                    snapshot = await call('session.resume', session_id=sid)
                    assert 'FOLLOWER' in json.dumps(snapshot['messages'])
                    replay = await call('session.events.since', session_id=sid,
                                        replay_epoch=snapshot['replay_epoch'], last_sequence=0)
                    completed = [e for e in replay['events'] if e['type'] == 'message.complete']
                    assert any(e['admission_id'] == follower['admission_id'] for e in completed)
                    assert (await call('prompt.submit', **params))['admission_id'] == accepted['admission_id']
                    return
                finally:
                    peer.release.set()
            raise AssertionError('automatic compression did not publish')

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(exercise(desc, owner))
        if ending == 'kill':
            requests_before = len(peer.requests)

            async def recovered(desc):
                async with websocket(home, desc) as ws:
                    resumed = await rpc(ws, 'session.resume', session_id=replay_params['session_id'])
                    assert 'result' in resumed, resumed
                    retried = await rpc(ws, 'prompt.submit', **replay_params)
                    assert retried.get('result', {}).get('status') == 'unknown', retried
                    pending = rows("SELECT request_id,target_session_id,status FROM session_admissions WHERE status!='terminal'")
                    assert pending == [
                        {'request_id': replay_params['input_id'], 'target_session_id': replay_params['session_id'], 'status': 'unknown'},
                        {'request_id': 'follower', 'target_session_id': replay_params['session_id'], 'status': 'queued'}]
                    assert len(peer.requests) == requests_before
                    print(json.dumps({'restart_pending': pending, 'inference_requests': requests_before}))

            with daemon(root, home, env, barrier=False) as (_, desc):
                asyncio.run(recovered(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
