"""Ordinary daemon + concurrent classic CLI, loopback-only launch receipts."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

from tests.gateway.fixtures.local_recovery_probe import daemon, websocket, rpc


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if not body.get('messages'):
            self.send_error(404)
            return
        with self.server.lock:
            first = not any(r['body']['model'] == body['model'] for r in self.server.requests)
            self.server.requests.append({'path': self.path, 'auth': self.headers.get('Authorization'), 'body': body})
        if first:
            self.server.barrier.wait(timeout=30)
        message = {'role': 'assistant', 'content': 'LAUNCH_POLICY_OK'}
        budget = any('BUDGET' in str(m.get('content', '')) for m in body['messages'] if m['role'] == 'user')
        if budget and body.get('tools'):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'index': 0, 'id': 'budget-' + str(len(body['messages'])), 'type': 'function',
                'function': {'name': 'terminal', 'arguments': json.dumps({'command': 'printf budget'})}}]}
        choice = {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}
        kind = 'application/json'
        payload = json.dumps({'id': 'local', 'choices': [choice], 'model': body['model'],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        if body.get('stream'):
            choice['delta'] = choice.pop('message')
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [choice]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def probe(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.lock, peer.barrier = [], threading.Lock(), threading.Barrier(2)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{peer.server_port}'
    cfg = {'model': {'provider': 'custom', 'default': 'system-model', 'base_url': origin + '/system/v1'},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []},
           'memory': {'memory_enabled': False, 'user_profile_enabled': False}}
    config = home / 'config.yaml'
    config.write_text(json.dumps(cfg))
    before = config.read_bytes()
    (home / 'SOUL.md').write_text('SOUL_LAUNCH_MARKER')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root))
    sessions, pids = {}, []
    keys = {'left': 'private-left-launch-token', 'right': 'private-right-launch-token'}

    def cli(side, resume=None, budget=False):
        cwd = tmp_path / side
        cwd.mkdir(exist_ok=True)
        (cwd / 'AGENTS.md').write_text('PROJECT_LAUNCH_MARKER_' + side)
        args = dict(query=('BUDGET_' if budget else 'POLICY_') + side, quiet=True)
        if resume:
            args['resume'] = resume
        else:
            args.update(provider='custom', model='gpt-5-' + side, api_key=keys[side],
                        base_url=origin + '/' + side + '/v1', reasoning='high' if side == 'left' else 'low',
                        max_turns=1 if side == 'left' else 3, ignore_rules=side == 'left', toolsets='terminal')
        result = subprocess.run([sys.executable, '-c', 'import cli; cli.main(**' + repr(args) + ')'],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=70)
        assert result.returncode == 0, result.stdout + result.stderr
        if not resume:
            sessions[side] = re.search(r'Session: (\S+)', result.stderr).group(1)
        return result.stdout

    async def unauthorized(desc):
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus
        try:
            async with connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws') as ws:
                denied = await rpc(ws, 'session.create', request_id='unauthorized',
                    api_key='must-not-be-stored', provider='custom', base_url=origin + '/denied/v1')
                assert denied['error']['message'] == 'permission_denied', denied
        except InvalidStatus as exc:
            assert exc.response.status_code in (401, 403)

    async def after_restart(desc):
        async with websocket(home, desc) as ws:
            for side, sid in sessions.items():
                result = await rpc(ws, 'session.resume', session_id=sid)
                assert 'result' in result, result
                assert 'LAUNCH_POLICY_OK' in json.dumps(result['result']['messages'])
                replay_epoch = result['result']['replay_epoch']
                result = await rpc(ws, 'prompt.submit', session_id=sid, input_id='restart-' + side, text='NEVER_INFER')
                assert 'result' in result, result
                admission = result['result']['admission_id']
                async with asyncio.timeout(20):
                    while True:
                        result = await rpc(ws, 'prompt.receipt', session_id=sid, admission_id=admission)
                        if result['result']['status'] == 'terminal':
                            break
                        await asyncio.sleep(.05)
                events = await rpc(ws, 'session.events.since', session_id=sid, last_sequence=0, replay_epoch=replay_epoch)
                assert 'launch_credentials_unavailable' in json.dumps(events), events

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(unauthorized(desc))
            with ThreadPoolExecutor(2) as pool:
                assert all('LAUNCH_POLICY_OK' in x for x in pool.map(cli, keys))
            assert config.read_bytes() == before
            cfg['agent'] = {'reasoning_effort': 'none', 'max_turns': 40}
            config.write_text(json.dumps(cfg))
            for side in keys:
                assert 'LAUNCH_POLICY_OK' in cli(side, sessions[side])
            for side in keys:
                captured = [x for x in peer.requests if x['body']['model'] == 'gpt-5-' + side]
                assert len(captured) == 2, captured
                assert all(x['path'] == '/' + side + '/v1/chat/completions' for x in captured)
                assert all(x['auth'] == 'Bearer ' + keys[side] for x in captured)
                bodies = [x['body'] for x in captured]
                assert all(b.get('reasoning_effort') == ('high' if side == 'left' else 'low') for b in bodies), bodies
                systems = [[m for m in b['messages'] if m['role'] in {'system', 'developer'}] for b in bodies]
                assert systems[0] == systems[1]
                text = json.dumps(systems[0])
                assert ('PROJECT_LAUNCH_MARKER_' + side in text) == (side == 'right'), text
                assert ('SOUL_LAUNCH_MARKER' in text) == (side == 'right'), text
            for side, expected in [('left', 1), ('right', 3)]:
                start = len(peer.requests)
                cli(side, sessions[side], budget=True)
                rounds = peer.requests[start:]
                assert sum(bool(x['body'].get('tools')) for x in rounds) == expected, rounds
        count = len(peer.requests)
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(after_restart(desc))
            assert len(peer.requests) == count
        leaks = []
        backups = home / 'backups' / 'config'  # copies of config.yaml itself (#109463), not a daemon leak
        for path in home.rglob('*'):
            if path.is_file() and backups not in path.parents \
                    and any(key.encode() in path.read_bytes() for key in keys.values()):
                leaks.append(str(path.relative_to(home)))
        assert not leaks, leaks
        return {'pids': pids, 'requests': count, 'concurrent_policies': True, 'auth_endpoint_model_reasoning': True,
                'frozen_prefix': True, 'max_turns': [1, 3], 'unauthorized_refused': True, 'ignore_rules': True, 'restart_fail_closed': True, 'no_durable_keys': True}
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
