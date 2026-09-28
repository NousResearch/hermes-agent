"""Real daemon restart driver; model and preclaim crash barriers are owned fixtures."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from websockets.asyncio.client import connect
from tests.gateway.test_normal_runtime_boot import control


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if text == 'BLOCK_STARTED':
            self.server.blocked.set()
            self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'RECOVERY_ACK_' + str(text)}
        payload = json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
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


# Windows children need the system environment (Winsock, temp, PATHEXT); POSIX uses the first three.
CHILD_ENV_KEYS = ('PATH', 'LANG', 'TZ', 'SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'PATHEXT',
                  'LOCALAPPDATA', 'APPDATA', 'PROGRAMDATA', 'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE')


def child_env():
    return {k: os.environ[k] for k in CHILD_ENV_KEYS if k in os.environ}


async def rpc(ws, method, **params):
    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    async with asyncio.timeout(20):
        while True:
            reply = json.loads(await ws.recv())
            if reply.get('id') == method:
                return reply


def websocket(home, desc):
    grant = control(home, 'session-ticket', {'profile_id': str(home), 'instance_id': desc['instance_id'], 'purpose': 'interactive'})
    return connect(desc['api_origin'].replace('http:', 'ws:') + '/api/ws',
                   subprotocols=['hermes-gateway-v1', 'hermes-gateway-ticket.' + grant['ticket']])


@contextmanager
def daemon(root, home, env, *, barrier, fixture='local_recovery_daemon.py'):
    command = [sys.executable, str(root / 'tests/gateway/fixtures' / fixture)] if barrier else [sys.executable, '-m', 'gateway.run']
    with (home / ('first.log' if barrier else 'restart.log')).open('w+') as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            desc = {}
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    desc = control(home, 'identify')
                    if desc.get('state') == 'ready':
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.05)
            log.flush()
            log.seek(0)
            assert desc.get('state') == 'ready', log.read()
            yield proc, desc
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


def probe(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'old-default', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    sessions, params, pids, epochs = {}, {}, [], []
    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return dict(db.execute('SELECT request_id,status FROM session_admissions'))
    async def first(desc, proc):
        async with websocket(home, desc) as ws:
            for name in ('safe', 'unknown', 'foreign', 'missing', 'corrupt'):
                params[name] = dict(request_id=name, source='gui' if name == 'safe' else 'tui', cwd=str(home), model='frozen-' + name, toolsets=[])
                result = await rpc(ws, 'session.create', **params[name])
                assert 'result' in result, result
                sessions[name] = result['result']['session_id']
            warm = await rpc(ws, 'prompt.submit', session_id=sessions['safe'], input_id='warm', text='WARM_HISTORY')
            assert warm['result']['status'] == 'queued', warm
            async with asyncio.timeout(30):
                while rows().get('warm') != 'terminal':
                    await asyncio.sleep(.02)
            await rpc(ws, 'prompt.submit', session_id=sessions['unknown'], input_id='started', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            await rpc(ws, 'prompt.submit', session_id=sessions['unknown'], input_id='follower', text='NEVER_REPLAY')
            for name in ('foreign', 'missing', 'corrupt'):
                await rpc(ws, 'prompt.submit', session_id=sessions[name], input_id=name, text=name.upper() + '_QUEUE')
            # The daemon stops itself immediately after this admission commits.
            await ws.send(json.dumps({'id': 'kill-boundary', 'method': 'prompt.submit', 'params': {
                'session_id': sessions['safe'], 'input_id': 'safe', 'text': 'RECOVER_QUEUED'}}))
            async with asyncio.timeout(20):
                while rows().get('safe') != 'queued':
                    await asyncio.sleep(.02)
            assert rows()['foreign'] == 'queued', rows()
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
    async def second(desc):
        async with websocket(home, desc) as ws:
            async with asyncio.timeout(35):
                while rows().get('safe') != 'terminal':
                    await asyncio.sleep(.05)
            result = await rpc(ws, 'session.resume', session_id=sessions['safe'])
            assert 'result' in result, result
            history = json.dumps(result['result']['messages'])
            assert 'RECOVERY_ACK_WARM_HISTORY' in history and 'RECOVERY_ACK_RECOVER_QUEUED' in history, result
            same = await rpc(ws, 'session.create', **params['safe'])
            assert same['result']['session_id'] == sessions['safe'], same
            assert same['result']['info']['source'] == 'gui' and same['result']['info']['model'] == 'frozen-safe'
            conflict = await rpc(ws, 'session.create', **{**params['safe'], 'model': 'changed'})
            assert conflict['error']['message'] == 'invalid_params', conflict
            for name, reason in [('foreign', 'profile_mismatch'), ('missing', 'storage_unavailable'), ('corrupt', 'storage_unavailable')]:
                rejected = await rpc(ws, 'session.resume', session_id=sessions[name])
                assert rejected['error']['message'] == reason, rejected
            unknown = await rpc(ws, 'session.resume', session_id=sessions['unknown'])
            assert any(r['status'] == 'unknown' for r in unknown['result']['pending']), unknown
    try:
        with daemon(root, home, env, barrier=True) as (proc, desc):
            pids.append(proc.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(first(desc, proc))
        peer.release.set()
        # Negative fixtures corrupt only owned durable policy after the actual crash.
        with sqlite3.connect(home / 'state.db') as db:
            prefix = 'gateway.local_policy.v1:'
            row = json.loads(db.execute('SELECT value FROM state_meta WHERE key=?', (prefix + sessions['foreign'],)).fetchone()[0])
            row['profile_id'] = str(user)
            db.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(row), prefix + sessions['foreign']))
            db.execute('DELETE FROM state_meta WHERE key=?', (prefix + sessions['missing'],))
            db.execute('UPDATE state_meta SET value=? WHERE key=?', ('{}', prefix + sessions['corrupt']))
        cfg['model']['default'] = 'changed-default'
        cfg['platform_toolsets']['cli'] = ['terminal']
        (home / 'config.yaml').write_text(json.dumps(cfg))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(second(desc))
            assert rows() == {'warm': 'terminal', 'started': 'unknown', 'follower': 'queued', 'foreign': 'queued', 'missing': 'queued', 'corrupt': 'queued', 'safe': 'terminal'}, rows()
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(second(desc))
        texts = [next((m.get('content') for m in reversed(r['messages']) if m['role'] == 'user'), '') for r in peer.requests]
        assert texts.count('BLOCK_STARTED') == 1 and texts.count('RECOVER_QUEUED') == 1, texts
        assert not any(t in texts for t in ('NEVER_REPLAY', 'FOREIGN_QUEUE', 'MISSING_QUEUE', 'CORRUPT_QUEUE')), texts
        recovered = next(r for r in peer.requests if any(m.get('content') == 'RECOVER_QUEUED' for m in r['messages']))
        assert recovered['model'] == 'frozen-safe' and not recovered.get('tools'), recovered
        assert epochs[1] > epochs[0] and pids[0] != pids[1]
        return {'pids': pids, 'epochs': epochs, 'sessions': sessions, 'inference_texts': texts,
                'queued_recovered': True, 'unknown_paused': True, 'foreign_missing_corrupt_refused': True,
                'history_retained': True, 'frozen_model_toolsets': True}
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
