"""HTTP webhook to a real local Bot Chat owner; only the model endpoint is synthetic."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import hashlib
import hmac
import json
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


class WebhookModel(Model):
    def do_POST(self):
        # Reuse the wire model, but keep source instructions out of its final result.
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(body)
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if text == 'BLOCK_STARTED':
            self.server.blocked.set()
            self.server.release.wait(60)
        answer = '[SILENT]' if text == 'SOURCE_SILENT' else ('SOURCE_FINAL' if text == 'SOURCE_PRIVATE_PROMPT' else 'RECEIVER_ACK')
        message = {'role': 'assistant', 'content': answer}
        payload = json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def rows(home):
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM session_admissions ORDER BY seq')]


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('mode', ['standalone', 'named-standalone', 'multiplex', 'multiplex-bare', 'multiplex-dynamic', 'multiplex-explicit'])
def test_authenticated_webhook_receipt_through_real_owner(tmp_path, monkeypatch, mode):
    from gateway.platforms.webhook_bot_chat import result_id
    from tools.bot_live_delivery import read_delivery_result

    root = Path(__file__).resolve().parents[2]
    home = tmp_path / 'state' / 'profiles' / 'secondary' if mode != 'standalone' else tmp_path / 'state'
    launch = tmp_path / 'state' if mode.startswith('multiplex') else home
    target_home = launch / 'profiles' / 'receiver' if mode == 'multiplex-explicit' else home
    user = tmp_path / 'user'
    home.mkdir(mode=0o700, parents=True)
    user.mkdir(mode=0o700)
    # Both processes use this private cwd, giving the Unix socket a short spelling.
    runtime = tmp_path / 'runtime'
    runtime.mkdir(mode=0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_RUNTIME_DIR', '/proc/self/cwd/runtime')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HOME', str(user))
    peer = ThreadingHTTPServer(('127.0.0.1', 0), WebhookModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'standalone': True},
           'platforms': {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
               'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'bot-chat'}}}}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'streaming': {'enabled': False}, 'platform_toolsets': {'gui': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    if mode.startswith('multiplex'):
        cfg['gateway'] = {'multiplex_profiles': True}
        if mode == 'multiplex-dynamic':
            from argparse import Namespace
            from hermes_cli.webhook import _cmd_subscribe
            monkeypatch.setenv('HERMES_HOME', str(launch))
            args = Namespace(name='fixture', prompt='{text}', events='', description='', skills='',
                             deliver='bot-chat', deliver_chat_id='', secret='owned-secret',
                             route_profile='secondary', script='', cron_job='', deliver_only=False,
                             mirror_to_session=False)
            assert _cmd_subscribe(args) != 1
            assert (launch / 'webhook_subscriptions.json').exists()
            assert not (home / 'webhook_subscriptions.json').exists()
            cfg['platforms']['webhook']['extra']['routes'] = {}
            monkeypatch.setenv('HERMES_HOME', str(home))
        elif mode in ('multiplex', 'multiplex-explicit'):
            cfg['platforms']['webhook']['extra']['routes']['fixture']['profile'] = 'secondary'
        else:
            cfg['gateway']['profile_routes'] = [{'name': 'fixture-secondary', 'platform': 'webhook',
                                                  'user_id': 'webhook:fixture', 'profile': 'secondary'}]
        (launch / 'config.yaml').write_text(json.dumps(cfg))
        (home / 'config.yaml').write_text(json.dumps({'model': cfg['model'], 'auxiliary': cfg['auxiliary'],
                                                     'terminal': {'cwd': str(home)}, 'platform_toolsets': {'gui': []}}))
        if mode == 'multiplex-explicit':
            cfg['platforms']['webhook']['extra']['routes']['fixture']['deliver'] = 'bot-chat:receiver'
            target_home.mkdir(mode=0o700)
            (target_home / 'config.yaml').write_text((home / 'config.yaml').read_text())
            (launch / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env() | {'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(launch),
                         'PYTHONPATH': str(root), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': base,
                         'PYTHONUNBUFFERED': '1', 'XDG_RUNTIME_DIR': '/proc/self/cwd/runtime',
                         'HTTP_PROXY': 'http://127.0.0.1:1', 'HTTPS_PROXY': 'http://127.0.0.1:1',
                         'NO_PROXY': '127.0.0.1,localhost'}

    async def scenario(desc):
        if mode.startswith('multiplex'):
            async with websocket(launch, desc) as default_ws:
                sentinel = await rpc(default_ws, 'session.create', request_id='default-target', source='gui', cwd=str(launch), toolsets=[])
                sentinel_id = sentinel['result']['session_id']
                renamed = await rpc(default_ws, 'session.mutate', session_id=sentinel_id, request_id='default-title',
                                    expected_revision=sentinel['result']['revision'], operation='rename', payload={'title': 'Bot Chat'})
                assert 'result' in renamed, renamed
        async with websocket(target_home, desc, control_home=launch) as ws, aiohttp.ClientSession() as client:
            created = await rpc(ws, 'session.create', request_id='target', source='gui', cwd=str(home), toolsets=[])
            assert 'result' in created, created
            sid = created['result']['session_id']
            renamed = await rpc(ws, 'session.mutate', session_id=sid, request_id='title',
                                expected_revision=created['result']['revision'], operation='rename', payload={'title': 'Bot Chat'})
            assert 'result' in renamed, renamed
            held = await rpc(ws, 'prompt.submit', session_id=sid, input_id='held', text='BLOCK_STARTED')
            assert 'result' in held, held
            assert await asyncio.to_thread(peer.blocked.wait, 30)
            prefix = '/p/secondary' if mode in ('multiplex', 'multiplex-dynamic', 'multiplex-explicit') else ''
            url = f'http://127.0.0.1:{port}{prefix}/webhooks/fixture'

            async def post(event, text, signature=True):
                body = json.dumps({'text': text}).encode()
                headers = {'X-GitHub-Delivery': event, 'X-GitHub-Event': 'push'}
                if signature is not None:
                    headers['X-Hub-Signature-256'] = ('sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()
                                                       if signature else 'sha256=bad')
                async with client.post(url, data=body, headers=headers) as response:
                    return response.status, await response.text()

            assert (await post('missing', 'SOURCE_PRIVATE_PROMPT', None))[0] == 401
            assert (await post('wrong', 'SOURCE_PRIVATE_PROMPT', False))[0] == 401
            assert len(rows(home)) == (1 if home == target_home else 0)
            assert (await post('event-one', 'SOURCE_PRIVATE_PROMPT'))[0] == 202
            await async_wait_until(lambda: any(r['request_id'] == 'event-one' and r['status'] == 'terminal' for r in rows(home)), timeout=40, what='source completion')
            source = next(r for r in rows(home) if r['request_id'] == 'event-one')
            assert source['outcome'] == 'completed'
            key = result_id(home, source['admission_id'])
            await async_wait_until(lambda: (read_delivery_result(target_home, key) or {}).get('status') == 'queued', timeout=30, what='queued native receipt')
            queued = read_delivery_result(target_home, key)
            target = next(r for r in rows(target_home) if r['request_id'] == 'bot:' + key)
            assert target['target_session_id'] == sid and source['target_session_id'] != sid
            assert target['status'] == 'queued'
            assert (await post('event-one', 'SOURCE_PRIVATE_PROMPT'))[0] in (200, 202)
            peer.release.set()
            await async_wait_until(lambda: (read_delivery_result(target_home, key) or {}).get('status') == 'settled', timeout=40, what='settled native receipt')
            settled = read_delivery_result(target_home, key)
            assert (await post('event-one', 'SOURCE_PRIVATE_PROMPT'))[0] in (200, 202)
            assert (await post('event-two', 'SOURCE_PRIVATE_PROMPT'))[0] == 202
            await async_wait_until(lambda: any(r['request_id'] == 'event-two' and r['status'] == 'terminal' for r in rows(home)), timeout=40, what='second source completion')
            source2 = next(r for r in rows(home) if r['request_id'] == 'event-two')
            key2 = result_id(home, source2['admission_id'])
            await async_wait_until(lambda: (read_delivery_result(target_home, key2) or {}).get('status') == 'settled', timeout=40, what='second settled receipt')
            assert key != key2
            assert (await post('silent', 'SOURCE_SILENT'))[0] == 202
            await async_wait_until(lambda: any(r['request_id'] == 'silent' and r['status'] == 'terminal' for r in rows(home)), timeout=40, what='silent source completion')
            silent = next(r for r in rows(home) if r['request_id'] == 'silent')
            assert read_delivery_result(target_home, result_id(home, silent['admission_id'])) is None
            all_rows = rows(home) + (rows(target_home) if target_home != home else [])
            assert len([r for r in all_rows if r['request_id'] in ('event-one', 'event-two')]) == 2
            assert len([r for r in all_rows if r['request_id'].startswith('bot:')]) == 2
            if mode.startswith('multiplex'):
                assert rows(launch) == []
                assert sid != sentinel_id
                assert settled['profile_home'] == str(target_home)
                assert read_delivery_result(launch, key) is None
            inputs = [next((m.get('content', '') for m in reversed(req['messages']) if m['role'] == 'user'), '') for req in peer.requests if req.get('messages')]
            assert inputs.count('SOURCE_PRIVATE_PROMPT') == 2
            wrapped = [text for text in inputs if isinstance(text, str) and 'Webhook automation' in text]
            assert len(wrapped) == 2 and all('SOURCE_FINAL' in text and 'SOURCE_PRIVATE_PROMPT' not in text for text in wrapped)
            print(json.dumps({'source_id': source['admission_id'], 'target_id': target['admission_id'],
                              'source_session': source['target_session_id'], 'target_session': sid,
                              'queued': queued, 'settled': settled, 'second_key': key2, 'inputs': inputs}))

    try:
        with daemon(tmp_path, launch, env, barrier=False) as (_, desc):
            try:
                asyncio.run(scenario(desc))
            except BaseException:
                print('DIAGNOSTIC ROWS', json.dumps(rows(home)))
                print('DIAGNOSTIC MODEL', json.dumps(peer.requests))
                print('DIAGNOSTIC LOG', (launch / 'restart.log').read_text())
                raise
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
