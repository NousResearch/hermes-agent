"""Real-import HTTP -> direct/hook/coalesced origin -> native FIFO/loopback model receipts."""
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

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_webhook_bot_chat_integration import WebhookModel, rows
from tests.waiting import async_wait_until


class OriginsModel(WebhookModel):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        messages = body.get('messages', [])
        text = next((m.get('content', '') for m in reversed(messages) if m['role'] == 'user'), '')
        finish = 'stop'
        message = {'role': 'assistant', 'content': 'RECEIVER_ACK'}
        if text == 'SOURCE_PROCESS':
            if any(m['role'] == 'tool' for m in messages):
                message['content'] = 'PROCESS_SOURCE_FINAL'
            else:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'real-process-tool',
                    'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({
                        'command': 'sleep 2; printf PROCESS_OUTPUT' + ('; exit 4' if self.server.fail_process else ''),
                        'background': True, 'notify': True})}}]}
                finish = 'tool_calls'
        elif 'PROCESS_OUTPUT' in str(text) and 'Webhook automation' not in str(text):
            message['content'] = 'PROCESS_COMPLETION_FINAL'
        payload = json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            if message.get('tool_calls'):
                message['tool_calls'][0]['index'] = 0
            payload = ('data: ' + json.dumps({'id': 'local', 'choices': [{'index': 0, 'delta': message,
                'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('process_failure', [False, True])
def test_direct_hook_and_existing_coalescer_real_native_receipts(tmp_path, monkeypatch, record_property, process_failure):
    from gateway.platforms.webhook_bot_chat import result_id
    from tools.bot_live_delivery import read_delivery_result
    root = Path(__file__).resolve().parents[2]
    home, user, runtime = (tmp_path / name for name in ('state', 'user', 'runtime'))
    for directory in (home, user, runtime):
        directory.mkdir(mode=0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(user))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('XDG_RUNTIME_DIR', '/proc/self/cwd/runtime')
    peer = ThreadingHTTPServer(('127.0.0.1', 0), OriginsModel)
    peer.requests = []
    peer.fail_process = process_failure
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    routes = {'direct': {'prompt': '{text}', 'deliver': 'bot-chat', 'deliver_only': True},
              'hook': {'prompt': '{text}', 'deliver': 'bot-chat'},
              'process': {'prompt': '{text}', 'deliver': 'bot-chat'},
              'group': {'prompt': '{text}', 'deliver': 'bot-chat',
                        'coalesce': {'key': '{group}', 'debounce_seconds': 0.2, 'max_wait_seconds': 2}}}
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'standalone': True},
           'platforms': {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
               'secret': 'owned-secret', 'routes': routes}}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'display': {'suppress_warning_notifications': process_failure},
           'streaming': {'enabled': False}, 'platform_toolsets': {'gui': [], 'webhook': ['terminal']}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env() | {'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
        'PYTHONPATH': str(root), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': base,
        'PYTHONUNBUFFERED': '1', 'XDG_RUNTIME_DIR': '/proc/self/cwd/runtime'}

    async def scenario(desc):
        async with websocket(home, desc) as ws, aiohttp.ClientSession() as client:
            created = await rpc(ws, 'session.create', request_id='target', source='gui', cwd=str(home), toolsets=[])
            sid = created['result']['session_id']
            renamed = await rpc(ws, 'session.mutate', session_id=sid, request_id='title',
                expected_revision=created['result']['revision'], operation='rename', payload={'title': 'Bot Chat'})
            assert 'result' in renamed, renamed
            async def post(route, key, text):
                body = json.dumps({'text': text, 'group': 'same'}).encode()
                headers = {'X-GitHub-Delivery': key, 'X-Hub-Signature-256':
                    'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
                async with client.post(f'http://127.0.0.1:{port}/webhooks/{route}', data=body, headers=headers) as response:
                    return response.status, await response.json()
            evidence = []
            for route, text, kind in [('direct', 'DIRECT_ACTUAL_TEMPLATE', 'direct_template'), ('hook', 'HOOK_INPUT', 'hook_reply')]:
                status, value = await post(route, route + '-request', text)
                assert status == 202, value
                assert value['origin_kind'] == kind and value['status'] in {'queued', 'claimed', 'settled'}
                key = value['delivery_id']
                await async_wait_until(lambda: (read_delivery_result(home, key) or {}).get('status') == 'settled',
                    timeout=40, what=kind + ' native settlement')
                receipt = read_delivery_result(home, key)
                assert receipt['session_id'] == sid and receipt['reply'] == 'RECEIVER_ACK'
                assert (await post(route, route + '-request', text))[1]['delivery_id'] == key
                evidence.append(receipt)
            assert not any(r['request_id'] in {'direct-request', 'hook-request'} for r in rows(home))
            assert (await post('group', 'superseded', 'SOURCE_PRIVATE_PROMPT'))[0] == 202
            assert (await post('group', 'latest', 'SOURCE_PRIVATE_PROMPT'))[0] == 202
            await async_wait_until(lambda: any(r['request_id'] == 'latest' and r['status'] == 'terminal' for r in rows(home)),
                timeout=40, what='coalesced real source final')
            source, = [r for r in rows(home) if r['request_id'] == 'latest']
            key = result_id(home, source['admission_id'])
            await async_wait_until(lambda: (read_delivery_result(home, key) or {}).get('status') == 'settled',
                timeout=40, what='coalesced native receipt')
            assert not any(r['request_id'] == 'superseded' for r in rows(home))
            evidence.append(read_delivery_result(home, key))
            inputs = [next((m.get('content', '') for m in reversed(req['messages']) if m['role'] == 'user'), '')
                      for req in peer.requests if req.get('messages')]
            assert 'DIRECT_ACTUAL_TEMPLATE' not in inputs and 'HOOK_INPUT' not in inputs
            grouped = [text for text in inputs if isinstance(text, str) and text.startswith('SOURCE_PRIVATE_PROMPT')]
            assert len(grouped) == 1 and 'earlier events are superseded' in grouped[0]
            assert len([text for text in inputs if 'Webhook automation' in str(text)]) == 3
            assert (await post('process', 'process-request', 'SOURCE_PROCESS'))[0] == 202
            def completions():
                return [r for r in rows(home) if json.loads(r['payload_json']).get('native_text_v1', {}).get('webhook_root')]
            await async_wait_until(lambda: any(r['status'] == 'terminal' for r in completions()),
                timeout=60, what='actual process completion agent final')
            completion, = completions()
            assert completion['outcome'] == 'completed'
            native = json.loads(completion['payload_json'])['native_text_v1']
            root_row, = [r for r in rows(home) if r['request_id'] == 'process-request']
            assert native['webhook_root'] == root_row['admission_id']
            def completion_intent():
                with closing(sqlite3.connect(home / 'state.db')) as db:
                    row = db.execute('SELECT value FROM state_meta WHERE key=?',
                        ('webhook.final.intent.v1.' + completion['admission_id'],)).fetchone()
                return json.loads(row[0]) if row else None
            await async_wait_until(lambda: completion_intent() is not None, timeout=40, what='durable completion intent')
            value = completion_intent()
            assert value is not None
            key = value['delivery_id']
            # A requested completion-agent final is a result even for nonzero process exit;
            # source warning suppression does not reclassify it as an automatic diagnostic.
            assert value['category'] == 'result'
            await async_wait_until(lambda: (read_delivery_result(home, key) or {}).get('status') == 'settled',
                timeout=40, what='actual completion final native settlement')
            receipt = read_delivery_result(home, key)
            assert receipt is not None
            assert receipt['message'].endswith('PROCESS_COMPLETION_FINAL'), receipt
            assert receipt['session_id'] == sid and key != result_id(home, root_row['admission_id'])
            evidence.append(receipt)
            record_property('native_receipts', json.dumps({'receipts': evidence, 'source_admission': source['admission_id'],
                'completion_admission': completion['admission_id'], 'process_identity': native['automation']['identity'],
                'completion_intent': value}))
            return evidence
    try:
        with daemon(tmp_path, home, env, barrier=True,
                    fixture=str(root / 'tests/gateway/fixtures/webhook_origins_daemon.py')) as (_, desc):
            try:
                evidence = asyncio.run(scenario(desc))
                assert (home / 'first.log').read_text().count('HOOK_CONSUMED hook-request') == 1
            except BaseException:
                print('DAEMON_LOG', (home / 'first.log').read_text())
                print('MODEL', json.dumps(peer.requests))
                raise
        prior_rows = rows(home)
        async def replay():
            async with aiohttp.ClientSession() as client:
                for route, text, receipt in [('direct', 'DIRECT_ACTUAL_TEMPLATE', evidence[0]),
                                             ('hook', 'HOOK_INPUT', evidence[1])]:
                    body = json.dumps({'text': text, 'group': 'same'}).encode()
                    headers = {'X-GitHub-Delivery': route + '-request', 'X-Hub-Signature-256':
                        'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
                    async with client.post(f'http://127.0.0.1:{port}/webhooks/{route}', data=body, headers=headers) as response:
                        assert response.status == 202
                        value = await response.json()
                        assert value['delivery_id'] == receipt['delivery_id'] and value['status'] == 'settled'
        # The ordinary gateway has no consumer installed. Durable readback, not the hook,
        # resolves both prior origins after an actual process restart.
        with daemon(tmp_path, home, env, barrier=False):
            asyncio.run(replay())
            assert rows(home) == prior_rows
            # Read the real producer ledgers through the authenticated operator API.
            # This does not infer settlement from acceptance or manufacture receipts.
            from starlette.testclient import TestClient
            from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
            operator = TestClient(app)
            operator.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
            request_count = len(peer.requests)
            response = operator.get('/api/webhooks/deliveries')
            assert response.status_code == 200, response.text
            projected = response.json()
            by_delivery = {item['delivery_id']: item for item in projected['deliveries']
                           if item['origin_kind'] != 'linked_native_admission'}
            assert {item['namespace'] for item in projected['deliveries']} == {
                'webhook.final.intent.v1', 'webhook.origin.v1'}
            for receipt in evidence:
                item = by_delivery[receipt['delivery_id']]
                assert item['native_admission_id'] == receipt['admission_id']
                assert item['native_status'] in {'queued', 'claimed', 'settled'}
                assert item['source_outcome'] == 'completed'
            assert all(by_delivery[item['delivery_id']]['state'] == 'settled' for item in evidence[:2])
            assert not any(text in response.text for text in ('DIRECT_ACTUAL_TEMPLATE', 'HOOK_INPUT',
                'SOURCE_PRIVATE_PROMPT', 'PROCESS_COMPLETION_FINAL', 'owned-secret', str(home)))
            assert len(peer.requests) == request_count and rows(home) == prior_rows
            record_property('operator_states', json.dumps(projected))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
