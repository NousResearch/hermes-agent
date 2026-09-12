"""Public owner controls never become a second admission or change the cached prefix."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest
from websockets.asyncio.client import connect

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket


class CorrectionModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        text = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
        message = {'role': 'assistant', 'content': 'CONTROL_ACK'}
        finish = 'stop'
        if text in {'BLOCK_STEER', 'BLOCK_REDIRECT'}:
            self.server.blocked.set()
            self.server.release.wait(30)
            if text == 'BLOCK_STEER':
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'owned-tool', 'type': 'function', 'function': {
                        'name': 'terminal', 'arguments': json.dumps({'command': 'printf controls', 'timeout': 5})}}]}
                finish = 'tool_calls'
        payload = json.dumps({'id': 'controls', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                              'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            delta = dict(message)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(delta['tool_calls'][0], index=0)]
            payload = ('data: ' + json.dumps({'id': 'controls', 'choices': [{'index': 0, 'delta': delta,
                'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextmanager
def owner(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), CorrectionModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'display': {'busy_input_mode': 'steer'},
        'model': {'provider': 'custom', 'default': 'controls-fixture', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}},
    }))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
        OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1',
        HERMES_DASHBOARD_SESSION_TOKEN='other-controls-actor')
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            yield home, peer, desc
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


def admissions(home):
    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
        return list(db.execute('SELECT request_id,status,seq FROM session_admissions ORDER BY seq'))


async def settled(home):
    async with asyncio.timeout(40):
        while not admissions(home) or any(r[1] != 'terminal' for r in admissions(home)):
            await asyncio.sleep(.05)


@pytest.mark.linux_only
def test_busy_policy_is_authorized_session_scoped_and_not_inference(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                ids = []
                for name in ('a', 'b'):
                    result = await rpc(ws, 'session.create', request_id=name, source='tui', toolsets=[])
                    ids.append(result['result']['session_id'])
                config_before = (home / 'config.yaml').read_bytes()
                result = await rpc(ws, 'config.get', session_id=ids[0], key='busy')
                assert result.get('result', {}).get('value') == 'steer', result
                result = await rpc(ws, 'config.set', session_id=ids[0], key='busy', value='queue')
                assert result.get('result', {}).get('scope') == 'session', result
                async with websocket(home, desc) as viewer:
                    assert (await rpc(viewer, 'config.get', session_id=ids[0], key='busy'))['result']['value'] == 'queue'
                assert (await rpc(ws, 'config.get', session_id=ids[1], key='busy'))['result']['value'] == 'steer'
                for method, params, reason in [
                    ('config.get', {'session_id': ids[0], 'key': 'busy', 'profile': 'foreign'}, 'profile_mismatch'),
                    ('config.set', {'key': 'busy', 'value': 'queue'}, 'invalid_params'),
                    ('config.set', {'session_id': ids[0], 'key': 'model', 'value': 'other'}, 'invalid_params'),
                ]:
                    result = await rpc(ws, method, **params)
                    assert result.get('error', {}).get('message') == reason, result
                other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=other-controls-actor'
                async with connect(other_url) as other:
                    for method in ('config.get', 'config.set'):
                        params = {'value': 'queue'} if method == 'config.set' else {}
                        result = await rpc(other, method, session_id=ids[0], key='busy', **params)
                        # This configured dashboard token is another verified operator.
                        assert result.get('result') == {
                            'key': 'busy', 'value': 'queue', 'scope': 'session'}, result
                assert (await rpc(ws, 'config.get', session_id=ids[0], key='busy'))['result']['value'] == 'queue'
                assert (await rpc(ws, 'config.get', session_id=ids[1], key='busy'))['result']['value'] == 'steer'
                assert (home / 'config.yaml').read_bytes() == config_before
                assert not admissions(home) and not peer.requests
        asyncio.run(probe())


@pytest.mark.linux_only
def test_corrections_are_generation_fenced_and_consumed_by_same_provider_loop(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                created = await rpc(ws, 'session.create', request_id='controls', source='tui', toolsets=['terminal'])
                sid = created['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM')
                await settled(home)
                for verb, status in [('steer', 'queued'), ('redirect', 'redirected')]:
                    peer.blocked.clear()
                    peer.release.clear()
                    await rpc(ws, 'prompt.submit', session_id=sid, input_id=verb, text='BLOCK_' + verb.upper())
                    assert await asyncio.to_thread(peer.blocked.wait, 25)
                    snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
                    generation = snapshot['execution_generation']
                    for extra, reason in [({'execution_generation': generation - 1}, 'stale_generation'),
                                          ({'profile': 'foreign'}, 'profile_mismatch')]:
                        params = {'session_id': sid, 'text': 'DENIED_MARKER', 'execution_generation': generation, **extra}
                        denied = await rpc(ws, 'session.' + verb, **params)
                        assert denied.get('error', {}).get('message') == reason, denied
                    other_url = desc['api_origin'].replace('http:', 'ws:') + '/api/ws?token=other-controls-actor'
                    async with connect(other_url) as other:
                        # Operator sharing does not bypass the execution-generation fence.
                        denied = await rpc(other, 'session.' + verb, session_id=sid,
                                           execution_generation=generation - 1, text='DENIED_MARKER')
                        assert denied.get('error', {}).get('message') == 'stale_generation', denied
                        result = await rpc(other, 'session.' + verb, session_id=sid,
                                           execution_generation=generation, text='CORRECT_' + verb.upper())
                    assert result.get('result', {}).get('status') == status, result
                    assert result['result']['execution_generation'] == generation
                    queue = await rpc(ws, 'prompt.submit', session_id=sid, input_id=verb + '-queue', text='FIFO_' + verb.upper())
                    assert queue['result']['status'] == 'queued'
                    peer.release.set()
                    await settled(home)
                    corrected = [r for r in peer.requests if 'CORRECT_' + verb.upper() in json.dumps(r['messages'])]
                    assert corrected, peer.requests
                    assert not any('DENIED_MARKER' in json.dumps(r) for r in peer.requests)
                    stale = await rpc(ws, 'session.' + verb, session_id=sid,
                                      execution_generation=generation, text='TOO_LATE')
                    assert stale.get('error', {}).get('message') == 'stale_generation', stale
                assert [r[0] for r in admissions(home)] == ['warm', 'steer', 'steer-queue', 'redirect', 'redirect-queue']
                prefixes = [[m for m in r['messages'] if m['role'] in {'system', 'developer'}] for r in peer.requests]
                assert prefixes and all(p == prefixes[0] for p in prefixes)
                tools = [r.get('tools') for r in peer.requests]
                assert all(t == tools[0] for t in tools)
                print(json.dumps({'ledger': admissions(home), 'model_requests': len(peer.requests),
                                  'same_prefix': True, 'corrections_consumed': True}))
        asyncio.run(probe())
