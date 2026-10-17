"""Desktop's ``llm.oneshot`` on the shared gateway borrows the chat's model, as main's sidecar did.

Main's sidecar looked the session up in its own table and lent its live agent's route
(``_main_runtime_from_agent``). On the shared gateway that table never holds an owner session, so a
session-bound one-shot (commit message, approval summary) silently ran on the ``task`` auxiliary
backend (``auto``: the profile's default model, not the chat's). Session-less one-shots (project
ideas) keep the task backend."""
import asyncio
from contextlib import contextmanager
import io
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


async def _settled(ws, sid, marker):
    deadline = time.monotonic() + 45
    while True:
        snap = (await rpc(ws, 'session.resume', session_id=sid))['result']
        if not snap['running'] and not snap['pending'] and marker in json.dumps(snap['messages']):
            return
        assert time.monotonic() < deadline, ('turn never settled', snap)
        await asyncio.sleep(.1)


class _HeaderModel(Model):
    """The loopback model, also recording each request's ``X-Route-Token`` with its body."""

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        self.server.tokens.append((self.headers.get('X-Route-Token'), raw.decode()))
        self.rfile = io.BytesIO(raw)
        super().do_POST()


@contextmanager
def _gateway(tmp_path, named=False):
    """A real gateway on a loopback model whose ``BLOCK_STARTED`` requests wait for ``release``;
    ``named`` serves the model as a named custom provider that requires its ``extra_headers``."""
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    work.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _HeaderModel)
    peer.requests, peer.tokens = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    model = {'provider': 'custom', 'default': 'chat-model', 'base_url': url}
    if named:
        model = {'provider': 'custom:edge', 'default': 'chat-model'}
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False},
        'model': model,
        'custom_providers': [{'name': 'edge', 'base_url': url, 'api_key': 'loopback-only',
                              'extra_headers': {'X-Route-Token': 'edge-token'}}] if named else [],
        # The task's default ``auto`` route: the borrowed session runtime, else the profile's model.
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            yield home, work, peer, desc
            peer.release.set()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=20)
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


@pytest.mark.platforms("linux")
def test_session_bound_oneshot_uses_the_chats_model_and_never_blocks_the_socket(tmp_path):
    out = {}

    async def probe(home, work, peer, desc):
        def model_for(text):
            return next(r['model'] for r in reversed(peer.requests)
                        if any(m.get('content') == text for m in r['messages']))

        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='rt7', source='gui', toolsets=[], cwd=str(work),
                                model='session-model')
            sid = created['result']['session_id']
            # No turn yet: no resident agent, the session's own frozen route.
            out['cold'] = await rpc(ws, 'llm.oneshot', session_id=sid, instructions='sys', input='ONESHOT_COLD')
            out['cold_model'] = model_for('ONESHOT_COLD')
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM')
            await _settled(ws, sid, 'RECOVERY_ACK_WARM')
            out['turn_model'] = model_for('WARM')
            out['warm'] = await rpc(ws, 'llm.oneshot', session_id=sid, instructions='sys', input='ONESHOT_WARM')
            out['warm_model'] = model_for('ONESHOT_WARM')
            out['free'] = await rpc(ws, 'llm.oneshot', instructions='sys', input='ONESHOT_FREE')
            out['free_model'] = model_for('ONESHOT_FREE')
            out['bad'] = await rpc(ws, 'llm.oneshot', session_id=sid, instructions='sys', input='x', bogus=1)
            # A one-shot waiting on its model never holds up the next request on this socket.
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 'slow', 'method': 'llm.oneshot',
                                      'params': {'session_id': sid, 'input': 'BLOCK_STARTED'}}))
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            out['ping'] = await rpc(ws, 'ping')
            peer.release.set()
            async with asyncio.timeout(20):
                while (reply := json.loads(await ws.recv())).get('id') != 'slow':
                    pass
            out['slow'] = reply
    with _gateway(tmp_path) as served:
        asyncio.run(probe(*served))
    assert out['cold']['result']['text'] == 'RECOVERY_ACK_ONESHOT_COLD', out
    assert out['turn_model'] == 'session-model', out
    assert (out['cold_model'], out['warm_model']) == ('session-model', 'session-model'), out
    assert out['warm']['result']['text'] == 'RECOVERY_ACK_ONESHOT_WARM', out
    assert out['free_model'] == 'chat-model', out
    assert out['free']['result']['text'] == 'RECOVERY_ACK_ONESHOT_FREE', out
    assert out['bad']['error']['message'] == 'invalid_params', out
    assert out['ping'].get('result') == {'pong': True}, out
    assert out['slow']['result']['text'] == 'RECOVERY_ACK_BLOCK_STARTED', out


@pytest.mark.platforms("linux")
def test_a_borrowed_named_provider_route_keeps_its_extra_headers(tmp_path):
    """The borrowed route reaches the aux client as provider ``custom`` with the session's endpoint and
    key; the named entry's required ``extra_headers`` still ride on it, cold and resident alike."""
    out = {}

    async def probe(home, work, peer, desc):
        def token_for(text):
            return next(token for token, raw in reversed(peer.tokens) if json.dumps(text) in raw)

        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='hdr', source='gui', toolsets=[], cwd=str(work),
                                model='session-model')
            sid = created['result']['session_id']
            out['cold'] = await rpc(ws, 'llm.oneshot', session_id=sid, instructions='sys', input='ONESHOT_COLD')
            out['cold_token'] = token_for('ONESHOT_COLD')
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM')
            await _settled(ws, sid, 'RECOVERY_ACK_WARM')
            out['turn_token'] = token_for('WARM')
            out['warm'] = await rpc(ws, 'llm.oneshot', session_id=sid, instructions='sys', input='ONESHOT_WARM')
            out['warm_token'] = token_for('ONESHOT_WARM')
            out['warm_model'] = next(r['model'] for r in reversed(peer.requests)
                                     if any(m.get('content') == 'ONESHOT_WARM' for m in r['messages']))
    with _gateway(tmp_path, named=True) as served:
        asyncio.run(probe(*served))
    assert out['turn_token'] == 'edge-token', out  # the chat itself always sent it
    assert out['cold']['result']['text'] == 'RECOVERY_ACK_ONESHOT_COLD', out
    assert out['warm']['result']['text'] == 'RECOVERY_ACK_ONESHOT_WARM', out
    assert (out['cold_token'], out['warm_token']) == ('edge-token', 'edge-token'), out
    assert out['warm_model'] == 'session-model', out


@pytest.mark.platforms("linux")
def test_a_burst_of_slow_oneshots_never_starves_the_rest_of_the_gateway(tmp_path):
    """More blocked one-shots than the loop's default executor has threads (``min(32, cpus + 4)``):
    another socket's work on that executor (``tools.show``) and its ``ping`` still answer at once."""
    burst, out = 40, {}

    async def probe(home, work, peer, desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='burst', source='gui', toolsets=[], cwd=str(work))
            sid = created['result']['session_id']
            for n in range(burst):
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': f'slow{n}', 'method': 'llm.oneshot',
                                          'params': {'session_id': sid, 'input': 'BLOCK_STARTED'}}))
            # Every one-shot thread is parked on the model: the owner's pool here, the default executor before.
            await async_wait_until(lambda: sum(any(m.get('content') == 'BLOCK_STARTED' for m in r['messages'])
                                               for r in list(peer.requests)) >= 4, timeout=20)
            try:
                async with asyncio.timeout(10), websocket(home, desc) as other:
                    mine = await rpc(other, 'session.create', request_id='other', source='gui', toolsets=[],
                                     cwd=str(work))
                    out['tools'] = await rpc(other, 'tools.show', session_id=mine['result']['session_id'])
                    out['ping'] = await rpc(other, 'ping')
            except TimeoutError:
                out['starved'] = True
            peer.release.set()
            replies = {}
            async with asyncio.timeout(60):
                while len(replies) < burst:
                    reply = json.loads(await ws.recv())
                    if str(reply.get('id', '')).startswith('slow'):
                        replies[reply['id']] = reply
            out['replies'] = replies
    with _gateway(tmp_path) as served:
        asyncio.run(probe(*served))
    assert 'starved' not in out, 'one-shots starved the gateway: tools.show/ping never answered'
    assert 'sections' in out['tools'].get('result', {}), out['tools']
    assert out['ping'].get('result') == {'pong': True}, out['ping']
    answers = [reply.get('result', {}).get('text') or reply.get('error', {}).get('message')
               for reply in out['replies'].values()]
    # Every call is answered: generated, or refused by name once the one-shot pool is full.
    assert set(answers) <= {'RECOVERY_ACK_BLOCK_STARTED', 'oneshot_busy'}, answers
    assert 'RECOVERY_ACK_BLOCK_STARTED' in answers, answers
