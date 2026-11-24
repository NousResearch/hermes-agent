"""A typed ``/model`` admitted as a prompt on a local session commits the session's policy (#138814)."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket
from tests.gateway.test_session_mutation_model import CatalogModel
from tests.waiting import async_wait_until


@pytest.mark.platforms("linux")
def test_typed_model_command_on_a_local_session_moves_the_next_request(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), CatalogModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 100000, 'output': 4000}}
        for name in ('original', 'switched')}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']

            async def turn(sid, text):
                await call('prompt.submit', session_id=sid, input_id=text, text=text)

                def terminal():
                    with sqlite3.connect(home / 'state.db') as db:
                        row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                    return row and row[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what=f'{text} terminal')

            sid = (await call('session.create', source='tui', request_id='owner', toolsets=[]))['session_id']
            await turn(sid, 'BEFORE')
            await turn(sid, '/model switched --reasoning high')
            await turn(sid, 'AFTER')
            assert [r['model'] for r in peer.requests] == ['original', 'switched'], [(r['model'], r['messages'][-1]['content']) for r in peer.requests]
            # The level typed with the model commits in the same selection.
            assert peer.requests[-1].get('reasoning_effort') == 'high', peer.requests[-1]
            assert (await call('session.info', session_id=sid))['model'] == 'switched'
            with sqlite3.connect(home / 'state.db') as db:
                assert db.execute('SELECT model FROM sessions WHERE id=?', (sid,)).fetchone()[0] == 'switched'
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(run(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


class _AliasEndpoint(Model):
    def do_GET(self):
        payload = json.dumps({'data': [{'id': name} for name in ('original', 'shared')]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self.server.auth.append(self.headers.get('Authorization'))
        super().do_POST()


@pytest.mark.platforms("linux")
def test_typed_alias_on_a_local_session_keeps_its_own_endpoint_and_key(tmp_path):
    """Two aliases name one model on ``custom`` at different endpoints: the typed (second) alias's
    endpoint and key serve the next request; the first alias's endpoint receives nothing."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peers = []
    for _ in range(3):
        peer = ThreadingHTTPServer(('127.0.0.1', 0), _AliasEndpoint)
        peer.requests, peer.auth = [], []
        peer.blocked, peer.release = threading.Event(), threading.Event()
        threading.Thread(target=peer.serve_forever, daemon=True).start()
        peers.append(peer)
    launch, first, second = (f'http://127.0.0.1:{p.server_port}/v1' for p in peers)
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': launch},
           'model_aliases': {
               'first-alias': {'model': 'shared', 'provider': 'custom', 'base_url': first, 'api_key': 'key-one'},
               'second-alias': {'model': 'shared', 'provider': 'custom', 'base_url': second, 'api_key': 'key-two'}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 100000, 'output': 4000}}
        for name in ('original', 'shared')}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=launch, PYTHONUNBUFFERED='1')

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def turn(sid, text):
                assert 'result' in await rpc(ws, 'prompt.submit', session_id=sid, input_id=text, text=text)

                def terminal():
                    with sqlite3.connect(home / 'state.db') as db:
                        row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                    return row and row[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what=f'{text} terminal')

            sid = (await rpc(ws, 'session.create', source='tui', request_id='owner', toolsets=[]))['result']['session_id']
            await turn(sid, '/model second-alias')
            await turn(sid, 'AFTER')
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(run(desc))
        assert [r['model'] for r in peers[2].requests] == ['shared'], [p.requests for p in peers]
        assert set(peers[2].auth) == {'Bearer key-two'}, peers[2].auth
        assert peers[1].auth == [], peers[1].auth
        assert peers[1].requests == [], peers[1].requests
    finally:
        for peer in peers:
            peer.release.set()
            peer.shutdown()
            peer.server_close()
