"""A guarded deferred model pick stays answerable across a real owner restart (#139604 review)."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, websocket
from tests.gateway.test_session_mutation_model import CatalogModel
from tests.waiting import async_wait_until


class _MeasuredModel(CatalogModel):
    """Streams report provider usage, so the owner measures the session's context."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        text = next((m.get('content', '') for m in reversed(body.get('messages', [])) if m['role'] == 'user'), '')
        if text == 'BLOCK_STARTED':
            self.server.blocked.set()
            self.server.release.wait(60)
        message = {'role': 'assistant', 'content': 'ACK ' + str(text)}
        usage = {'prompt_tokens': 250000, 'completion_tokens': 10, 'total_tokens': 250010}
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'probe', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'stop'}],
                                              'usage': usage}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps({'id': 'probe', 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                                  'usage': usage}).encode()
            kind = 'application/json'
        try:
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Client:
    def __init__(self, ws):
        self.ws, self.events, self.count = ws, [], 0

    async def call(self, method, **params):
        self.count += 1
        await self.ws.send(json.dumps({'jsonrpc': '2.0', 'id': self.count, 'method': method, 'params': params}))
        async with asyncio.timeout(30):
            while True:
                message = json.loads(await self.ws.recv())
                if message.get('id') == self.count:
                    assert 'result' in message, message
                    return message['result']
                self.events.append(message)

    async def pump(self, until, what):
        async with asyncio.timeout(20):
            while not until():
                self.events.append(json.loads(await self.ws.recv()))

    def asks(self):
        return [e['params']['payload'] for e in self.events if e.get('method') == 'event'
                and e['params'].get('type') == 'session.updated'
                and e['params']['payload'].get('status') == 'confirmation_required']


@pytest.mark.platforms("linux")
def test_a_guarded_deferred_pick_is_asked_again_after_an_owner_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), _MeasuredModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    # Any measured context trips the large-context switch guard: the pick needs the user's consent.
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url, 'switch_context_confirm_tokens': 1},
           'compression': {'enabled': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'switched')}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def terminal(name):
        with sqlite3.connect(home / 'state.db') as db:
            row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (name,)).fetchone()
        return row and row[0] == 'terminal'

    def pending(sid):
        with sqlite3.connect(home / 'state.db') as db:
            row = db.execute('SELECT value FROM state_meta WHERE key=?', ('gateway.deferred_pick.v1.' + sid,)).fetchone()
        return json.loads(row[0]) if row else None

    async def turn(c, sid, name):
        await c.call('prompt.submit', session_id=sid, input_id=name, text=name)
        await async_wait_until(lambda: terminal(name), timeout=30, what=name)

    async def first(desc):
        async with websocket(home, desc) as ws:
            c = _Client(ws)
            sid = (await c.call('session.create', source='tui', request_id='guarded', toolsets=[]))['session_id']
            await turn(c, sid, 'WARM')
            await c.call('prompt.submit', session_id=sid, input_id='block', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            snap = await c.call('session.resume', session_id=sid)
            busy = await c.call('session.mutate', session_id=sid, request_id='guarded-pick', operation='model',
                                payload={'model': 'switched'}, expected_revision=snap['revision'],
                                expected_generation=snap['execution_generation'])
            assert busy['status'] == 'deferred', busy
            await c.call('prompt.submit', session_id=sid, input_id='follower', text='FOLLOWER')
            peer.release.set()
            await async_wait_until(lambda: terminal('follower'), timeout=30, what='queued follower')
            assert (pending(sid) or {}).get('awaiting_confirm'), (pending(sid), [r['model'] for r in peer.requests])
            return sid

    async def second(desc, sid):
        async with websocket(home, desc) as ws:
            c = _Client(ws)
            snap = await c.call('session.resume', session_id=sid)
            # The restarted owner asks the newly attached viewer again, with the guard re-checked.
            await c.pump(lambda: c.asks(), 'the pending pick asked again after the restart')
            ask = c.asks()[-1]
            assert ask['target_model'] == 'switched' and ask['payload'] == {'model': 'switched'}, ask
            snap = await c.call('session.resume', session_id=sid)
            confirmed = await c.call('session.mutate', session_id=sid, request_id='confirmed', operation='model',
                                     payload={'model': 'switched', 'confirm': ask['confirm']},
                                     expected_revision=snap['revision'],
                                     expected_generation=snap['execution_generation'])
            assert confirmed.get('model') == 'switched', confirmed
            await turn(c, sid, 'AFTER_RESTART')
            assert pending(sid) is None

    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid = asyncio.run(first(desc))
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(second(desc, sid))
        assert peer.requests[-1]['model'] == 'switched', [r['model'] for r in peer.requests]
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
