"""Ordinary daemon, real WebSocket owner, and a loopback provider: speech reaches the next turn."""
import asyncio
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest
from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until


@contextmanager
def speech_daemon(tmp_path, *, review):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False, 'managed_workers': False},
              'model': {'provider': 'custom', 'default': 'voice-context', 'base_url': url},
              'memory': {'nudge_interval': 2 if review else 0},
              'auxiliary': {'title_generation': {'enabled': False},
                            'background_review': {'enabled': review, 'defer': 'never'}}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    try:
        yield root, home, env, peer
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('review', [False, True])
def test_recorded_speech_reconnect_next_turn_and_native_memory_review(tmp_path, review):
    with speech_daemon(tmp_path, review=review) as (root, home, env, peer):
        async def probe(desc):
            async with websocket(home, desc) as ws:
                async def call(method, **params):
                    reply = await rpc(ws, method, **params)
                    assert 'result' in reply, reply
                    return reply['result']
                created = await call('session.create', source='gui', request_id='voice',
                                     cwd=str(home), toolsets=['memory'] if review else [])
                sid = created['session_id']
                params = dict(session_id=sid, role='user', content='VOICE_FIRST', client_id='one')
                first = await call('session.append_message', **params)
                assert peer.requests == []
                second = await call('session.append_message', session_id=sid, role='user',
                                    content='VOICE_CORRECTION', client_id='two')
                assert first['id'] != second['id']
                assert await call('session.append_message', **params) == first
                resumed = await call('session.resume', session_id=sid)
                assert [m['content'] for m in resumed['messages']] == ['VOICE_FIRST', 'VOICE_CORRECTION']
                if review:
                    await async_wait_until(lambda: bool(peer.requests), timeout=45, what='automatic memory review model call')
                    text = json.dumps(peer.requests[0]['messages'])
                    assert 'VOICE_FIRST' in text and 'VOICE_CORRECTION' in text
                    with sqlite3.connect(home / 'state.db') as db:
                        assert db.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
                    return params, first
                assert peer.requests == []
                await call('session.append_message', session_id=sid, role='assistant',
                           content='VOICE_REPLY', client_id='reply')
                await call('prompt.submit', session_id=sid, input_id='active', text='BLOCK_STARTED')
                assert await asyncio.to_thread(peer.blocked.wait, 20)
                await call('session.append_message', session_id=sid, role='user',
                           content='VOICE_WHILE_WORKING', client_id='during-work')
                peer.release.set()

                def terminal():
                    with sqlite3.connect(home / 'state.db') as db:
                        row = db.execute("SELECT status FROM session_admissions WHERE request_id='active'").fetchone()
                    return row and row[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what='existing worker completed after speech append')
                peer.requests.clear()
                return params, first
        with daemon(root, home, env, barrier=False) as (_, desc):
            params, first = asyncio.run(probe(desc))
        if review:
            return

        async def reconnect(desc):
            async with websocket(home, desc) as ws:
                assert (await rpc(ws, 'session.append_message', **params))['result'] == first
                resumed = await rpc(ws, 'session.resume', session_id=params['session_id'])
                assert 'result' in resumed, resumed
                submitted = await rpc(ws, 'prompt.submit', session_id=params['session_id'],
                                      input_id='next', text='USE_RECORDED_SPEECH')
                assert 'result' in submitted, submitted
                await async_wait_until(lambda: bool(peer.requests), timeout=45, what='next native turn reaching provider')
                text = json.dumps(peer.requests[-1]['messages'])
                assert all(word in text for word in ('VOICE_FIRST', 'VOICE_CORRECTION', 'VOICE_WHILE_WORKING',
                                                     'USE_RECORDED_SPEECH'))
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(reconnect(desc))
