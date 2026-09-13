"""Real daemon delete/restart: history disappears, exact authorized retries survive."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import httpx
import pytest
from tests.gateway.fixtures.local_recovery_probe import daemon, Model, websocket, rpc
from tests.gateway.test_native_http_auth import ticket, headers as native_headers


@pytest.mark.linux_only
def test_used_api_and_native_delete_restart_exact_retry(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700); user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'platforms': {'api_server': {'enabled': True, 'port': port, 'host': '127.0.0.1'}}, 'multiplex_profiles': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(port), PYTHONUNBUFFERED='1',
               HERMES_DASHBOARD_SESSION_TOKEN='foreign-dashboard-owner')
    url = f'http://127.0.0.1:{port}/v1'
    headers = {'Authorization': 'Bearer ordinary-daemon-owned-secret'}
    body = {'messages': [{'role': 'user', 'content': 'API_USED_ONCE'}]}
    identity = {'X-Hermes-Session-Id': 'used-api', 'Idempotency-Key': 'once'}
    receipts = {}

    async def api_request():
        async with aiohttp.ClientSession(headers=headers) as client:
            async with client.post(url + '/chat/completions', json=body, headers=identity) as response:
                result = await response.json()
                assert response.status == 200, result
                return result['choices'][0]['message']['content'], result['usage']

    async def native_turn(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='native-used', source='gui', cwd=str(home), toolsets=[])
            sid = created['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='native-once', text='NATIVE_USED_ONCE')
            assert submitted['result']['status'] == 'queued', submitted
            async with asyncio.timeout(30):
                while True:
                    with sqlite3.connect(home / 'state.db') as db:
                        row = db.execute("SELECT status FROM session_admissions WHERE request_id='native-once'").fetchone()
                    if row and row[0] == 'terminal':
                        break
                    await asyncio.sleep(.03)
            # Wait for the drain's publication, not only the earlier DB terminal commit.
            async with asyncio.timeout(20):
                while True:
                    event = json.loads(await ws.recv())
                    if event.get('params', {}).get('type') == 'message.complete':
                        break
            return sid

    def delete(desc, sid):
        path = '/api/sessions/' + sid
        with httpx.Client(base_url=desc['api_origin'], trust_env=False) as client:
            snapshot = client.get(path + '/mutation-snapshot', headers=native_headers(ticket(home, desc)))
            assert snapshot.status_code == 200, snapshot.text
            snap = snapshot.json()
            params = dict(request_id='delete-' + sid, expected_revision=snap['runtime_revision'],
                          expected_generation=snap['runtime_generation'])
            result = client.delete(path, params=params, headers=native_headers(ticket(home, desc)))
            assert result.status_code == 200, result.text
            assert client.get(path, headers=native_headers(ticket(home, desc))).status_code == 404
            receipts[sid] = (params, result.json())

    async def stale_native(desc, sid):
        async with websocket(home, desc) as ws:
            denied = await rpc(ws, 'prompt.submit', session_id=sid, input_id='stale-new', text='MUST_NOT_RUN')
            assert 'error' in denied, denied

    async def busy_delete_refused(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='native-busy', source='gui', cwd=str(home), toolsets=[])
            sid = created['result']['session_id']
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='busy-once', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            with httpx.Client(base_url=desc['api_origin'], trust_env=False) as client:
                snap = client.get(f'/api/sessions/{sid}/mutation-snapshot', headers=native_headers(ticket(home, desc))).json()
                refused = client.delete('/api/sessions/' + sid, headers=native_headers(ticket(home, desc)),
                                        params=dict(request_id='delete-busy', expected_revision=snap['runtime_revision'],
                                                    expected_generation=snap['runtime_generation']))
            assert refused.status_code == 409 and 'session_busy' in refused.text, refused.text
            peer.release.set()
            with sqlite3.connect(home / 'state.db') as db:
                assert db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone()
                assert db.execute("SELECT 1 FROM session_admissions WHERE request_id='busy-once'").fetchone()

    try:
        with daemon(root, home, env, barrier=True) as (proc, desc):
            first = asyncio.run(api_request())
            native_sid = asyncio.run(native_turn(desc))
            asyncio.run(busy_delete_refused(desc))
            delete(desc, 'used-api'); delete(desc, native_sid)
            before = len(peer.requests)
        with daemon(root, home, env, barrier=False) as (proc, desc):
            with httpx.Client(base_url=desc['api_origin'], trust_env=False) as client:
                for sid, (params, receipt) in receipts.items():
                    path = '/api/sessions/' + sid
                    retry = client.delete(path, params=params, headers=native_headers(ticket(home, desc)))
                    assert retry.status_code == 200 and retry.json() == receipt, retry.text
                    changed = client.delete(path, params=params | {'expected_revision': 999}, headers=native_headers(ticket(home, desc)))
                    assert changed.status_code == 409 and 'admission_conflict' in changed.text
                    foreign = client.delete(path, params=params, headers={'Authorization': 'Bearer foreign-dashboard-owner'})
                    assert foreign.status_code != 200, foreign.text
            assert asyncio.run(api_request()) == first
            asyncio.run(stale_native(desc, native_sid))
            with sqlite3.connect(home / 'state.db') as db:
                for sid in receipts:
                    assert db.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone() is None
                    assert db.execute('SELECT 1 FROM messages WHERE session_id=?', (sid,)).fetchone() is None
                    assert db.execute('SELECT 1 FROM session_admissions WHERE target_session_id=?', (sid,)).fetchone() is None
                assert not db.execute('PRAGMA foreign_key_check').fetchall()
            assert len(peer.requests) == before
        print(json.dumps({'dispatch_hook_composed': True, 'restart_ordinary': True,
                          'deleted_ids': list(receipts), 'exact_api_reply': first, 'inference_count': before}))
    finally:
        peer.release.set(); peer.shutdown(); peer.server_close()
