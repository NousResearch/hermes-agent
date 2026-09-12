"""SIGKILL recovery exercises authenticated HTTP on ordinary fresh daemons."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import daemon, Model


def test_api_crash_queue_and_identified_result_retry(tmp_path):
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
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(port), PYTHONUNBUFFERED='1')
    url = f'http://127.0.0.1:{port}/v1'
    headers = {'Authorization': 'Bearer ordinary-daemon-owned-secret'}
    runs = {}
    pids, epochs = [], []

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return dict(db.execute('SELECT request_id,status FROM session_admissions'))

    async def run(client, name, text, session):
        async with client.post(url + '/runs', json={'input': text, 'session_id': session},
                               headers={'Idempotency-Key': name}) as response:
            data = await response.json()
            assert response.status == 202, data
            if name in runs:
                assert runs[name] == data['run_id'] and data['replayed'], data
            runs[name] = data['run_id']
            return data

    async def first():
        async with aiohttp.ClientSession(headers=headers) as client:
            # Warm before the exact model barrier (onboarding may decorate first input).
            async with client.post(url + '/chat/completions', json={'messages': [{'role': 'user', 'content': 'WARM'}]},
                                   headers={'X-Hermes-Session-Id': 'unknown'}) as response:
                assert response.status == 200, await response.text()
            async def block():
                async with client.post(url + '/chat/completions',
                    json={'messages': [{'role': 'user', 'content': 'BLOCK_STARTED'}]},
                    headers={'X-Hermes-Session-Id': 'unknown', 'Idempotency-Key': 'started'}) as response:
                    await response.read()
            blocked = asyncio.create_task(block())
            assert await asyncio.to_thread(peer.blocked.wait, 15)
            runs['started'] = 'chat:started'
            blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
            await run(client, 'follower', 'NEVER_REPLAY', 'unknown')
            await run(client, 'safe', 'SAFE_QUEUE', 'safe')
            assert rows()[runs['safe']] == 'queued'
            assert rows()[runs['follower']] == 'queued'

    async def restarted():
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as client:
            async with asyncio.timeout(20):
                while rows()[runs['safe']] != 'terminal':
                    await asyncio.sleep(.05)
            assert rows()[runs['started']] == 'unknown' and rows()[runs['follower']] == 'queued', rows()
            async with client.post(url + '/chat/completions',
                json={'messages': [{'role': 'user', 'content': 'BLOCK_STARTED'}]},
                headers={'X-Hermes-Session-Id': 'unknown', 'Idempotency-Key': 'started'}) as response:
                unknown = await response.json()
                assert response.status == 409 and unknown['error']['code'] == 'unknown_execution', unknown
            before = len(peer.requests)
            await run(client, 'safe', 'SAFE_QUEUE', 'safe')
            async with client.get(url + '/runs/' + runs['safe']) as response:
                result = await response.json()
                assert result['status'] == 'completed' and 'SAFE_QUEUE' in result['output'], result
            assert len(peer.requests) == before
            # A new identified chat submission executes once; retry after another SIGKILL
            # must use retained results, independent of delivery format.
            body = {'messages': [{'role': 'user', 'content': 'RESULT_ONCE'}]}
            hs = {'X-Hermes-Session-Id': 'result', 'Idempotency-Key': 'result-once'}
            async with client.post(url + '/chat/completions', json=body, headers=hs) as response:
                reply = await response.json()
                assert response.status == 200 and 'RESULT_ONCE' in reply['choices'][0]['message']['content'], reply
            assert rows()['chat:result-once'] == 'terminal'

    try:
        with daemon(root, home, env, barrier=True, fixture='api_recovery_daemon.py') as (proc, desc):
            pids.append(proc.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(first())
            proc.kill(); proc.wait(timeout=10)
        peer.release.set()
        for _ in range(2):
            with daemon(root, home, env, barrier=False) as (proc, desc):
                pids.append(proc.pid); epochs.append(desc['authority_epoch'])
                asyncio.run(restarted())
                proc.kill(); proc.wait(timeout=10)
        texts = [next((m.get('content', '') for m in reversed(req['messages']) if m['role'] == 'user'), '') for req in peer.requests]
        assert sum('SAFE_QUEUE' in str(t) for t in texts) == 1, texts
        assert sum('RESULT_ONCE' in str(t) for t in texts) == 1, texts
        assert texts.count('BLOCK_STARTED') == 1 and not any('NEVER_REPLAY' in str(t) for t in texts), texts
        assert epochs == sorted(set(epochs)) and len(set(pids)) == 3
        print(json.dumps({'pids': pids, 'epochs': epochs, 'rows': rows(), 'inference_texts': texts}))
    finally:
        peer.release.set(); peer.shutdown(); peer.server_close()
