"""A committed result survives owner SIGKILL without pausing its follower."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import pytest
from tests.gateway.fixtures.local_recovery_probe import daemon, Model


@pytest.mark.linux_only
def test_terminal_commit_before_publication_survives_kill(tmp_path):
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
    body = {'messages': [{'role': 'user', 'content': 'BLOCK_STARTED'}]}
    identity = {'X-Hermes-Session-Id': 'used', 'Idempotency-Key': 'once'}
    pids = []

    def snapshot():
        with sqlite3.connect(home / 'state.db') as db:
            rows = dict(db.execute('SELECT request_id,status FROM session_admissions'))
            result = db.execute("SELECT value FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%' AND value LIKE '%RECOVERY_ACK_BLOCK_STARTED%'").fetchone()
            return rows, json.loads(result[0]) if result else None

    async def first(proc):
        async with aiohttp.ClientSession(headers=headers) as client:
            async with client.post(url + '/chat/completions', json={'messages': [{'role': 'user', 'content': 'WARM'}]},
                                   headers={'X-Hermes-Session-Id': 'used'}) as response:
                assert response.status == 200, await response.text()
            async def request():
                async with client.post(url + '/chat/completions', json=body, headers=identity) as response:
                    return await response.json()
            pending = asyncio.create_task(request())
            try:
                assert await asyncio.to_thread(peer.blocked.wait, 20)
                async with client.post(url + '/runs', json={'input': 'FOLLOW_ON', 'session_id': 'used'},
                                       headers={'Idempotency-Key': 'follow'}) as response:
                    follower = await response.json()
                    assert response.status == 202, follower
                peer.release.set()
                async with asyncio.timeout(30):
                    while snapshot()[1] is None:
                        await asyncio.sleep(.02)
                rows, saved = snapshot()
                assert not pending.done(), 'HTTP result published before crash barrier'
                proc.kill(); await asyncio.to_thread(proc.wait, 10)
                assert rows[follower['run_id']] == 'queued'
                return saved, follower['run_id']
            finally:
                pending.cancel(); await asyncio.gather(pending, return_exceptions=True)

    async def restart(saved, follower):
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as client:
            async with client.post(url + '/chat/completions', json=body, headers=identity) as response:
                reply = await response.json()
                assert response.status == 200, reply
                assert reply['choices'][0]['message']['content'] == saved['result']['final_response']
            async with asyncio.timeout(25):
                while snapshot()[0][follower] != 'terminal':
                    await asyncio.sleep(.03)
            assert snapshot()[0]['chat:once'] == 'terminal'
            assert snapshot()[1] == saved

    try:
        with daemon(root, home, env, barrier=True, fixture='terminal_commit_daemon.py') as (proc, desc):
            pids.append(proc.pid)
            saved, follower = asyncio.run(first(proc))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            pids.append(proc.pid)
            asyncio.run(restart(saved, follower))
        texts = [next(m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user') for r in peer.requests]
        assert sum('BLOCK_STARTED' in str(t) for t in texts) == 1, texts
        assert sum('FOLLOW_ON' in str(t) for t in texts) == 1, texts
        print(json.dumps({'pids': pids, 'inference_texts': texts, 'exact_result': saved, 'rows': snapshot()[0]}))
    finally:
        peer.release.set(); peer.shutdown(); peer.server_close()
