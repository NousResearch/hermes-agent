"""Ordinary daemon/API listener and private authenticated WS share execution."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp

from tests.gateway.fixtures.local_recovery_probe import daemon, websocket, rpc
from tests.gateway.fixtures.api_usage_peer import UsageModelPeer


def test_ordinary_daemon_api_ws_fifo_retry(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), UsageModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        api_port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'platforms': {'api_server': {'enabled': True, 'port': api_port, 'host': '127.0.0.1'}}, 'multiplex_profiles': False},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, API_SERVER_KEY='ordinary-daemon-owned-secret',
               API_SERVER_ENABLED='true', API_SERVER_PORT=str(api_port), PYTHONUNBUFFERED='1')

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return [dict(zip(('request_id', 'status', 'payload'), row)) for row in db.execute(
                'SELECT request_id,status,payload_json FROM session_admissions WHERE target_session_id=? ORDER BY seq', ('api-daemon',))]

    async def exercise(desc):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as client:
            url = f'http://127.0.0.1:{api_port}/v1/chat/completions'
            headers = {'Authorization': 'Bearer ordinary-daemon-owned-secret', 'X-Hermes-Session-Id': 'api-daemon'}
            def body(text, stream=False):
                return {'messages': [{'role': 'system', 'content': 'DAEMON_API_SYSTEM'}, {'role': 'user', 'content': text}], 'stream': stream}
            async with client.post(url, json=body('UNAUTHORIZED')) as response:
                assert response.status == 401
            async with client.post(url, json=body('WARM'), headers=headers) as response:
                assert response.status == 200, await response.text()
            async with websocket(home, desc) as ws:
                resumed = await rpc(ws, 'session.resume', session_id='api-daemon')
                assert 'result' in resumed, resumed
                await rpc(ws, 'prompt.submit', session_id='api-daemon', input_id='ws-block', text='BLOCK_FIFO')
                assert await asyncio.to_thread(peer.blocked.wait, 10)
                async def follow(stream=False):
                    async with client.post(url, json=body('FOLLOW', stream), headers={**headers, 'Idempotency-Key': 'daemon-retry'}) as response:
                        text = await response.text()
                        assert response.status == 200, text
                        return text
                pending = asyncio.create_task(follow())
                async with asyncio.timeout(10):
                    while not any(row['request_id'] == 'chat:daemon-retry' and row['status'] == 'queued' for row in rows()):
                        await asyncio.sleep(.02)
                peer.release.set()
                completed = json.loads(await pending)
                assert completed['usage'] == {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}, completed
                before = len(peer.requests)
                streamed = await follow(True)
                assert 'LOCAL_ACK' in streamed and 'data: [DONE]' in streamed, streamed
                assert len(peer.requests) == before
                assert 'BLOCK_FIFO' in json.dumps(peer.requests[-1]['messages'])
                identities = [json.loads(line)['agent'] for line in (home / 'agents.jsonl').read_text().splitlines()]
                assert len(identities) == 3 and len(set(identities)) == 1, identities
                assert len(rows()) == 3 and all(row['status'] == 'terminal' for row in rows()), rows()
                print(json.dumps({'ordinary_daemon': True, 'same_agent': identities, 'admissions': rows()}))
    try:
        with daemon(root, home, env, barrier=True, fixture='api_daemon_witness.py') as (_, desc):
            asyncio.run(exercise(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
