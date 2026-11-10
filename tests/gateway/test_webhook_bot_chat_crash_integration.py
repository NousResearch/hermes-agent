"""Crash after the real webhook source commit, then recover its real Bot Chat handoff."""
import asyncio
import hashlib
import hmac
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_webhook_bot_chat_integration import WebhookModel, rows
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
def test_committed_source_handoff_recovers_after_owned_daemon_crash(tmp_path, monkeypatch):
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
    peer = ThreadingHTTPServer(('127.0.0.1', 0), WebhookModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'standalone': True},
           'platforms': {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
               'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'bot-chat'}}}}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'terminal': {'cwd': str(home)},
           'streaming': {'enabled': False}, 'platform_toolsets': {'gui': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    marker = tmp_path / 'barrier'
    env = child_env() | {'HOME': str(user), 'USERPROFILE': str(user), 'HERMES_HOME': str(home),
        'PYTHONPATH': str(root), 'OPENAI_API_KEY': 'loopback-only', 'OPENAI_BASE_URL': base,
        'PYTHONUNBUFFERED': '1', 'XDG_RUNTIME_DIR': '/proc/self/cwd/runtime', 'TEST_HANDOFF_BARRIER': str(marker)}

    async def first(proc, desc):
        async with websocket(home, desc) as ws, aiohttp.ClientSession() as client:
            created = await rpc(ws, 'session.create', request_id='target', source='gui', cwd=str(home), toolsets=[])
            sid = created['result']['session_id']
            renamed = await rpc(ws, 'session.mutate', session_id=sid, request_id='title',
                expected_revision=created['result']['revision'], operation='rename', payload={'title': 'Bot Chat'})
            assert 'result' in renamed, renamed
            body = json.dumps({'text': 'SOURCE_PRIVATE_PROMPT'}).encode()
            signature = 'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()
            async with client.post(f'http://127.0.0.1:{port}/webhooks/fixture', data=body,
                headers={'X-GitHub-Delivery': 'crash-event', 'X-GitHub-Event': 'push', 'X-Hub-Signature-256': signature}) as response:
                assert response.status == 202
            await async_wait_until(marker.exists, timeout=40, what='pre-handoff barrier')
            source = next(r for r in rows(home) if r['request_id'] == 'crash-event')
            assert source['status'] == 'terminal' and source['outcome'] == 'completed'
            key = result_id(home, source['admission_id'])
            assert read_delivery_result(home, key) is None
            assert not any(r['request_id'] == 'bot:' + key for r in rows(home))
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
            return sid, source, key

    async def second(sid, source, key):
        await async_wait_until(lambda: (read_delivery_result(home, key) or {}).get('status') == 'settled',
                               timeout=40, what='real recovered receipt')
        receipt = read_delivery_result(home, key)
        target = next(r for r in rows(home) if r['request_id'] == 'bot:' + key)
        assert receipt['session_id'] == sid and receipt['reply'] == 'RECEIVER_ACK'
        assert target['admission_id'] == receipt['admission_id']
        assert len([r for r in rows(home) if r['request_id'] == 'crash-event']) == 1
        with sqlite3.connect(home / 'state.db') as db:
            ended = db.execute('SELECT ended_at, end_reason FROM sessions WHERE id=?',
                               (source['target_session_id'],)).fetchone()
        assert ended[0] is not None and ended[1] == 'webhook_complete'
        inputs = [next((m.get('content', '') for m in reversed(req['messages']) if m['role'] == 'user'), '')
                  for req in peer.requests if req.get('messages')]
        assert inputs.count('SOURCE_PRIVATE_PROMPT') == 1
        assert len([x for x in inputs if isinstance(x, str) and 'Webhook automation' in x]) == 1
        print(json.dumps({'source': source['admission_id'], 'delivery_id': key, 'target': target['admission_id'], 'receipt': receipt}))

    try:
        fixture = str(root / 'tests/gateway/fixtures/webhook_handoff_barrier.py')
        with daemon(tmp_path, home, env, barrier=True, fixture=fixture) as (proc, desc):
            sid, source, key = asyncio.run(first(proc, desc))
        with daemon(tmp_path, home, env, barrier=False) as (_, desc):
            asyncio.run(second(sid, source, key))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
