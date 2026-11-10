"""Real source stays alive while an explicit recipient comes online after settlement.

Only the model is a loopback wire fixture; two owned real native gateways, actual
signed HTTP, distinct stores, housekeeping, admission and settled native receipt.
"""
import asyncio
import hashlib
import hmac
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sqlite3
import threading

import aiohttp
import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.gateway.test_webhook_bot_chat_integration import WebhookModel, rows
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('origin', ['agent', 'direct', 'hook'])
def test_offline_recipient_resumes_on_active_gateway_housekeeping(tmp_path, monkeypatch, record_property, origin):
    _exercise_active_reconciliation(tmp_path, monkeypatch, record_property, origin, dynamic=False)


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('origin', ['direct', 'hook'])
def test_dynamic_settled_duplicate_survives_subscription_reload(tmp_path, monkeypatch, record_property, origin):
    _exercise_active_reconciliation(tmp_path, monkeypatch, record_property, origin, dynamic=True)


def _exercise_active_reconciliation(tmp_path, monkeypatch, record_property, origin, *, dynamic):
    from gateway.platforms.webhook_bot_chat import result_id
    from tools.bot_live_delivery import read_delivery_result
    home, user, runtime = [tmp_path / name for name in ('state', 'user', 'runtime')]
    target = home / 'profiles' / 'receiver'
    for directory in (home, user, runtime, target):
        directory.mkdir(mode=0o700, parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(user))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('XDG_RUNTIME_DIR', '/proc/self/cwd/runtime')
    model = ThreadingHTTPServer(('127.0.0.1', 0), WebhookModel)
    model_requests = []
    model.requests = model_requests
    model.blocked, model.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=model.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = dict(model={'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
        gateway={'standalone': True}, auxiliary={'title_generation': {'enabled': False}},
        streaming={'enabled': False}, platform_toolsets={'gui': []})
    (target / 'config.yaml').write_text(json.dumps(cfg))
    cfg['platforms'] = {'webhook': {'enabled': True, 'extra': {'host': '127.0.0.1', 'port': port,
        'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', 'deliver': 'bot-chat:receiver'}}}}}
    if origin == 'direct':
        cfg['platforms']['webhook']['extra']['routes']['fixture']['deliver_only'] = True
    if dynamic:
        from hermes_cli import webhook as subscriptions
        route = cfg['platforms']['webhook']['extra']['routes'].pop('fixture')
        subscriptions._replace_subscription('fixture', route, subscriptions._MISSING_SUBSCRIPTION)
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env() | dict(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
        PYTHONPATH=str(Path(__file__).resolve().parents[2]), OPENAI_API_KEY='loopback-only',
        OPENAI_BASE_URL=base, XDG_RUNTIME_DIR='/proc/self/cwd/runtime', PYTHONUNBUFFERED='1')

    def source_intent(origin_id):
        prefix = 'webhook.final.intent.v1.' if origin == 'agent' else 'webhook.origin.v1.'
        with sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True) as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (prefix + origin_id,)).fetchone()
        return json.loads(row[0]) if row else None

    async def send():
        body = json.dumps({'text': 'HOOK_INPUT' if origin == 'hook' else 'SOURCE_PRIVATE_PROMPT'}).encode()
        signature = 'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()
        async with aiohttp.ClientSession() as client:
            async with client.post(f'http://127.0.0.1:{port}/webhooks/fixture', data=body,
                headers={'X-GitHub-Delivery': 'offline-final', 'X-Hub-Signature-256': signature}) as response:
                assert response.status == 202, await response.text()
                accepted = await response.json()
        if origin == 'agent':
            await async_wait_until(lambda: any(r['request_id'] == 'offline-final' and r['status'] == 'terminal'
                for r in rows(home)), timeout=40, what='real source result')
            source = next(r for r in rows(home) if r['request_id'] == 'offline-final')
            origin_id = source['admission_id']
            key = result_id(home, origin_id)
        else:
            origin_id, key = accepted['origin_id'], accepted['delivery_id']
            assert accepted['status'] == 'pending'
        await async_wait_until(lambda: source_intent(origin_id) is not None,
                               timeout=20, what='offline intent persistence')
        saved = source_intent(origin_id)
        assert saved and saved['state'] == 'pending' and saved['target_pin']['home'] == str(target)
        assert read_delivery_result(target, key) is None
        return {'origin_id': origin_id, 'delivery_id': key}

    async def observed(source):
        key = source['delivery_id']
        await async_wait_until(lambda: (read_delivery_result(target, key) or {}).get('status') == 'settled',
                               timeout=100, what='active-housekeeping native settlement')
        receipt = read_delivery_result(target, key)
        assert receipt['reply'] == 'RECEIVER_ACK'
        assert len(rows(home)) == (1 if origin == 'agent' else 0) and len(rows(target)) == 1
        assert rows(target)[0]['admission_id'] == receipt['admission_id']
        inputs = [next((m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user'), '')
                  for r in model.requests if r.get('messages')]
        assert inputs.count('SOURCE_PRIVATE_PROMPT') == (1 if origin == 'agent' else 0)
        assert sum('Webhook automation' in str(value) for value in inputs) == 1
        record_property('native_receipt', json.dumps({'origin': origin, 'origin_id': source['origin_id'], 'receipt': receipt}))

    async def replay_settled(source):
        body = json.dumps({'text': 'HOOK_INPUT' if origin == 'hook' else 'SOURCE_PRIVATE_PROMPT'}).encode()
        signature = 'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()
        async with aiohttp.ClientSession() as client:
            async with client.post(f'http://127.0.0.1:{port}/webhooks/fixture', data=body,
                headers={'X-GitHub-Delivery': 'offline-final', 'X-Hub-Signature-256': signature}) as response:
                assert response.status == 202, await response.text()
                replay = await response.json()
        assert replay['status'] == 'settled'
        assert replay['origin_id'] == source['origin_id'] and replay['delivery_id'] == source['delivery_id']
        return replay

    try:
        fixture = str(Path(__file__).resolve().parent / 'fixtures/webhook_origins_daemon.py')
        with daemon(tmp_path, home, env, barrier=origin == 'hook', fixture=fixture) as (source_proc, _):
            source = asyncio.run(send())
            with daemon(tmp_path, target, env | {'HERMES_HOME': str(target)}, barrier=False):
                asyncio.run(observed(source))
                assert source_proc.poll() is None, 'source owner must not restart to retry delivery'
            if origin != 'agent':
                settled = asyncio.run(replay_settled(source))
                saved = source_intent(source['origin_id'])
                assert saved is not None
                request_count = len(model_requests)
                target.rename(target.with_name('removed-receiver'))
                if dynamic:
                    from utils import file_cache_version
                    path = home / 'webhook_subscriptions.json'
                    previous = file_cache_version(path)
                    subscriptions._replace_subscription('unrelated', {'secret': 'unrelated-secret', 'deliver': 'log'},
                                                        subscriptions._MISSING_SUBSCRIPTION)
                    assert subscriptions._load_subscriptions()['fixture'] == route
                    assert file_cache_version(path) != previous
                assert asyncio.run(replay_settled(source)) == settled
                assert source_intent(source['origin_id']) == saved
                assert source_proc.poll() is None and len(model_requests) == request_count
                assert not rows(home), 'direct/hook replay must not invent a producer admission'
                if origin == 'hook':
                    assert (home / 'first.log').read_text().count('HOOK_CONSUMED offline-final') == 1
                record_property('removed_target_duplicate', json.dumps({'origin': origin, 'dynamic_reload': dynamic,
                    'source_pid': source_proc.pid,
                    'response': settled, 'native_admission_id': saved['native_admission_id'],
                    'target_pin': saved['target_pin'], 'model_requests': request_count}))
    finally:
        model.release.set()
        model.shutdown()
        model.server_close()
        thread.join(timeout=5)
