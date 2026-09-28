import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from gateway.platforms.webhook_responsibilities import ResponsibilityIngress
from responsibilities import webhook_store


@pytest.mark.asyncio
async def test_ingress_signature_dedup_handshake_and_retirement(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('PROVIDER_WEBHOOK_SECRET','test-secret')
    package = tmp_path/'responsibilities'/'inbox'
    (package/'webhooks').mkdir(parents=True)
    (package/'RESPONSIBILITY.md').write_text('---\nname: inbox\ntrigger: Handle inbox events\n---\nRead events.\n')
    (package/'STATE.md').write_text('')
    (package/'references').mkdir()
    declaration = package/'webhooks/events.yaml'
    declaration.write_text('scope: Handle event\nreport: muted\nkey: customer\nhandshake: challenge\nverify:\n  signature: hmac_sha256(body)\n  header: X-Signature\n  secret: env:PROVIDER_WEBHOOK_SECRET\n')
    scan = webhook_store.reconcile()
    assert not scan.package_errors
    assert not scan.entries[0].webhook_errors
    with webhook_store.database() as db:
        token = db.execute('SELECT token FROM routes').fetchone()[0]
    ingress = ResponsibilityIngress(SimpleNamespace(_max_body_bytes=1000000))
    app = web.Application();app.router.add_route('*','/responsibilities/{token}',ingress.handle)
    async with TestClient(TestServer(app)) as client:
        path = '/responsibilities/'+token
        response = await client.get(path+'?challenge=hello')
        assert response.status == 200 and await response.text() == 'hello'
        raw = json.dumps({'customer':'one','event':'changed'}).encode()
        assert (await client.post(path,data=raw)).status == 403
        signature = hmac.new(b'test-secret',raw,hashlib.sha256).hexdigest()
        headers = {'X-Signature':signature,'X-GitHub-Delivery':'delivery-1'}
        assert (await client.post(path,data=raw,headers=headers)).status == 202
        assert (await client.post(path,data=raw,headers=headers)).status == 202
        ids, deliveries = webhook_store.claim(token,'"one"')
        assert len(ids) == 1 and deliveries[0]['payload']['event'] == 'changed'
        webhook_store.complete(ids)
        for payload, expected in (({'type': 'event_callback', 'event_id': 'Ev123'}, 1),
                                  ({'object': 'event', 'id': 'evt_123'}, 1),
                                  ({'id': 'resource-123'}, 2)):
            raw = json.dumps(payload).encode()
            signature = hmac.new(b'test-secret', raw, hashlib.sha256).hexdigest()
            for _ in range(2):
                assert (await client.post(path, data=raw, headers={'X-Signature': signature})).status == 202
            streams = webhook_store.streams()
            queued_ids, _ = webhook_store.claim(*streams[0])
            assert len(queued_ids) == expected
            webhook_store.complete(queued_ids)
        declaration.write_text('scope: [malformed')
        webhook_store.reconcile()
        assert webhook_store.route(token)
        declaration.unlink()
        webhook_store.reconcile()
        assert (await client.post(path,data=raw,headers=headers)).status == 404


@pytest.mark.asyncio
async def test_native_dispatch_keeps_stream_session_but_new_message_ids(tmp_path, monkeypatch):
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.config import PlatformConfig
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    events = []
    async def receive(event):
        events.append(event)
    monkeypatch.setattr(adapter, 'handle_message', receive)
    route = {'responsibility_stream':'stable-key','toolsets':['hermes-cli'],'deliver':'log'}
    for delivery in ('first','second'):
        await adapter._spawn_agent_run({}, 'fresh prompt', delivery, 1, route_config=route,
                                       route_name='responsibility-stable-key', profile=None, event_type='responsibility')
    assert events[0].source.chat_id == events[1].source.chat_id
    assert events[0].message_id != events[1].message_id
    adapter._routes.clear()  # a native dynamic-subscription reload must not strip trusted tools
    assert adapter.toolsets_for_source(events[-1].source) == ['hermes-cli']


@pytest.mark.asyncio
async def test_processing_completion_records_failure_without_replay(tmp_path, monkeypatch):
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.config import PlatformConfig
    from gateway.platforms.event import ProcessingOutcome
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._responsibility_ingress = ResponsibilityIngress(adapter)
    source = SimpleNamespace(user_id='webhook:responsibility-stream', chat_id='webhook:stream')
    for index, (outcome, expected) in enumerate(((ProcessingOutcome.SUCCESS, 'done'),
            (ProcessingOutcome.FAILURE, 'failed'), (ProcessingOutcome.CANCELLED, 'interrupted'))):
        webhook_store.accept('token', 'stream', str(index), {'event': index})
        ids, _ = webhook_store.claim('token', 'stream')
        adapter._responsibility_ingress.active['stream'] = (tmp_path, ids)
        await adapter.on_processing_complete(SimpleNamespace(source=source), outcome)
        with webhook_store.database() as db:
            assert db.execute('SELECT state FROM deliveries WHERE id=?', (ids[0],)).fetchone()[0] == expected
        assert not webhook_store.streams()


@pytest.mark.asyncio
@pytest.mark.parametrize('report', ['muted', 'local'])
async def test_shared_listener_routes_and_completes_in_owning_profile(tmp_path, monkeypatch, report):
    from pathlib import Path
    from gateway.run import _profile_runtime_scope
    from gateway.platforms.webhook import WebhookAdapter
    from gateway.platforms.event import ProcessingOutcome
    from gateway.config import PlatformConfig
    from hermes_cli import profiles
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    homes = {'default': tmp_path / '.hermes', 'second': tmp_path / '.hermes/profiles/second'}
    monkeypatch.setenv('HERMES_HOME', str(homes['default']))
    monkeypatch.setattr(profiles, 'profiles_to_serve', lambda **kwargs: list(homes.items()))
    tokens = {}
    for name, home in homes.items():
        package = home / 'responsibilities/inbox'
        (package / 'webhooks').mkdir(parents=True)
        (package / 'references').mkdir()
        (package / 'RESPONSIBILITY.md').write_text('---\nname: inbox\ntrigger: Events\n---\n' + name)
        (package / 'STATE.md').write_text('')
        (package / 'webhooks/events.yaml').write_text('scope: Handle event\nreport: ' + report + '\nverify:\n  signature: hmac_sha256(body)\n  header: X-Signature\n  secret: env:PROVIDER_WEBHOOK_SECRET\n')
        (home / '.env').write_text('PROVIDER_WEBHOOK_SECRET=' + name + '\n')
        (home / 'config.yaml').write_text('platforms:\n  webhook:\n    enabled: true\nwebhook:\n  public_url: https://example.test\n')
        with _profile_runtime_scope(home):
            webhook_store.reconcile()
            receipt = webhook_store.receipt('inbox', 'events')
            with webhook_store.database() as db:
                tokens[name] = db.execute('SELECT token FROM routes').fetchone()[0]
            assert receipt['webhook_url'].endswith(('/p/second' if name == 'second' else '') + '/responsibilities/' + tokens[name])
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    ingress = ResponsibilityIngress(adapter)
    events = []
    tasks = []
    async def receive(event):
        events.append(event)
    monkeypatch.setattr(adapter, 'handle_message', receive)
    original = adapter._spawn_agent_run
    def spawn(*args, **kwargs):
        task = original(*args, **kwargs)
        tasks.append(task)
        return task
    monkeypatch.setattr(adapter, '_spawn_agent_run', spawn)
    app = web.Application()
    app.router.add_route('*', '/responsibilities/{token}', ingress.handle)
    app.router.add_route('*', '/p/{profile}/responsibilities/{token}', ingress.handle)
    async with TestClient(TestServer(app)) as client:
        for index, name in enumerate(('default', 'second', 'default')):
            raw = json.dumps({'event': index}).encode()
            signature = hmac.new(name.encode(), raw, hashlib.sha256).hexdigest()
            path = ('/p/second' if name == 'second' else '') + '/responsibilities/' + tokens[name]
            assert (await client.post(path, data=raw, headers={'X-Signature': signature})).status == 202
            for profile, home in ingress._profiles():
                with _profile_runtime_scope(home):
                    assert ingress._enabled()
                    ingress._dispatch_profile(profile, home)
            await tasks[-1]
            event = events[-1]
            assert (await adapter.send(event.source.chat_id, 'Completed report')).success
            assert event.source.profile == name
            ingress.completed(event.source, ProcessingOutcome.SUCCESS)
            with _profile_runtime_scope(homes[name]), webhook_store.database() as db:
                assert db.execute("SELECT count(*) FROM deliveries WHERE state!='done'").fetchone()[0] == 0
        assert (await client.post('/p/unknown/responsibilities/' + tokens['default'])).status == 404
        assert (await client.post('/p/second/responsibilities/' + tokens['default'])).status == 404
    assert len(events) == 3


@pytest.mark.asyncio
async def test_unknown_webhook_does_not_scan_or_create_database(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    def forbidden_scan():
        raise AssertionError('public request must never scan responsibility files')
    monkeypatch.setattr(webhook_store, 'reconcile', forbidden_scan)
    ingress = ResponsibilityIngress(SimpleNamespace(_max_body_bytes=1000000))
    app = web.Application()
    app.router.add_route('*', '/responsibilities/{token}', ingress.handle)
    async with TestClient(TestServer(app)) as client:
        assert (await client.post('/responsibilities/unknown')).status == 404
    assert not (tmp_path / 'webhooks/responsibilities.db').exists()


def test_webhook_receipt_requires_enabled_ingress(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = tmp_path / 'config.yaml'
    with webhook_store.database() as db:
        db.execute("INSERT INTO routes VALUES ('token','inbox','event','{}','hash',NULL)")
    for enabled in ('false', 'true', 'false'):
        config.write_text('webhook:\n  public_url: https://example.test\nplatforms:\n  webhook:\n    enabled: ' + enabled + '\n')
        receipt = webhook_store.receipt('inbox', 'event')
        if enabled == 'true':
            assert receipt['webhook_url'].endswith('/responsibilities/token')
        else:
            assert 'warning' in receipt and 'webhook_url' not in receipt
