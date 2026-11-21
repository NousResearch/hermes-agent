"""Public webhook operator contracts, independent of recipient inference."""
import argparse
import json
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def operator(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    import hermes_cli.webhook as wh
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    home = tmp_path / '.hermes'
    home.mkdir()
    for name in ('producer', 'recipient'):
        p = home / 'profiles' / name
        p.mkdir(parents=True)
        (p / 'config.yaml').write_text('{}')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(wh, '_is_webhook_enabled', lambda: True)
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return home, client, _SESSION_HEADER_NAME


@pytest.mark.parametrize('target', ['bot-chat', 'bot-chat:recipient'])
def test_http_target_and_execution_profile_roundtrip(operator, target):
    import hermes_cli.webhook as wh
    home, client, _ = operator
    response = client.post('/api/webhooks', json=dict(name='fixture', deliver=target,
        route_profile='producer', deliver_only=True, mirror_to_session=True))
    assert response.status_code == 200, response.text
    route = response.json()
    assert route['profile'] == 'producer'
    assert route['deliver'] == target
    assert route['mirror_to_session'] is True
    assert route['url'].endswith('/p/producer/webhooks/fixture')
    stored = wh._load_subscriptions()['fixture']
    assert stored['profile'] == 'producer' and stored['deliver'] == target
    listing = client.get('/api/webhooks').json()['subscriptions'][0]
    assert listing == {k: v for k, v in route.items() if k != 'secret'}
    assert response.json()['secret'] not in json.dumps(listing)
    assert not (home / 'profiles/recipient/webhook_subscriptions.json').exists()


def seed(home, prefix, number, **overrides):
    value = dict(origin_id=f'{number:064x}', delivery_id=f'{number + 100:064x}',
        origin_kind='direct_template', source_outcome='completed', target='bot-chat:recipient',
        state='queued', native_status='queued', native_admission_id='a' * 32,
        route='fixture', reason='owner_unavailable', message='PRIVATE-MESSAGE',
        producer_home='/private/home', target_home='/private/recipient',
        exception='PRIVATE-EXCEPTION', credentials='PRIVATE-CREDENTIAL')
    value.update(overrides)
    with sqlite3.connect(home / 'state.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS state_meta(key TEXT PRIMARY KEY,value TEXT)')
        conn.execute('INSERT INTO state_meta VALUES (?,?)', (prefix + value['origin_id'], json.dumps(value)))
    return value


def test_authenticated_bounded_delivery_inspection_and_cli(operator, capsys):
    from hermes_cli.subcommands.webhook import build_webhook_parser
    import hermes_cli.webhook as wh
    home, client, header = operator
    prefixes = ('webhook.final.intent.v1.', 'webhook.origin.v1.')
    for n in range(55):
        seed(home, prefixes[n % 2], n, origin_kind='agent_final' if n % 2 == 0 else 'hook_reply')
    seed(home / 'profiles/producer', prefixes[0], 90, state='settled')
    response = client.get('/api/webhooks/deliveries')
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data['deliveries']) == 50 and data['next_cursor']
    next_page = client.get('/api/webhooks/deliveries', params={'cursor': data['next_cursor']}).json()
    assert len(next_page['deliveries']) == 5 and next_page['next_cursor'] is None
    rows = data['deliveries'] + next_page['deliveries']
    assert {r['namespace'] for r in rows} == {'webhook.final.intent.v1', 'webhook.origin.v1'}
    for row in rows:
        assert row['state'] == 'queued' and row['native_status'] == 'queued'
        assert row['source_outcome'] == 'completed' and row['target_profile'] == 'recipient'
        assert set(row) == {'namespace', 'origin_id', 'origin_kind', 'delivery_id', 'source_outcome',
                            'target_profile', 'state', 'native_status', 'native_admission_id', 'source_admission_id', 'reason'}
    assert 'PRIVATE' not in response.text and '/private' not in response.text
    named = client.get('/api/webhooks/deliveries?profile=producer').json()['deliveries']
    assert len(named) == 1 and named[0]['state'] == 'settled'
    assert client.get('/api/webhooks/deliveries?profile=missing').status_code == 404
    assert client.get('/api/webhooks/deliveries?cursor=../../private').status_code == 400
    client.headers.pop(header)
    assert client.get('/api/webhooks/deliveries').status_code == 401
    parser = argparse.ArgumentParser()
    build_webhook_parser(parser.add_subparsers(), cmd_webhook=wh.webhook_command)
    args = parser.parse_args(['webhook', 'deliveries', '--cursor', data['next_cursor']])
    assert wh.webhook_command(args) in (None, 0)
    assert json.loads(capsys.readouterr().out) == next_page


def test_corrupt_projection_fields_never_echo_private_values(operator):
    home, client, _ = operator
    seed(home, 'webhook.origin.v1.', 1, reason='PRIVATE-EXCEPTION /private/home',
         native_status='PRIVATE-TOKEN', source_outcome='PRIVATE-OUTCOME', target='bot-chat:../../private')
    response = client.get('/api/webhooks/deliveries')
    assert response.status_code == 200
    assert 'PRIVATE' not in response.text and '/private' not in response.text
    row = response.json()['deliveries'][0]
    assert row['reason'] == 'unrecognized' and row['native_status'] is None
    assert row['source_outcome'] == 'unknown' and row['target_profile'] is None


@pytest.mark.asyncio
async def test_mcp_list_readback_is_limited_to_this_servers_routes(operator, monkeypatch):
    from tools import mcp_tool_events as events
    import hermes_cli.webhook as wh
    home, _, _ = operator
    wh._mutate_subscriptions(lambda subs: subs.update({'fixture': dict(deliver='bot-chat:recipient',
        profile='producer', mcp_event=dict(server='fixture-server', name='changed', arguments={}))}))
    seed(home, 'webhook.origin.v1.', 1)
    seed(home, 'webhook.final.intent.v1.', 2, route='other-server')
    async def send(*args):
        return {'events': []}
    monkeypatch.setattr(events, '_send', send)
    response = await events._list_event_types(None, {}, 'fixture-server')
    assert response['subscriptions'][0]['deliver'] == 'bot-chat:recipient'
    assert response['subscriptions'][0]['profile'] == 'producer'
    assert len(response['deliveries']['deliveries']) == 1
    assert 'PRIVATE' not in json.dumps(response)


@pytest.mark.parametrize('invalid', [
    {'deliver': 'bot-chat:'}, {'deliver': 'bot-chat:recipient/session'},
    {'deliver': 'bot-chat:missing'}, {'deliver': 'bot-chat:recipient,producer'},
    {'deliver': 'bot-chat', 'deliver_chat_id': 'other'},
    {'deliver': 'bot-chat', 'deliver_extra': {'session_id': 'other'}},
    {'deliver': 'bot-chat', 'route_profile': 'missing'},
])
def test_http_rejects_invalid_addressing_before_writing(operator, invalid):
    import hermes_cli.webhook as wh
    _, client, _ = operator
    response = client.post('/api/webhooks', json={'name': 'invalid', **invalid})
    assert response.status_code in (400, 422), response.text
    assert 'invalid' not in wh._load_subscriptions()


@pytest.mark.parametrize('target', ['bot-chat', 'bot-chat:recipient'])
def test_cli_create_and_mcp_consent_preserve_complete_destination(operator, capsys, monkeypatch, target):
    import hermes_cli.webhook as wh
    from hermes_cli.subcommands.webhook import build_webhook_parser
    from tools.mcp_tool_events import _consent_error, _subscribe_args_error
    prompts = []
    monkeypatch.setattr('tools.approval_prompt.request_elicitation_consent',
                        lambda *args, **kwargs: prompts.append(args) or 'accept')
    parser = argparse.ArgumentParser()
    build_webhook_parser(parser.add_subparsers(), cmd_webhook=wh.webhook_command)
    args = parser.parse_args(['webhook', 'subscribe', 'fixture', '--route-profile', 'producer',
                             '--deliver', target, '--deliver-only', '--mirror-to-session'])
    wh.webhook_command(args)
    output = capsys.readouterr().out
    assert 'zero LLM cost' not in output
    assert 'receiving Bot Chat may run its model' in output
    assert target in output
    saved = wh._load_subscriptions()['fixture']
    assert saved['profile'] == 'producer' and saved['deliver'] == target
    assert saved['mirror_to_session'] is True
    assert _subscribe_args_error({'deliver': target}) is None
    assert _consent_error('fixture-server', {'name': 'changed', 'deliver': target}) is None
    assert target in prompts[0][0] and 'may run its model' in prompts[0][0]



def test_scoped_implicit_route_profile_and_empty_inspection_do_not_create_a_database(operator):
    home, client, _ = operator
    empty = client.get('/api/webhooks/deliveries?profile=producer')
    assert empty.status_code == 200 and empty.json() == {'deliveries': [], 'next_cursor': None}
    assert not (home / 'profiles/producer/state.db').exists()
    created = client.post('/api/webhooks?profile=producer', json={'name': 'local', 'deliver': 'bot-chat'})
    assert created.status_code == 200
    assert created.json()['profile'] == 'producer' and created.json()['route_profile'] == 'default'
    assert created.json()['url'].endswith('/webhooks/local')
    assert '/p/producer/' not in created.json()['url']
    assert not (home / 'webhook_subscriptions.json').exists()
    listing = client.get('/api/webhooks?profile=producer').json()['subscriptions']
    assert listing[0]['profile'] == 'producer'


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['bot-chat', 'bot-chat:recipient'])
async def test_mcp_creation_and_readback_use_the_same_target(operator, monkeypatch, target):
    import hermes_cli.webhook as wh
    from tools import mcp_tool_events as events
    from tests.tools.test_mcp_events import FakeSession, _grant
    monkeypatch.setattr(events, '_callback_base', lambda: 'https://fixture.example')
    session = FakeSession(_grant(), {'events': []})
    args = {'name': 'changed', 'deliver': target}
    assert events._subscribe_args_error(args) is None
    result = await events._subscribe_event(session, args, 'fixture-server')
    assert result['subscribed']['deliver'] == target
    route, = wh._load_subscriptions().values()
    assert route['deliver'] == target and route['profile'] == 'default'
    listed = await events._list_event_types(session, {}, 'fixture-server')
    assert listed['subscriptions'][0]['deliver'] == target
    assert listed['subscriptions'][0]['profile'] == 'default'
    assert route['secret'] not in json.dumps(listed)
