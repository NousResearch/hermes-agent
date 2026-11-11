"""Real atomic subscription reload must not erase an authorized durable origin.

This owns the live-subscription boundary, distinct from static-route origin tests.
No recipient receipt or model result is manufactured here; settled native readback
is covered by the active-reconciliation integration sibling.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from hermes_cli import webhook as subscriptions
from tests.gateway.test_webhook_origin_removed_target import consume_reply
from tests.gateway.test_webhook_prepared_origins import ingress, post, records


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    from hermes_cli.profiles import get_profile_dir
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = get_profile_dir('receiver')
    home.mkdir(parents=True)
    (home / 'config.yaml').write_text('{}\n')
    return home


def publish_fixture(adapter):
    route = deepcopy(adapter._routes['fixture'])
    route['deliver'] = 'bot-chat:receiver'
    route['script'] = 'owned-script'
    subscriptions._replace_subscription('fixture', route, subscriptions._MISSING_SUBSCRIPTION)
    adapter._static_routes.clear()
    adapter._routes.clear()
    return route


def unrelated_update(adapter, route):
    previous = adapter._dynamic_routes_stat
    subscriptions._replace_subscription('unrelated', {'secret': 'unrelated-secret', 'deliver': 'log'},
                                        subscriptions._MISSING_SUBSCRIPTION)
    assert subscriptions._load_subscriptions()['fixture'] == route
    adapter._reload_dynamic_routes()
    assert adapter._dynamic_routes_stat != previous


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [True, False])
@pytest.mark.parametrize('text', ['', 'actual template'])
async def test_dynamic_duplicate_survives_unrelated_reload_and_target_removal(
        receiver, monkeypatch, record_property, direct, text):
    script_calls, hook_calls = [], []
    if not direct:
        consume_reply(monkeypatch, hook_calls, text or None)
    async with ingress(direct=direct) as (client, adapter, authority):
        route = publish_fixture(adapter)
        def script(command, payload):
            script_calls.append(payload)
            return True, payload
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        status, first = await post(client, text)
        assert status == 202 and first['status'] == ('pending' if text else 'suppressed')
        before, = records(authority)
        receiver.rename(receiver.with_name('removed-receiver'))
        unrelated_update(adapter, route)
        status, replay = await post(client, text)
        assert status == 202, replay
        assert replay['origin_id'] == first['origin_id'] and replay['delivery_id'] == first['delivery_id']
        assert replay['status'] == ('refused' if text else 'suppressed')
        retained, = records(authority)
        assert retained['target_pin'] == before['target_pin'] and retained['native'] == before['native']
        assert (await post(client, text)) == (202, replay)
        assert script_calls == [{'text': text}] and len(hook_calls) == (0 if direct else 1)
        assert authority.db._read_all('SELECT * FROM session_admissions') == []
        assert (await post(client, text, key='new-event'))[0] == 503
        assert records(authority) == [retained]
        assert script_calls == [{'text': text}] and len(hook_calls) == (0 if direct else 1)
        record_property('dynamic_removed_target_duplicate', json.dumps({'direct': direct, 'first': first,
            'replay': replay, 'target_pin': retained['target_pin'], 'scripts': len(script_calls), 'hooks': len(hook_calls)}))
        fresh = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-secret',
            'host': '127.0.0.1', 'port': 0}))
        try:
            assert await fresh.connect()
            assert fresh._routes['fixture'] == route
        finally:
            await fresh.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize('change,expected', [
    ({'enabled': False}, 403), (None, 404), ({'secret': 'changed-secret'}, 401),
    ({'secret': ''}, 404), ({'secret': 1}, 404),
    ({'deliver': 'bot-chat:../receiver'}, 404), ({'deliver_extra': {'session_id': 'elsewhere'}}, 404),
    ({'coalesce': {'key': '{text}', 'window_seconds': 1}}, 404),
    ({'profile': 'unserved'}, 404), ({'prompt': 'changed: {text}'}, 409),
    ({'deliver': 'bot-chat:missing'}, 409),
])
async def test_dynamic_reload_still_enforces_current_route_grant(receiver, change, expected):
    async with ingress() as (client, adapter, authority):
        route = publish_fixture(adapter)
        # No script is needed for this suppressed origin.
        route.pop('script')
        subscriptions._replace_subscription('fixture', route, subscriptions._load_subscriptions()['fixture'])
        assert (await post(client, ''))[0] == 202
        before, = records(authority)
        receiver.rename(receiver.with_name('removed-receiver'))
        def mutate(routes):
            if change is None:
                routes.pop('fixture')
            else:
                routes['fixture'].update(change)
        subscriptions._mutate_subscriptions(mutate)
        status, refused = await post(client, '')
        assert status == expected, refused
        assert 'origin_id' not in refused and 'delivery_id' not in refused
        assert records(authority) == [before]


def test_configuration_validation_still_requires_existing_target(receiver):
    from gateway.platforms.webhook_delivery import validate_bot_chat_route
    receiver.rename(receiver.with_name('removed-receiver'))
    with pytest.raises(ValueError, match='target profile unavailable'):
        validate_bot_chat_route({'deliver': 'bot-chat:receiver'})
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-secret'}))
    with pytest.raises(ValueError, match='target profile unavailable'):
        adapter._validate_route('configured', {'deliver': 'bot-chat:receiver'})
