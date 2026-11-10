"""Authenticated duplicate readback when an explicit recipient disappears."""
from pathlib import Path

import pytest

from tests.gateway.test_webhook_prepared_origins import ingress, post, records


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [True, False])
async def test_suppressed_duplicate_after_recipient_removal(tmp_path, monkeypatch, direct):
    from hermes_cli.profiles import get_profile_dir
    from hermes_cli.plugins import get_plugin_manager

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    receiver = get_profile_dir('receiver')
    receiver.mkdir(parents=True)
    (receiver / 'config.yaml').write_text('{}\n')
    calls = []
    if not direct:
        manager = get_plugin_manager()
        manager.discover_and_load()
        def consume(**event):
            calls.append(event)
            return {'action': 'handled', 'reply': None}
        monkeypatch.setitem(manager._hooks, 'post_gateway_admission', [consume])
    async with ingress(direct=direct) as (client, adapter, authority):
        adapter._routes['fixture']['deliver'] = 'bot-chat:receiver'
        if direct:
            adapter._routes['fixture']['script'] = 'owned-script'
            def script(command, payload):
                calls.append(payload)
                return True, payload
            monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        status, first = await post(client, '')
        assert status == 202 and first['status'] == 'suppressed'
        receiver.rename(receiver.with_name('removed-receiver'))
        status, replay = await post(client, '')
        assert status == 202
        assert replay == first
        assert len(calls) == 1
        assert records(authority)[0]['delivery_id'] == first['delivery_id']
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    from hermes_cli.profiles import get_profile_dir
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = get_profile_dir('receiver')
    home.mkdir(parents=True)
    (home / 'config.yaml').write_text('{}\n')
    return home


def consume_reply(monkeypatch, calls, reply):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    manager.discover_and_load()
    def consume(**event):
        calls.append(event)
        return {'action': 'handled', 'reply': reply}
    monkeypatch.setitem(manager._hooks, 'post_gateway_admission', [consume])


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [True, False])
async def test_pending_duplicate_refuses_removed_or_recreated_target_without_rebinding(receiver, monkeypatch, direct):
    calls = []
    if not direct:
        consume_reply(monkeypatch, calls, 'actual hook reply')
    async with ingress(direct=direct) as (client, adapter, authority):
        adapter._routes['fixture']['deliver'] = 'bot-chat:receiver'
        status, first = await post(client)
        assert status == 202 and first['status'] == 'pending'
        original, = records(authority)
        receiver.rename(receiver.with_name('removed-receiver'))
        status, replay = await post(client)
        assert status == 202 and replay['status'] == 'refused'
        assert replay['origin_id'] == first['origin_id'] and replay['delivery_id'] == first['delivery_id']
        refused, = records(authority)
        assert refused['reason'] == 'target_or_receipt_mismatch'
        assert refused['target_pin'] == original['target_pin'] and refused['native'] == original['native']
        receiver.mkdir()
        (receiver / 'config.yaml').write_text('{}\n')
        assert (await post(client)) == (202, replay)
        assert not (receiver / 'state.db').exists()
        assert len(calls) == (0 if direct else 1)
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [True, False])
async def test_new_event_missing_target_refuses_before_script_or_hook(receiver, monkeypatch, direct):
    calls = []
    if not direct:
        consume_reply(monkeypatch, calls, 'reply')
    async with ingress(direct=direct) as (client, adapter, authority):
        route = adapter._routes['fixture']
        route['deliver'] = 'bot-chat:receiver'
        route['script'] = 'owned-script'
        def script(command, payload):
            calls.append(payload)
            return True, payload
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        assert (await post(client, ''))[0] == 202
        before, = records(authority)
        call_count = len(calls)
        receiver.rename(receiver.with_name('removed-receiver'))
        assert (await post(client, '', key='new-event'))[0] == 503
        assert records(authority) == [before] and len(calls) == call_count


@pytest.mark.asyncio
@pytest.mark.parametrize('change,expected', [('prompt', 409), ('target', 409), ('disabled', 403),
    ('removed', 404), ('source_revoked', 503), ('source_identity', 409), ('transport_revoked', 503), ('credential', 401)])
async def test_terminal_duplicate_still_revalidates_current_grant(receiver, monkeypatch, change, expected):
    async with ingress() as (client, adapter, authority):
        adapter._routes['fixture']['deliver'] = 'bot-chat:receiver'
        assert (await post(client, ''))[0] == 202
        before, = records(authority)
        receiver.rename(receiver.with_name('removed-receiver'))
        route = adapter._routes['fixture']
        original = adapter._admission_source
        def changed(*args):
            source = original(*args)
            source.user_name = 'different authenticated source name'
            return source
        changes = {
            'prompt': lambda: route.update(prompt='changed: {text}'),
            'target': lambda: route.update(deliver='bot-chat:missing'),
            'disabled': lambda: route.update(enabled=False),
            'removed': lambda: adapter._routes.pop('fixture'),
            'source_revoked': lambda: monkeypatch.setattr(authority.runner, '_is_user_authorized_for_source',
                                                         lambda *a, **kw: False),
            'source_identity': lambda: monkeypatch.setattr(adapter, '_admission_source', changed),
            'transport_revoked': authority.runner.adapters.clear,
            'credential': lambda: route.update(secret='changed-secret'),
        }
        changes[change]()
        status, refused = await post(client, '')
        assert status == expected, refused
        assert 'origin_id' not in refused and 'delivery_id' not in refused
        assert records(authority) == [before]


@pytest.mark.asyncio
async def test_origin_unknown_duplicate_after_target_removal_never_repeats_script(receiver, monkeypatch):
    calls = []
    async with ingress() as (client, adapter, authority):
        adapter._routes['fixture'].update(deliver='bot-chat:receiver', script='interrupted-script')
        def interrupted(command, payload):
            calls.append(payload)
            raise RuntimeError('side effect interrupted')
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', interrupted)
        assert (await post(client))[0] == 503
        original, = records(authority)
        assert original['state'] == 'preparing'
        receiver.rename(receiver.with_name('removed-receiver'))
        status, replay = await post(client)
        assert status == 202 and replay['status'] == 'origin_unknown'
        assert replay['origin_id'] == original['origin_id'] and replay['delivery_id'] == original['delivery_id']
        assert (await post(client)) == (202, replay) and len(calls) == 1


@pytest.mark.asyncio
async def test_rendered_coalescer_sibling_uses_retained_pin_and_identity(receiver, monkeypatch):
    from gateway.webhook_origins import admit_rendered_origin
    from hermes_state_runtime import RuntimeStoreError
    calls = []
    consume_reply(monkeypatch, calls, None)
    async with ingress(direct=False) as (_client, adapter, authority):
        config = adapter._routes['fixture']
        config.update(deliver='bot-chat:receiver', coalesce={'key': '{group}', 'debounce_seconds': 1})
        async def submit(key='group-event', prompt='rendered group'):
            return await admit_rendered_origin(adapter, config, 'fixture', None, {'group': 'same'}, prompt, key)
        assert await submit() == 'accepted'
        before, = records(authority)
        assert before['state'] == 'suppressed'
        receiver.rename(receiver.with_name('removed-receiver'))
        assert await submit() == 'accepted'
        assert records(authority) == [before] and len(calls) == 1
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await submit(prompt='changed rendered group')
        with pytest.raises(ValueError, match='target profile unavailable'):
            await submit(key='new-group-event')
        config['prompt'] = 'changed trusted contract'
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await submit()
        assert records(authority) == [before] and len(calls) == 1
