"""Durable producer retries retain batch identities after an owner restart."""
import asyncio
from copy import deepcopy

import pytest


@pytest.mark.asyncio
async def test_coalesced_siblings_survive_lost_ack_and_owner_restart(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_ingress_context import native_callback
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_completion_admission import pending
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tools import async_delegation as delegation

    # Fault boundary: stop after the actual SQLite commit but before execution.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='before')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    human = MessageEvent(text='human', source=source, message_id='human')
    from hermes_constants import get_hermes_home
    with native_callback(runner, human, get_hermes_home()):
        await authority.admit_native(human)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    events = [dict(pending(key, 'durable-sibling-' + str(i)), parent_session_id=sid) for i in range(2)]
    assert await runner._deliver_async_delegation_group(events) is True
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 2 and all(e['delegation_id'] in rows[-1]['payload']['text'] for e in events)
    # Simulate lost ACK of the committed producer handoff, then real new epoch.
    for event in events:
        delegation._persist_dispatch(event)
        delegation._persist_completion(event, {'status': 'completed', 'summary': event['summary']})
    runner._completion_deliveries_delivered.clear()
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='after')
    duplicate = deepcopy(events[1])
    assert await runner._deliver_async_delegation_group([duplicate]) is True
    rows_after = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows_after) == len(rows), rows_after
    fresh = dict(pending(key, 'fresh-sibling'), parent_session_id=sid)
    assert await runner._deliver_async_delegation_group([events[0], fresh]) is True
    final = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(final) == 3 and 'fresh-sibling' in final[-1]['payload']['text']
    assert 'durable-sibling' not in final[-1]['payload']['text']
    for event in [*events, fresh]:
        assert delegation.get_durable_delegation(event['delegation_id'])['delivery_state'] == 'delivered'
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_unowned_completion_refunds_every_sibling_without_consuming_budget(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_completion_admission import pending
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tools import async_delegation as delegation

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='guards')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    event = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, event, get_hermes_home()):
        await authority.admit_native(event)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    events = [pending(key, 'unowned-' + str(i)) for i in range(2)]
    for _ in range(10):
        assert await runner._deliver_async_delegation_group(events) is False
    for event in events:
        record = delegation.get_durable_delegation(event['delegation_id'])
        assert (record['delivery_state'], record['delivery_attempts']) == ('pending', 0)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1
    # An explicit live but foreign parent must not be switched into this route.
    authority.db.create_session('other-owner', source='discord')
    for event in events:
        event['parent_session_id'] = 'other-owner'
    for _ in range(10):
        assert await runner._deliver_async_delegation_group(events) is False
    assert runner.session_store.peek_session_id(key) == sid
    for event in events:
        record = delegation.get_durable_delegation(event['delegation_id'])
        assert (record['delivery_state'], record['delivery_attempts']) == ('pending', 0)
        event['parent_session_id'] = sid
    assert await runner._deliver_async_delegation_group(events) is True
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 2
