"""Recipient, native commit and inventory-continuation denial boundaries."""
import asyncio
from dataclasses import replace
import threading

import pytest

from gateway import hosted_rooms as rooms
from gateway.session_controls import AuthorityConnection
from gateway.session_group_controls import GROUP_METHODS, dispatch_group_control
from gateway.session_group_messaging_read import _attest_inventory, read_messaging_inventory_page
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch
from tests.gateway.test_messaging_inventory_binding import (
    Receiver, bound, enrolled, grant_params, recipient, rpc, snapshot,
)

__all__ = ['bound']


@pytest.mark.asyncio
@pytest.mark.parametrize('denial', [
    'unbound', 'user', 'chat', 'thread', 'scope', 'platform', 'transport_profile', 'runtime_profile',
    'bot', 'edit', 'anonymous', 'unknown_relay', 'shared', 'not_one_to_one', 'unstamped',
    'unregistered', 'replaced', 'route_rejected', 'talk_policy', 'command_policy',
    'missing_service', 'missing_checker', 'wrong_service_db', 'no_registry', 'raw_bot', 'raw_edit',
])
async def test_denied_sources_never_fetch_private_inventory(bound, monkeypatch, denial):
    if denial != 'unbound':
        await enrolled(bound)
    event = bound.event()
    if denial == 'user':
        event.source.user_id = 'person-2'
        bound.adapter.config.extra['allow_from'].append('person-2')
        bound.adapter.config.extra['allow_admin_from'].append('person-2')
    elif denial in {'chat', 'thread', 'scope'}:
        setattr(event.source, denial + '_id', 'other')
    elif denial == 'platform':
        from gateway.config import Platform
        event.source.platform = Platform.TELEGRAM
    elif denial == 'transport_profile':
        bound.runner.adapters.clear()
        bound.runner._profile_adapters = {'other': {bound.adapter.platform: bound.adapter}}
    elif denial == 'runtime_profile':
        event.source.profile = 'other'
    elif denial == 'bot':
        event.source.is_bot = True
    elif denial == 'edit':
        event.source.message_is_edit = True
    elif denial == 'anonymous':
        event.source.user_id = 'anonymous'
    elif denial == 'unknown_relay':
        event.source.delivered_via_upstream_relay = True
    elif denial == 'shared':
        event.source.chat_type = 'group'
    elif denial == 'not_one_to_one':
        event.source.is_one_to_one = None
    elif denial == 'unstamped':
        del event.source._transport_adapter_ref
    elif denial == 'unregistered':
        bound.runner.adapters.clear()
    elif denial == 'replaced':
        bound.runner.adapters[bound.adapter.platform] = Receiver(bound.runner, bound.adapter.config)
    elif denial == 'route_rejected':
        event.source.profile_route_rejected = True
    elif denial == 'talk_policy':
        bound.adapter.config.extra['allow_from'] = ['other']
    elif denial == 'command_policy':
        bound.adapter.config.extra['allow_admin_from'] = ['other']
    elif denial == 'missing_service':
        bound.authority.hosted_room_service = None
    elif denial == 'missing_checker':
        bound.service.authorize_room = None
    elif denial == 'wrong_service_db':
        bound.service.db_path = bound.home / 'wrong.db'
    elif denial == 'no_registry':
        bound.runner.session_authorities = None
    elif denial == 'raw_bot':
        event.raw_message = {'subtype': 'bot_message'}
    elif denial == 'raw_edit':
        event.raw_message = {'subtype': 'message_changed'}
    monkeypatch.setattr(rooms, 'list_rooms', bound.forbidden)
    before = snapshot(bound.db)
    with pytest.raises(RuntimeStoreError):
        await read_messaging_inventory_page(bound.runner, event)
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
async def test_attestation_never_inherits_owner_or_other_method_authority(bound, monkeypatch):
    await enrolled(bound)
    event = bound.event()
    context = _attest_inventory(bound.runner, event)
    assert context.actor.subject.startswith('messaging:')
    assert context.actor.subject != bound.alice.actor.subject
    assert context.actor.capabilities == frozenset({'session:read'})
    before = snapshot(bound.db)
    for method in [*GROUP_METHODS, 'profiles.list', 'unknown']:
        if method != 'groups.list':
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                await dispatch_group_control(context, method, {})
    with pytest.raises(RuntimeStoreError):
        bound.service.authorize_room(context.actor.subject, 'alice-room')
    monkeypatch.setattr(rooms, 'list_rooms', bound.forbidden)
    for params in ({'limit': 8, 'offset': 0, 'include_disbanded': True}, {'owner': 'native-alice'}, {}):
        with pytest.raises(RuntimeStoreError):
            await dispatch_group_control(context, 'groups.list', params)
    # An unbound context cannot silently adopt a later generation.
    with pytest.raises(RuntimeStoreError):
        await dispatch_group_control(replace(context, state_json=None), 'groups.list', {'limit': 8, 'offset': 0})
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
async def test_attested_subclass_cannot_escape_inventory_boundary(bound):
    from gateway.session_group_messaging_read import _InventoryRead
    await enrolled(bound)
    context = _attest_inventory(bound.runner, bound.event())
    class DerivedRead(_InventoryRead):
        pass
    inherited = DerivedRead(**vars(context))
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        await dispatch_group_control(inherited, 'groups.capabilities', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'expected_generation': True}, {'expected_generation': -1}, {'expected_generation': '0'},
    {'expected_generation': 2**63}, {'request_id': ''}, {'request_id': 'bad\n'},
    {'request_id': 'x' * 129}, {'recipient': {}}, {'recipient': recipient(user_id=12)},
    {'recipient': recipient(thread_id='')}, {'recipient': recipient(platform='made-up')},
    {'recipient': recipient(scope_id='x' * 257)}, {'recipient': recipient(owner='native-alice')},
    {'recipient': recipient(runtime_profile='other')}, {'subject': 'native-alice'},
])
async def test_native_strict_fields_write_nothing(bound, changes):
    before = snapshot(bound.db)
    assert 'error' in await rpc(bound.alice, params=grant_params(**changes))
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('identity_change,operator', [
    ({'native_bootstrap': False}, True), ({'provider': 'remote'}, True), ({}, False),
    ({'capabilities': ['session:read', 'session:create']}, True),
    ({'instance_id': 'other'}, True),
])
async def test_operator_flag_or_wire_native_claim_is_not_sufficient(bound, identity_change, operator):
    identity = dict(user_id='native-alice', provider='local', native_bootstrap=True,
        profile_id=bound.authority.profile_id, instance_id=bound.authority.instance_id,
        capabilities=['session:read', 'session:create', 'session:control']) | identity_change
    connection = AuthorityConnection(bound.authority, object(), identity, operator=operator)
    before = snapshot(bound.db)
    assert 'error' in await rpc(connection)
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('limit,offset', [(0, 0), (9, 0), (True, 0), ('8', 0), (1, -1),
                                        (1, True), (1, 4096), (8, 4095)])
async def test_page_bounds_deny_before_private_reads(bound, monkeypatch, limit, offset):
    await enrolled(bound)
    monkeypatch.setattr(rooms, 'list_rooms', bound.forbidden)
    with pytest.raises(RuntimeStoreError):
        await read_messaging_inventory_page(bound.runner, bound.event(), limit=limit, offset=offset)


def drift_native(bound, drift, original_write):
    a, c = bound.authority, bound.alice
    if drift == 'actor':
        c.actor = replace(c.actor, subject='native-bob')
    elif drift == 'transport':
        c.transport = object()
    elif drift == 'native_binding':
        a._native_legacy_transports.clear()
    elif drift == 'close':
        a.events.clear()
    elif drift == 'registry':
        from gateway.session_authorities import SessionAuthorities
        bound.runner.session_authorities = SessionAuthorities(bound.home)
        bound.runner.session_authorities.add(bound.home, a)
    elif drift == 'epoch':
        a.epoch += 1
    elif drift == 'stored_epoch':
        original_write(lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
    elif drift == 'instance':
        a.instance_id = 'replacement'
    elif drift == 'stored_instance':
        original_write(lambda conn: conn.execute("UPDATE runtime_epoch SET instance_id='replacement'"))
    elif drift == 'profile':
        a.profile_id = str(bound.home / 'other')
    elif drift == 'db':
        a.db = object()
    elif drift == 'native':
        c.native_owner = False


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['grant', 'revoke'])
@pytest.mark.parametrize('drift', ['actor', 'transport', 'native_binding', 'close', 'registry',
    'epoch', 'stored_epoch', 'instance', 'stored_instance', 'profile', 'db', 'native'])
async def test_native_context_frozen_before_actual_owner_write(bound, monkeypatch, method, drift):
    params = grant_params()
    if method == 'revoke':
        first = await enrolled(bound)
        params = dict(request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'],
            expected_generation=first['generation'])
    before = binding_rows(bound)
    entered, resume = threading.Event(), threading.Event()
    original = bound.db._execute_write
    def held(write):
        entered.set()
        assert resume.wait(10)
        return original(write)
    monkeypatch.setattr(bound.db, '_execute_write', held)
    pending = asyncio.create_task(rpc(bound.alice, method, params))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        drift_native(bound, drift, original)
    finally:
        resume.set()
    response = await asyncio.wait_for(pending, 10)
    assert 'error' in response, response
    assert binding_rows(bound) == before


def binding_rows(bound):
    with bound.db._read_ctx() as conn:
        return tuple(tuple(row) for row in conn.execute(
            "SELECT key,value FROM state_meta WHERE key LIKE 'gateway.messaging.read.%' ORDER BY key"))


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['revoke', 'aba', 'adapter', 'receiver_profile', 'event_user',
    'policy', 'registry', 'epoch', 'stored_epoch', 'instance', 'stored_instance', 'service', 'room_owner'])
async def test_captured_page_never_discloses_after_revocation_or_context_drift(bound, monkeypatch, drift):
    from gateway import session_group_controls as controls
    first = await enrolled(bound)
    entered, resume = threading.Event(), threading.Event()
    original = controls._group
    def held(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result['rooms'][0]['name'] == 'Alice inventory'
        entered.set()
        assert resume.wait(10)
        return result
    monkeypatch.setattr(controls, '_group', held)
    event = bound.event()
    context = _attest_inventory(bound.runner, event)
    pending = asyncio.create_task(dispatch_group_control(context, 'groups.list', {'limit': 8, 'offset': 0}))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if drift in {'revoke', 'aba'}:
            revoke = dict(request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'],
                expected_generation=first['generation'])
            assert (await rpc(bound.alice, 'revoke', revoke))['result']['active'] is False
            if drift == 'aba':
                new = (await rpc(bound.alice, params=grant_params(request_id='grant-two', expected_generation=2)))['result']
                assert new['binding_id'] != first['binding_id']
        elif drift == 'adapter':
            bound.runner.adapters[bound.adapter.platform] = Receiver(bound.runner, bound.adapter.config)
        elif drift == 'receiver_profile':
            bound.runner._primary_profile_name = 'other'
        elif drift == 'event_user':
            event.source.user_id = 'other'
        elif drift == 'policy':
            bound.adapter.config.extra['allow_from'] = ['other']
        elif drift == 'service':
            bound.authority.hosted_room_service = None
        elif drift == 'room_owner':
            bound.db._execute_write(lambda conn: conn.execute(
                "UPDATE state_meta SET value=? WHERE key=?", ('native-bob', 'gateway.hosted.owner.v1:alice-room')))
        else:
            drift_native(bound, drift, bound.db._execute_write)
    finally:
        resume.set()
    with pytest.raises(RuntimeStoreError):
        await asyncio.wait_for(pending, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['registry', 'adapter'])
async def test_context_drift_during_consent_read_is_not_accepted(bound, monkeypatch, drift):
    from gateway.session_group_messaging_read import _InventoryRead
    await enrolled(bound)
    context = _attest_inventory(bound.runner, bound.event())
    original = _InventoryRead._consent
    def changed(self, conn, recipient):
        result = original(self, conn, recipient)
        if drift == 'registry':
            drift_native(bound, 'registry', bound.db._execute_write)
        else:
            bound.runner.adapters[bound.adapter.platform] = Receiver(bound.runner, bound.adapter.config)
        return result
    monkeypatch.setattr(_InventoryRead, '_consent', changed)
    with pytest.raises(RuntimeStoreError):
        context.require_current()


@pytest.mark.asyncio
async def test_restart_preserves_durable_consent_but_not_old_attestation(bound):
    from gateway.session_authority import SessionAuthority
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_hosted_service import CanonicalHostedRoomService
    await enrolled(bound)
    old = _attest_inventory(bound.runner, bound.event())
    before = binding_rows(bound)
    epoch = begin_runtime_epoch(bound.db, instance_id='restarted')
    successor = SessionAuthority(bound.runner, profile_id=str(bound.home), instance_id='restarted',
        db=bound.db, epoch=epoch)
    successor.hosted_room_service = CanonicalHostedRoomService(successor, asyncio.get_running_loop())
    bound.runner.session_authority = successor
    bound.runner.session_authorities = SessionAuthorities(bound.home)
    bound.runner.session_authorities.add(bound.home, successor)
    with pytest.raises(RuntimeStoreError):
        await dispatch_group_control(old, 'groups.list', {'limit': 8, 'offset': 0})
    result = await read_messaging_inventory_page(bound.runner, bound.event())
    assert result['rooms'] == [{'name': 'Alice inventory', 'member_count': 1}]
    assert binding_rows(bound) == before


@pytest.mark.asyncio
async def test_grants_reserve_receipt_capacity_for_revocation(bound, monkeypatch):
    from gateway import session_group_messaging_read as binding
    monkeypatch.setattr(binding, '_MAX_REQUESTS', 1)
    before = binding_rows(bound)
    assert 'error' in await rpc(bound.alice)
    assert binding_rows(bound) == before
    monkeypatch.setattr(binding, '_MAX_REQUESTS', 2)
    first = await enrolled(bound)
    revoke = dict(request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'], expected_generation=1)
    assert (await rpc(bound.alice, 'revoke', revoke))['result']['active'] is False
    assert 'error' in await rpc(bound.alice, params=grant_params(request_id='new-grant', expected_generation=2))


@pytest.mark.asyncio
async def test_corrupt_binding_and_sql_failure_do_not_create_authority(bound, monkeypatch):
    first = await enrolled(bound)
    before = binding_rows(bound)
    with bound.db._read_ctx() as conn:
        binding_key = conn.execute("SELECT key FROM state_meta WHERE key LIKE 'gateway.messaging.read.v1.binding.%'").fetchone()[0]
    bound.db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.messaging.read.v1.request.%' "
        "BEGIN SELECT RAISE(ABORT, 'synthetic receipt failure'); END"))
    revoke = dict(request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'], expected_generation=1)
    assert (await rpc(bound.alice, 'revoke', revoke))['error']['message'] == 'storage_unavailable'
    assert binding_rows(bound) == before
    bound.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', ('{}', binding_key)))
    monkeypatch.setattr(rooms, 'list_rooms', bound.forbidden)
    with pytest.raises(RuntimeStoreError):
        await read_messaging_inventory_page(bound.runner, bound.event())
