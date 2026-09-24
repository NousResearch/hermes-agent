"""Native writer and messaging continuation fences for exact room-read consent."""
import asyncio
import json
import threading

import pytest

from gateway import hosted_rooms as rooms
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_messaging_inventory_binding import (
    Receiver,
    bound,
    enrolled,
    grant_params,
    recipient,
    rpc,
)
from tests.gateway.test_messaging_room_read_binding import (
    add_room,
    room_grant_params,
    room_revoke_params,
    room_rows,
    room_rpc,
)

__all__ = ['bound']


async def granted(bound):
    inventory = await enrolled(bound)
    response = await room_rpc(bound.alice, params=room_grant_params(inventory))
    assert 'result' in response, response
    return inventory, response['result']


def mutate_inventory(bound, write):
    def change(conn):
        row = conn.execute(
            "SELECT key,value FROM state_meta "
            "WHERE key LIKE 'gateway.messaging.read.v1.binding.%'"
        ).fetchone()
        value = json.loads(row[1])
        value['active'] = False
        value['generation'] += 1
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                     (json.dumps(value, sort_keys=True, separators=(',', ':')), row[0]))
    write(change)


def drift_native_room(bound, drift, write):
    authority, connection = bound.authority, bound.alice
    if drift == 'actor':
        from dataclasses import replace
        connection.actor = replace(connection.actor, subject='native-bob')
    elif drift == 'transport':
        connection.transport = object()
    elif drift == 'native_binding':
        authority._native_legacy_transports.clear()
    elif drift == 'closed_transport':
        authority.events.clear()
    elif drift == 'registry':
        from gateway.session_authorities import SessionAuthorities
        bound.runner.session_authorities = SessionAuthorities(bound.home)
        bound.runner.session_authorities.add(bound.home, authority)
    elif drift == 'epoch':
        authority.epoch += 1
    elif drift == 'stored_epoch':
        write(lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
    elif drift == 'instance':
        authority.instance_id = 'replacement'
    elif drift == 'stored_instance':
        write(lambda conn: conn.execute(
            "UPDATE runtime_epoch SET instance_id='replacement'"))
    elif drift == 'profile':
        authority.profile_id = str(bound.home / 'other')
    elif drift == 'db':
        authority.db = object()
    elif drift == 'native':
        connection.native_owner = False
    elif drift == 'service':
        authority.hosted_room_service = None
    elif drift == 'owner':
        write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=?',
            ('native-bob', 'gateway.hosted.owner.v1:alice-room')))
    elif drift == 'inventory':
        mutate_inventory(bound, write)


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['grant', 'revoke'])
@pytest.mark.parametrize('drift', [
    'actor', 'transport', 'native_binding', 'closed_transport', 'registry', 'epoch',
    'stored_epoch', 'instance', 'stored_instance', 'profile', 'db', 'native',
    'service', 'owner', 'inventory',
])
async def test_native_room_write_rechecks_every_owner_boundary(
        bound, monkeypatch, method, drift):
    inventory = await enrolled(bound)
    params = room_grant_params(inventory)
    if method == 'revoke':
        grant = (await room_rpc(bound.alice, params=params))['result']
        params = room_revoke_params(inventory, grant)
    before = room_rows(bound)
    entered, resume = threading.Event(), threading.Event()
    original_write = bound.db._execute_write

    def held(write):
        entered.set()
        assert resume.wait(10)
        return original_write(write)

    monkeypatch.setattr(bound.db, '_execute_write', held)
    pending = asyncio.create_task(room_rpc(bound.alice, method, params))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        drift_native_room(bound, drift, original_write)
    finally:
        resume.set()
    response = await asyncio.wait_for(pending, 10)
    assert 'error' in response, response
    assert room_rows(bound) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', [
    'list_revoke', 'room_revoke', 'room_aba', 'adapter', 'receiver_profile',
    'event_user', 'policy', 'registry', 'epoch', 'stored_epoch', 'instance',
    'stored_instance', 'service', 'owner',
])
async def test_room_attestation_rechecks_consent_receiver_runtime_and_owner(
        bound, drift):
    from gateway.session_group_messaging_read import _attest_room_read

    inventory, grant = await granted(bound)
    event = bound.event()
    context = _attest_room_read(bound.runner, event, grant['room_ref'])
    assert context.require_current()['active'] is True

    if drift == 'list_revoke':
        response = await rpc(bound.alice, 'revoke', dict(
            request_id='list-revoke',
            recipient=recipient(),
            binding_id=inventory['binding_id'],
            expected_generation=inventory['generation'],
        ))
        assert 'result' in response
    elif drift in {'room_revoke', 'room_aba'}:
        revoked = (await room_rpc(
            bound.alice, 'revoke', room_revoke_params(inventory, grant)))['result']
        if drift == 'room_aba':
            response = await room_rpc(bound.alice, params=room_grant_params(
                inventory,
                request_id='room-regrant',
                expected_generation=revoked['generation'],
            ))
            assert response['result']['room_ref'] != grant['room_ref']
    elif drift == 'adapter':
        bound.runner.adapters[bound.adapter.platform] = Receiver(
            bound.runner, bound.adapter.config)
    elif drift == 'receiver_profile':
        bound.runner._primary_profile_name = 'other'
    elif drift == 'event_user':
        event.source.user_id = 'other'
    elif drift == 'policy':
        bound.adapter.config.extra['allow_from'] = ['other']
    elif drift == 'registry':
        from gateway.session_authorities import SessionAuthorities
        bound.runner.session_authorities = SessionAuthorities(bound.home)
        bound.runner.session_authorities.add(bound.home, bound.authority)
    elif drift == 'epoch':
        bound.authority.epoch += 1
    elif drift == 'stored_epoch':
        bound.db._execute_write(
            lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
    elif drift == 'instance':
        bound.authority.instance_id = 'replacement'
    elif drift == 'stored_instance':
        bound.db._execute_write(lambda conn: conn.execute(
            "UPDATE runtime_epoch SET instance_id='replacement'"))
    elif drift == 'service':
        bound.authority.hosted_room_service = None
    elif drift == 'owner':
        bound.db._execute_write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=?',
            ('native-bob', 'gateway.hosted.owner.v1:alice-room')))

    with pytest.raises(RuntimeStoreError):
        context.require_current()


@pytest.mark.asyncio
async def test_room_context_is_room_method_and_generation_exact(bound):
    from dataclasses import replace
    from gateway.session_group_messaging_read import _attest_room_read

    inventory, grant = await granted(bound)
    add_room(bound, 'alice-other')
    context = _attest_room_read(bound.runner, bound.event(), grant['room_ref'])
    assert context.require_current(method='groups.log', room_id='alice-room')['generation'] == 1
    for method, room_id in [
        ('groups.list', 'alice-room'),
        ('groups.send', 'alice-room'),
        ('groups.state', 'alice-other'),
    ]:
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            context.require_current(method=method, room_id=room_id)
    with pytest.raises(RuntimeStoreError):
        replace(context, generation=2).require_current()
    with pytest.raises(RuntimeStoreError):
        replace(context, owner='native-bob').require_current()
    assert context.actor.capabilities == frozenset({'session:read'})
    assert inventory['active'] is True


@pytest.mark.asyncio
async def test_owner_drift_cannot_be_repaired_by_generation_or_fresh_request(bound):
    inventory, first = await granted(bound)
    revoked = (await room_rpc(
        bound.alice, 'revoke', room_revoke_params(inventory, first)))['result']
    bound.db._execute_write(lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key=?',
        ('gateway.hosted.owner.v1:alice-room',)))
    before = room_rows(bound)
    response = await room_rpc(bound.alice, params=room_grant_params(
        inventory,
        request_id='fresh-generation-without-owner',
        expected_generation=revoked['generation'],
    ))
    assert response['error']['message'] in {'permission_denied', 'invalid_params'}
    assert room_rows(bound) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('corruption', ['binding', 'reference', 'counter'])
async def test_room_ref_resolution_is_read_only_and_corruption_fails_closed(
        bound, corruption):
    from gateway.session_group_messaging_read import _attest_room_read

    _, grant = await granted(bound)
    before = tuple(bound.db._conn.iterdump())
    context = _attest_room_read(bound.runner, bound.event(), grant['room_ref'])
    assert context.require_current()['binding_id'] == grant['binding_id']
    assert tuple(bound.db._conn.iterdump()) == before

    keys = {part: key for key, _ in room_rows(bound) for part in (
        'binding' if '.room-binding.' in key else
        'reference' if '.room-reference.' in key else
        'counter' if '.room-ref-counter.' in key else '',
    ) if part}
    if corruption == 'binding':
        bound.db._execute_write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=?', ('{}', keys[corruption])))
    else:
        bound.db._execute_write(lambda conn: conn.execute(
            'DELETE FROM state_meta WHERE key=?', (keys[corruption],)))
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _attest_room_read(bound.runner, bound.event(), grant['room_ref'])


@pytest.mark.asyncio
async def test_room_revoke_rolls_back_if_receipt_insert_fails(bound):
    inventory, grant = await granted(bound)
    before = room_rows(bound)
    bound.db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_room_revoke_receipt BEFORE INSERT ON state_meta "
        "WHEN NEW.key LIKE 'gateway.messaging.read.v1.room-request.%' "
        "BEGIN SELECT RAISE(ABORT, 'synthetic room revoke receipt failure'); END"
    ))
    response = await room_rpc(
        bound.alice, 'revoke', room_revoke_params(inventory, grant))
    assert response['error']['message'] == 'storage_unavailable'
    assert room_rows(bound) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('regrant', [False, True])
async def test_projection_rechecks_room_grant_after_final_inventory_wait(bound, monkeypatch, regrant):
    from gateway.session_group_messaging_read import (
        _InventoryRead, _attest_inventory, read_messaging_inventory_page,
    )

    inventory, grant = await granted(bound)
    context = _attest_inventory(bound.runner, bound.event())
    page = {'rooms': [room for room in rooms.list_rooms(bound.db.db_path)
                      if room['room_id'] == 'alice-room'], 'next_offset': None}
    assert len(page['rooms']) == 1
    entered, resume = threading.Event(), threading.Event()
    original = _InventoryRead.require_current
    calls = 0

    def held(self):
        nonlocal calls
        if self is context:
            calls += 1
            if calls == 2:
                entered.set()
                assert resume.wait(10)
        return original(self)

    monkeypatch.setattr(_InventoryRead, 'require_current', held)
    pending = asyncio.create_task(asyncio.to_thread(context.project, page))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        revoked = await room_rpc(bound.alice, 'revoke', room_revoke_params(inventory, grant))
        assert 'result' in revoked
        replacement = None
        if regrant:
            response = await room_rpc(bound.alice, params=room_grant_params(
                inventory, request_id='projection-regrant',
                expected_generation=revoked['result']['generation']))
            replacement = response['result']['room_ref']
            assert replacement != grant['room_ref']
    finally:
        resume.set()
    with pytest.raises(RuntimeStoreError, match='messaging_room_read_stale'):
        await asyncio.wait_for(pending, 10)
    assert calls == 2
    fresh = await read_messaging_inventory_page(bound.runner, bound.event())
    assert fresh['rooms'][0].get('room_ref') == replacement
