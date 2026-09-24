"""Exact native room-read consent and recipient-local reference contracts."""
import json

import pytest

from gateway import hosted_rooms as rooms
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_messaging_inventory_binding import (
    bound,
    enrolled,
    grant_params,
    recipient,
    rpc,
    snapshot,
)

__all__ = [
    'bound', 'room_grant_params', 'room_revoke_params', 'room_rpc', 'room_rows',
]


def room_grant_params(inventory, **changes):
    return dict(
        request_id='room-grant-one',
        recipient=recipient(),
        inventory_binding_id=inventory['binding_id'],
        room_id='alice-room',
        expected_generation=0,
    ) | changes


def room_revoke_params(inventory, grant, **changes):
    return dict(
        request_id='room-revoke-one',
        recipient=recipient(),
        inventory_binding_id=inventory['binding_id'],
        room_id='alice-room',
        expected_generation=grant['generation'],
        binding_id=grant['binding_id'],
    ) | changes


async def room_rpc(connection, method='grant', params=None):
    assert params is not None
    return await connection.dispatch(dict(
        id=2,
        method='groups.messaging.room.read.' + method,
        params=params,
    ))


def room_rows(bound):
    with bound.db._read_ctx() as conn:
        return tuple(tuple(row) for row in conn.execute(
            "SELECT key,value FROM state_meta "
            "WHERE key LIKE 'gateway.messaging.read.v1.room-%' ORDER BY key"
        ))


def add_room(bound, room_id, *, owner=None, now=10):
    owner = owner or bound.alice
    bound.service.authorize_room(owner.actor.subject, room_id, create=True)
    rooms.create_room(
        bound.db.db_path,
        room_id=room_id,
        name='Room ' + room_id,
        now=now,
        members=[dict(member_id='writer', profile='default', handle='writer')],
        authority_gateway_id='inert-gateway',
    )


@pytest.mark.asyncio
async def test_registered_native_room_grant_read_record_revoke_and_regrant(bound):
    from gateway.session_group_messaging_read import read_messaging_inventory_page

    inventory = await enrolled(bound)
    assert await read_messaging_inventory_page(bound.runner, bound.event()) == {
        'rooms': [{'name': 'Alice inventory', 'member_count': 1}],
        'next_offset': None,
    }

    response = await room_rpc(bound.alice, params=room_grant_params(inventory))
    assert 'result' in response, response  # Semantic RED: real registered dispatch is missing.
    from gateway.session_group_messaging_read import _attest_room_read
    grant = response['result']
    assert grant == {
        'binding_id': grant['binding_id'],
        'generation': 1,
        'active': True,
        'room_ref': 1,
    }
    assert grant['binding_id'].startswith('mrr-')
    # A separate detail grant must not break the existing list consumer.
    from gateway.hosted_room_messaging_runtime import render_inventory_page
    from gateway.session_group_messaging_read import _attest_inventory
    listing = await render_inventory_page(_attest_inventory(bound.runner, bound.event()), 1, '/')
    assert 'Alice inventory' in listing
    assert (await room_rpc(bound.alice, params=room_grant_params(inventory)))['result'] == grant

    rows = room_rows(bound)
    records = [json.loads(value) for key, value in rows if '.room-binding.' in key]
    assert records == [{
        'version': 1,
        'recipient': recipient(),
        'profile_id': str(bound.home),
        'owner': 'native-alice',
        'room_id': 'alice-room',
        'scope': ['groups.log', 'groups.state'],
        'inventory_binding_id': inventory['binding_id'],
        'binding_id': grant['binding_id'],
        'generation': 1,
        'active': True,
        'room_ref': 1,
    }]
    assert any('.room-request.' in key for key, _ in rows)
    assert any('.room-ref-counter.' in key for key, _ in rows)

    page = await read_messaging_inventory_page(bound.runner, bound.event())
    assert page == {
        'rooms': [{'name': 'Alice inventory', 'member_count': 1, 'room_ref': 1}],
        'next_offset': None,
    }
    context = _attest_room_read(bound.runner, bound.event(), 1)
    state = context.require_current(method='groups.state', room_id='alice-room')
    assert state['binding_id'] == grant['binding_id']
    assert context.actor.subject.startswith('messaging:')
    assert context.actor.subject != context.owner == bound.alice.actor.subject
    assert context.actor.capabilities == frozenset({'session:read'})
    assert context.methods == frozenset({'groups.log', 'groups.state'})

    revoke_params = room_revoke_params(inventory, grant)
    revoked = (await room_rpc(bound.alice, 'revoke', revoke_params))['result']
    assert revoked == {
        'binding_id': grant['binding_id'],
        'generation': 2,
        'active': False,
        'room_ref': 1,
    }
    assert (await room_rpc(bound.alice, 'revoke', revoke_params))['result'] == revoked
    with pytest.raises(RuntimeStoreError, match='messaging_room_read_stale'):
        context.require_current()
    assert await read_messaging_inventory_page(bound.runner, bound.event()) == {
        'rooms': [{'name': 'Alice inventory', 'member_count': 1}],
        'next_offset': None,
    }

    second_params = room_grant_params(
        inventory,
        request_id='room-grant-two',
        expected_generation=revoked['generation'],
    )
    second = (await room_rpc(bound.alice, params=second_params))['result']
    assert second['generation'] == 3
    assert second['binding_id'] != grant['binding_id']
    assert second['room_ref'] == 2
    with pytest.raises(RuntimeStoreError, match='messaging_room_read_stale'):
        _attest_room_read(bound.runner, bound.event(), grant['room_ref'])
    assert _attest_room_read(bound.runner, bound.event(), second['room_ref']).room_id == 'alice-room'
    print('ROOM_READ_RECEIPTS', json.dumps(
        {'grant': grant, 'revoke': revoked, 'regrant': second}, sort_keys=True))


@pytest.mark.asyncio
async def test_inventory_consent_alone_remains_exact_old_shape_and_cannot_attest_room(bound):
    from gateway.session_group_messaging_read import (
        _attest_room_read,
        read_messaging_inventory_page,
    )

    await enrolled(bound)
    before = snapshot(bound.db)
    assert await read_messaging_inventory_page(bound.runner, bound.event()) == {
        'rooms': [{'name': 'Alice inventory', 'member_count': 1}],
        'next_offset': None,
    }
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        _attest_room_read(bound.runner, bound.event(), 1)
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
async def test_room_grant_requires_exact_owner_room_recipient_inventory_and_service(bound):
    inventory = await enrolled(bound)
    good = room_grant_params(inventory)
    before = room_rows(bound)

    attempts = [
        (bound.bob, good),
        (bound.alice, good | {'room_id': 'bob-room'}),
        (bound.alice, good | {'room_id': 'missing-room'}),
        (bound.alice, good | {'recipient': recipient(user_id='other')}),
        (bound.alice, good | {'inventory_binding_id': 'mr-' + '0' * 32}),
        (bound.alice, good | {'expected_generation': 1}),
        (bound.alice, good | {'room_id': 'alice room'}),
    ]
    for connection, params in attempts:
        assert 'error' in await room_rpc(connection, params=params)
        assert room_rows(bound) == before

    service = bound.authority.hosted_room_service
    bound.authority.hosted_room_service = None
    try:
        assert 'error' in await room_rpc(bound.alice, params=good)
        assert room_rows(bound) == before
    finally:
        bound.authority.hosted_room_service = service


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'expected_generation': True},
    {'expected_generation': -1},
    {'expected_generation': 2**63},
    {'request_id': ''},
    {'request_id': 'bad\n'},
    {'recipient': recipient(runtime_profile='other')},
    {'inventory_binding_id': 'bad'},
    {'room_id': ''},
    {'binding_id': 'mrr-' + '0' * 32},
    {'unexpected': True},
])
async def test_room_grant_body_is_fixed_and_writes_nothing(bound, changes):
    inventory = await enrolled(bound)
    before = room_rows(bound)
    response = await room_rpc(bound.alice, params=room_grant_params(inventory, **changes))
    assert 'error' in response
    assert room_rows(bound) == before


@pytest.mark.asyncio
async def test_room_generation_binding_and_request_replays_are_exact(bound):
    inventory = await enrolled(bound)
    params = room_grant_params(inventory)
    first = (await room_rpc(bound.alice, params=params))['result']
    assert (await room_rpc(bound.alice, params=params))['result'] == first

    add_room(bound, 'alice-other')
    conflict = params | {'room_id': 'alice-other'}
    assert (await room_rpc(bound.alice, params=conflict))['error']['message'] == 'admission_conflict'
    assert 'error' in await room_rpc(bound.alice, params=room_grant_params(
        inventory, request_id='second-active'))
    assert 'error' in await room_rpc(bound.alice, 'revoke', room_revoke_params(
        inventory, first, binding_id='mrr-' + '0' * 32))
    assert 'error' in await room_rpc(bound.alice, 'revoke', room_revoke_params(
        inventory, first, expected_generation=0))

    revoke = room_revoke_params(inventory, first)
    revoked = (await room_rpc(bound.alice, 'revoke', revoke))['result']
    assert (await room_rpc(bound.alice, 'revoke', revoke))['result'] == revoked
    assert 'error' in await room_rpc(bound.alice, params=params)


@pytest.mark.asyncio
async def test_inventory_regrant_invalidates_room_grant_without_recycling_reference(bound):
    from gateway.session_group_messaging_read import (
        _attest_room_read,
        read_messaging_inventory_page,
    )

    inventory = await enrolled(bound)
    first = (await room_rpc(bound.alice, params=room_grant_params(inventory)))['result']
    list_revoke = dict(
        request_id='list-revoke', recipient=recipient(), binding_id=inventory['binding_id'],
        expected_generation=inventory['generation'],
    )
    assert 'result' in await rpc(bound.alice, 'revoke', list_revoke)
    replacement_inventory = (await rpc(bound.alice, params=grant_params(
        request_id='list-regrant', expected_generation=2)))['result']

    assert await read_messaging_inventory_page(bound.runner, bound.event()) == {
        'rooms': [{'name': 'Alice inventory', 'member_count': 1}],
        'next_offset': None,
    }
    with pytest.raises(RuntimeStoreError, match='messaging_room_read_stale'):
        _attest_room_read(bound.runner, bound.event(), first['room_ref'])
    replacement = (await room_rpc(bound.alice, params=room_grant_params(
        replacement_inventory,
        request_id='room-after-list-regrant',
        expected_generation=first['generation'],
    )))['result']
    assert replacement['binding_id'] != first['binding_id']
    assert replacement['room_ref'] > first['room_ref']


@pytest.mark.asyncio
async def test_references_are_recipient_local_room_exact_and_survive_room_deletion(bound):
    from gateway.session_group_messaging_read import (
        _attest_room_read,
        read_messaging_inventory_page,
    )

    first_inventory = await enrolled(bound)
    first = (await room_rpc(bound.alice, params=room_grant_params(first_inventory)))['result']
    add_room(bound, 'alice-other')
    other = (await room_rpc(bound.alice, params=room_grant_params(
        first_inventory,
        request_id='other-room-grant',
        room_id='alice-other',
    )))['result']
    assert other['room_ref'] == first['room_ref'] + 1

    page = await read_messaging_inventory_page(bound.runner, bound.event())
    refs = {row['name']: row.get('room_ref') for row in page['rooms']}
    assert refs['Alice inventory'] == first['room_ref']
    assert refs['Room alice-other'] == other['room_ref']
    assert _attest_room_read(bound.runner, bound.event(), first['room_ref']).room_id == 'alice-room'
    assert _attest_room_read(bound.runner, bound.event(), other['room_ref']).room_id == 'alice-other'

    second_recipient = recipient(user_id='person-2', chat_id='private-two', scope_id='scope-2')
    second_inventory = (await rpc(bound.alice, params=grant_params(
        request_id='list-person-two', recipient=second_recipient)))['result']
    second_grant = (await room_rpc(bound.alice, params=room_grant_params(
        second_inventory,
        request_id='room-person-two',
        recipient=second_recipient,
    )))['result']
    assert second_grant['room_ref'] == 1
    bound.adapter.config.extra['allow_from'].append('person-2')
    bound.adapter.config.extra['allow_admin_from'].append('person-2')
    second_event = bound.event(user_id='person-2', chat_id='private-two', scope_id='scope-2')
    assert _attest_room_read(bound.runner, second_event, 1).actor.subject != (
        _attest_room_read(bound.runner, bound.event(), 1).actor.subject)

    transient_revoke = room_revoke_params(first_inventory, other,
        request_id='delete-room-revoke', room_id='alice-other')
    assert 'result' in await room_rpc(bound.alice, 'revoke', transient_revoke)
    bound.db._execute_write(lambda conn: (
        conn.execute('DELETE FROM hosted_rooms WHERE room_id=?', ('alice-other',)),
        conn.execute('DELETE FROM state_meta WHERE key=?',
                     ('gateway.hosted.owner.v1:alice-other',)),
    ))
    add_room(bound, 'replacement-room', now=20)
    replacement = (await room_rpc(bound.alice, params=room_grant_params(
        first_inventory,
        request_id='replacement-room-grant',
        room_id='replacement-room',
    )))['result']
    assert replacement['room_ref'] > other['room_ref']


@pytest.mark.asyncio
async def test_room_capacity_reserves_revoke_and_transaction_rollback(bound, monkeypatch):
    from gateway import session_group_messaging_read as binding

    inventory = await enrolled(bound)
    monkeypatch.setattr(binding, '_MAX_ROOM_REQUESTS', 1)
    before = room_rows(bound)
    assert 'error' in await room_rpc(bound.alice, params=room_grant_params(inventory))
    assert room_rows(bound) == before

    monkeypatch.setattr(binding, '_MAX_ROOM_REQUESTS', 2)
    grant = (await room_rpc(bound.alice, params=room_grant_params(inventory)))['result']
    assert 'result' in await room_rpc(bound.alice, 'revoke', room_revoke_params(inventory, grant))
    add_room(bound, 'capacity-room')
    assert 'error' in await room_rpc(bound.alice, params=room_grant_params(
        inventory, request_id='capacity-new', room_id='capacity-room'))

    # A receipt failure rolls back the state row and recipient-local counter together.
    other_recipient = recipient(user_id='rollback-person')
    other_inventory = (await rpc(bound.alice, params=grant_params(
        request_id='rollback-list', recipient=other_recipient)))['result']
    monkeypatch.setattr(binding, '_MAX_ROOM_REQUESTS', 20)
    bound.db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_room_receipt BEFORE INSERT ON state_meta "
        "WHEN NEW.key LIKE 'gateway.messaging.read.v1.room-request.%' "
        "BEGIN SELECT RAISE(ABORT, 'synthetic room receipt failure'); END"
    ))
    before = room_rows(bound)
    failed = await room_rpc(bound.alice, params=room_grant_params(
        other_inventory,
        request_id='rollback-room',
        recipient=other_recipient,
    ))
    assert failed['error']['message'] == 'storage_unavailable'
    assert room_rows(bound) == before
