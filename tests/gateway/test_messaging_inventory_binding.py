"""Native consent and independently authenticated canonical metadata reads (no services)."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway import hosted_rooms as rooms
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


class Receiver:
    """Test-owned transport; source stamping and runner authorization stay real."""
    build_source = BasePlatformAdapter.build_source

    def __init__(self, runner, config):
        self.platform, self.gateway_runner, self.config = Platform.SIGNAL, runner, config


@pytest_asyncio.fixture
async def bound(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    db = SessionDB(home / 'state.db')
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SIGNAL: PlatformConfig(enabled=True,
        extra={'allow_from': ['person-1'], 'allow_admin_from': ['person-1']})})
    runner._primary_profile_name = 'default'
    runner._profile_adapters = {}
    runner._draining = False
    adapter = Receiver(runner, runner.config.platforms[Platform.SIGNAL])
    runner.adapters = {Platform.SIGNAL: adapter}
    authority = SessionAuthority(runner, profile_id=str(home), instance_id='inert', db=db,
        epoch=begin_runtime_epoch(db, instance_id='inert'))
    runner.session_authority = authority
    runner.session_authorities = SessionAuthorities(home)
    runner.session_authorities.add(home, authority)

    def forbidden(*args, **kwargs):
        raise AssertionError('startup/execution/private-detail/legacy boundary reached')

    from tui_gateway.hosted_room_driver import HostedRoomRuntime
    from gateway import session_hosted_service
    from tui_gateway import methods_groups
    monkeypatch.setattr(HostedRoomRuntime, 'start', forbidden)
    monkeypatch.setattr(session_hosted_service, 'ensure_hosted_service', forbidden)
    monkeypatch.setattr(methods_groups, 'get_hosted_room_service', forbidden)
    monkeypatch.setattr(authority, '_schedule', forbidden)
    monkeypatch.setattr(authority, 'admit_native', forbidden)
    service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
    authority.hosted_room_service = service
    caps = ['session:read', 'session:create', 'session:control']
    def native(subject):
        return AuthorityConnection(authority, object(), dict(user_id=subject, provider='local',
            profile_id=str(home), instance_id=authority.instance_id, capabilities=caps,
            native_bootstrap=True), operator=True)
    alice, bob = native('native-alice'), native('native-bob')
    for connection, rid, name, now in [(alice, 'alice-room', 'Alice inventory', 1),
                                     (bob, 'bob-room', 'BOB PRIVATE SENTINEL', 2)]:
        service.authorize_room(connection.actor.subject, rid, create=True)
        rooms.create_room(db.db_path, room_id=rid, name=name, now=now,
            members=[dict(member_id='writer', profile='default', handle='writer')],
            authority_gateway_id='inert-gateway')
    monkeypatch.setattr(rooms, 'read_events', forbidden)
    monkeypatch.setattr(rooms, 'room_state', forbidden)
    for method in ('send', 'stop_room', 'retry_room_task', 'discard_room_task', 'approve_room_task'):
        monkeypatch.setattr(service, method, forbidden)
    # Listing must not create or inspect the shared-store identity file.
    monkeypatch.setattr(rooms, 'local_authority_gateway_id', forbidden)
    def event(**changes):
        source = adapter.build_source(chat_id='private-chat', user_id='person-1',
            thread_id=None, scope_id='scope-1', message_id='message-1')
        source.is_one_to_one = True
        for key, value in changes.items():
            setattr(source, key, value)
        return MessageEvent(text='/group list', message_type=MessageType.COMMAND,
            user_id=source.user_id, message_id='message-1', source=source)
    yield SimpleNamespace(home=home, db=db, runner=runner, authority=authority,
        service=service, adapter=adapter, alice=alice, bob=bob, native=native, event=event,
        forbidden=forbidden)
    db.close()


def recipient(**changes):
    return dict(platform='signal', user_id='person-1', chat_id='private-chat',
        thread_id=None, scope_id='scope-1', transport_profile='default',
        runtime_profile='default') | changes


def grant_params(**changes):
    return dict(request_id='grant-one', recipient=recipient(), expected_generation=0) | changes


async def rpc(connection, method='grant', params=None):
    return await connection.dispatch(dict(id=1, method='groups.messaging.read.' + method,
        params=grant_params() if params is None else params))


async def enrolled(bound):
    response = await rpc(bound.alice)
    assert 'result' in response, response
    return response['result']


def snapshot(db):
    with db._read_ctx() as conn:
        return tuple(conn.iterdump())


@pytest.mark.asyncio
async def test_native_enrollment_reads_only_consented_inventory(bound):
    receipt = await enrolled(bound)  # Semantic RED reaches the real missing-method response.
    assert receipt['active'] is True and receipt['generation'] == 1
    assert (await rpc(bound.alice))['result'] == receipt
    from gateway.session_group_messaging_read import read_messaging_inventory_page
    before = snapshot(bound.db)
    result = await read_messaging_inventory_page(bound.runner, bound.event())
    assert result == {'rooms': [{'name': 'Alice inventory', 'member_count': 1}], 'next_offset': None}
    assert snapshot(bound.db) == before
    assert bound.alice.actor.subject == 'native-alice'
    assert 'BOB PRIVATE' not in json.dumps(result)
    print('REAL_POSITIVE', json.dumps(result, sort_keys=True))


@pytest.mark.asyncio
async def test_conflicts_revoke_replay_and_regrant_are_owner_scoped(bound):
    first = await enrolled(bound)
    assert 'error' in await rpc(bound.bob)
    assert 'error' in await rpc(bound.alice, params=grant_params(owner='native-bob'))
    assert 'error' in await rpc(bound.alice, params=grant_params(recipient=recipient(user_id='other')))
    revoke = dict(request_id='revoke-one', recipient=recipient(), binding_id=first['binding_id'],
        expected_generation=first['generation'])
    assert 'error' in await rpc(bound.bob, 'revoke', revoke)
    revoked = (await rpc(bound.alice, 'revoke', revoke))['result']
    assert revoked == dict(binding_id=first['binding_id'], generation=2, active=False)
    assert (await rpc(bound.alice, 'revoke', revoke))['result'] == revoked
    assert 'error' in await rpc(bound.alice)
    from gateway.session_group_messaging_read import read_messaging_inventory_page
    with pytest.raises(RuntimeStoreError):
        await read_messaging_inventory_page(bound.runner, bound.event())
    second = (await rpc(bound.alice, params=grant_params(request_id='grant-two',
        expected_generation=revoked['generation'])))['result']
    assert second['generation'] > revoked['generation']
    assert second['binding_id'] != first['binding_id']
    assert 'error' in await rpc(bound.alice, 'revoke', revoke)
    assert (await read_messaging_inventory_page(bound.runner, bound.event()))['rooms']


@pytest.mark.asyncio
async def test_filtered_empty_page_preserves_real_cursor(bound):
    await enrolled(bound)
    from gateway.session_group_messaging_read import read_messaging_inventory_page
    before = snapshot(bound.db)
    page = await read_messaging_inventory_page(bound.runner, bound.event(), limit=1)
    assert page == {'rooms': [], 'next_offset': 1}
    second = await read_messaging_inventory_page(bound.runner, bound.event(), limit=1, offset=page['next_offset'])
    assert second == {'rooms': [{'name': 'Alice inventory', 'member_count': 1}], 'next_offset': 2}
    assert await read_messaging_inventory_page(bound.runner, bound.event(), limit=1, offset=2) == {
        'rooms': [], 'next_offset': None}
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
async def test_bounded_projection_and_exact_recipient_with_real_thread(bound):
    request = grant_params(recipient=recipient(thread_id='real-thread', scope_id='other-scope'))
    assert 'result' in await rpc(bound.alice, params=request)
    long_name = 'Safe\n' + 'name ' * 30
    bound.db._execute_write(lambda conn: conn.execute('UPDATE hosted_rooms SET name=? WHERE room_id=?',
        (long_name, 'alice-room')))
    from gateway.session_group_messaging_read import read_messaging_inventory_page
    before = snapshot(bound.db)
    result = await read_messaging_inventory_page(bound.runner, bound.event(
        thread_id='real-thread', scope_id='other-scope'))
    room, = result['rooms']
    assert set(result) == {'rooms', 'next_offset'} and set(room) == {'name', 'member_count'}
    assert len(room['name']) <= 72 and '\n' not in room['name']
    assert room['member_count'] == 1
    assert snapshot(bound.db) == before


@pytest.mark.asyncio
async def test_competing_native_owners_cannot_combine_bindings(bound):
    responses = await asyncio.gather(rpc(bound.alice), rpc(bound.bob))
    winners = [i for i, response in enumerate(responses) if 'result' in response]
    assert len(winners) == 1
    from gateway.session_group_messaging_read import read_messaging_inventory_page
    result = await read_messaging_inventory_page(bound.runner, bound.event())
    assert result['rooms'] == [{'name': ['Alice inventory', 'BOB PRIVATE SENTINEL'][winners[0]],
                               'member_count': 1}]
