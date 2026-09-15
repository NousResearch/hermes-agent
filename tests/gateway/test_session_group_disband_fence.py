"""Canonical Disband ordering with temporary SQL and inert cleanup callbacks."""

import threading
from functools import partial
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_link_records as links
from gateway import hosted_rooms as rooms
from gateway.session_group_controls import dispatch_group_control
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(rooms, 'local_authority_gateway_id', lambda: 'home')
    db = SessionDB(tmp_path / 'state.db')
    rooms.create_room(db.db_path, room_id='room', name='Workshop',
        members=[{'profile': 'ops'}], authority_gateway_id='home')
    message(db, 'seed')
    actor = SimpleNamespace(subject='alice', profile_id=str(tmp_path),
        capabilities=frozenset({'session:control'}))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, hosted_room_service=None)
    try:
        yield SimpleNamespace(authority=authority, actor=actor)
    finally:
        db.close()


def message(db, event_id):
    return rooms.append_event(db.db_path, room_id='room', event_id=event_id,
        kind='message.user', actor={'kind': 'user', 'id': 'alice'},
        payload={'text': event_id}, authority_gateway_id='home', authority_epoch=1)


def inert_service(db, *, stopping=False):
    def authorize(subject, room_id, **kwargs):
        if (subject, room_id) != ('alice', 'room'):
            raise RuntimeStoreError('permission_denied')
        return True
    service = SimpleNamespace(db_path=db.db_path, _policy_lock=threading.RLock(),
        authorize_room=authorize,
        runtime=SimpleNamespace(status=lambda: {'running': True, 'stopping': stopping}))
    # Use the actual data-only source fence; no HostedRoomService is started.
    service.begin_room_disband = partial(HostedRoomService.begin_room_disband, service)
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['stop', 'revoke', None])
async def test_disband_fences_before_inert_cleanup(owner, failure):
    db, phases = owner.authority.db, []
    service = inert_service(db)

    def stopped(room_id, **kwargs):
        assert (room_id, kwargs) == ('room', {
            'cancel_id': 'close-request', 'require_acknowledged': True})
        phases.append(('stop', links.room_link_retirement_started(db.db_path, room_id='room')))
        if failure == 'stop':
            raise RuntimeError('inert Stop refused')
        return 0

    def revoked(room_id):
        phases.append(('revoke', links.room_link_retirement_started(db.db_path, room_id=room_id)))
        # Real route cleanup also starts retirement; this must not be the first fence.
        links.begin_room_link_retirement(db.db_path, room_id=room_id,
            authority_gateway_id='home', authority_epoch=1)
        if failure == 'revoke':
            raise RuntimeError('inert revocation refused')
        links.complete_room_link_retirement(db.db_path, room_id=room_id,
            authority_gateway_id='home', authority_epoch=1)
        return 0

    service.stop_room, service.revoke_room_routes = stopped, revoked
    owner.authority.hosted_room_service = service
    params = {'room_id': 'room', 'cancel_id': 'close-request'}
    if failure:
        with pytest.raises(RuntimeError, match='inert'):
            await dispatch_group_control(owner, 'groups.disband', params)
    else:
        result = await dispatch_group_control(owner, 'groups.disband', params)
        assert result['tombstone']['idempotent'] is False

    assert phases[0] == ('stop', True)
    assert links.room_link_retirement_started(db.db_path, room_id='room')
    state = rooms.room_state(db.db_path, room_id='room', include_disbanded=True)
    if failure:
        assert state.get('disbanded_at') is None
        with pytest.raises(rooms.HostedRoomError, match='being disbanded'):
            message(db, 'new-input')
        with pytest.raises(driver.RoomUnavailableError, match='being disbanded'):
            driver.admit_task(db.db_path, driver.TaskIdentity('room', 'new', 'thread', 'turn'),
                payload={'target_profile': 'ops', 'prompt': 'New work', 'source_event_seq': 1},
                clock=lambda: 100)
        assert message(db, 'seed')['idempotent']
    else:
        assert state['disbanded_at'] is not None
        assert phases == [('stop', True), ('revoke', True)]
        repeated = await dispatch_group_control(owner, 'groups.disband', params)
        assert repeated['tombstone']['idempotent']
        assert phases == [('stop', True), ('revoke', True)]


@pytest.mark.asyncio
@pytest.mark.parametrize('outstanding', ['task', 'route', 'empty'])
@pytest.mark.parametrize('coordinator', ['missing', 'stopping'])
async def test_unavailable_disband_preserves_outstanding_state(owner, outstanding, coordinator):
    from gateway import hosted_room_links as stored_links
    from gateway.hosted_room_peer import GatewayRoomCatalog, catalog_mapping

    db = owner.authority.db
    if coordinator == 'stopping':
        owner.authority.hosted_room_service = inert_service(db, stopping=True)
    if outstanding == 'task':
        driver.admit_task(db.db_path, driver.TaskIdentity('room', 'accepted', 'thread', 'turn'),
            payload={'target_profile': 'ops', 'prompt': 'Accepted work', 'source_event_seq': 1},
            clock=lambda: 100)
    elif outstanding == 'route':
        catalog = GatewayRoomCatalog.from_mapping(catalog_mapping(
            installation_id='peer', target_profile='ops', persistent_process=True))
        route = stored_links.make_stored_link(room_id='room', member_id='ops',
            target_url='http://127.0.0.1:9999', target_profile='ops', grant='retained-grant',
            catalog=catalog, cancellation_scope_id='cancel', trace_id='trace')
        stored_links.save_room_link(db.db_path, route)
    tasks_before = driver.list_tasks(db.db_path, room_id='room')
    routes_before = links.list_room_link_records(db.db_path, room_id='room')

    if outstanding == 'empty':
        result = await dispatch_group_control(owner, 'groups.disband', {'room_id': 'room'})
        assert result['tombstone']['disbanded_at'] is not None
    else:
        with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
            await dispatch_group_control(owner, 'groups.disband', {'room_id': 'room'})
        assert rooms.room_state(db.db_path, room_id='room').get('disbanded_at') is None
    assert links.room_link_retirement_started(db.db_path, room_id='room')
    assert driver.list_tasks(db.db_path, room_id='room') == tasks_before
    assert links.list_room_link_records(db.db_path, room_id='room') == routes_before
