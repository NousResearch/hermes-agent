"""Disband held by a turn whose member profile is no longer configured and whose home is unknown
names the way out (re-add the profile), on both RPC surfaces, with a stable reason, and following
it unblocks Disband."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks
from gateway import hosted_rooms
from gateway.session_controls import AuthorityConnection
from persistence.gateway.hosted_rooms import RoomStopPendingError
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.test_hosted_room_unserved_member_stop import deferred as deferred_fixture
from tui_gateway import methods_groups, methods_groups_forward, server

deferred = deferred_fixture


@pytest.mark.parametrize('surface,code', [('canonical', 4001), ('standalone', 4113)])
def test_unknown_member_home_refusal_says_how_to_unblock(tmp_path, deferred, monkeypatch, surface, code):
    service, db, identity, roster = deferred(tmp_path.resolve() / 'custom' / 'two')
    raw_holder(db)._execute_write(  # as left by a gateway that never recorded member homes
        lambda conn: conn.execute("DELETE FROM state_meta WHERE key LIKE 'gateway.hosted.member_homes.v1:%'"))
    del roster['two']
    service.runtime.status = lambda: {'running': True, 'stopping': False}
    frame = {'id': 1, 'method': 'groups.disband', 'params': {'room_id': 'room'}}
    if surface == 'canonical':
        authority = service.authority
        authority.hosted_room_service, authority.events, authority.sessions = service, {}, {}
        authority.instance_id = 'test'
        owner = AuthorityConnection(authority, object(), {'user_id': 'alice'}, operator=True)
        raw_holder(db)._execute_write(  # the room belongs to this connection's subject
            lambda conn: conn.execute("UPDATE state_meta SET value=? WHERE key='gateway.hosted.owner.v1:room'",
                                      (owner.actor.subject,)))
        reply = asyncio.run(owner.dispatch(frame))
    else:
        monkeypatch.setattr(methods_groups, '_service', service)
        monkeypatch.setattr(methods_groups_forward, '_discover', lambda home: SimpleNamespace(state='absent'))
        reply = server.handle_request(frame)
    error = reply['error']
    assert error['code'] == code, reply
    assert error['data']['reason'] == 'member_home_unknown', reply
    assert "'two'" in error['message'] and 'hosted_rooms.profiles' in error['message'], error
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'deferred'
    assert hosted_rooms.room_state(db, placement='profile', room_id='room').get('disbanded_at') is None


def test_a_known_home_keeps_the_generic_stop_pending_refusal(tmp_path, deferred):
    service, _db, _identity, _roster = deferred(tmp_path.resolve() / 'profiles' / 'two')
    with pytest.raises(RoomStopPendingError) as refused:
        service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)
    assert refused.value.reason == 'room_stop_pending'


def test_re_adding_the_profile_unblocks_disband(tmp_path, deferred):
    """The advice works: re-added, the profile's home is known (Disband now waits on its daemon's
    Stop, recording that home); once its daemon is gone, Disband settles the turn."""
    home = tmp_path.resolve() / 'custom' / 'two'
    service, db, identity, roster = deferred(home)
    raw_holder(db)._execute_write(
        lambda conn: conn.execute("DELETE FROM state_meta WHERE key LIKE 'gateway.hosted.member_homes.v1:%'"))
    del roster['two']
    with pytest.raises(RoomStopPendingError) as refused:
        service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)
    assert refused.value.reason == 'member_home_unknown'
    roster['two'] = home  # re-added to hosted_rooms.profiles
    with pytest.raises(RoomStopPendingError) as refused:
        service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)
    assert refused.value.reason == 'room_stop_pending'  # served: only its daemon's Stop decides
    del roster['two']  # unserved again, but its home is now recorded and no daemon runs there
    service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'cancelled'
