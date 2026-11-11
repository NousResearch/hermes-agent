"""A turn deferred behind the unconfirmed Stop of a local member another profile's daemon serves,
once that profile is no longer served on this host: its daemon never confirms anything again, so
Stop and Disband settle the turn once nothing on this host can still run it, and only then. A
daemon that is alive (it holds its home's gateway lock) but does not answer keeps the turn."""
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks
from gateway.runtime_ownership import ProfileOwnership
from persistence.gateway.hosted_rooms import HostedTaskIdentity, RoomStopPendingError
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture
def deferred(tmp_path, monkeypatch):
    """A room on this home whose member 'two' is served by its own daemon at *member_home*, with
    a turn deferred because that daemon never confirmed its Stop. Yields a factory."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    root = tmp_path.resolve()
    monkeypatch.setenv('HERMES_HOME', str(root))
    dbs = []

    def build(member_home):
        member_home.mkdir(parents=True, exist_ok=True)
        db = open_persistence(profile_scope(home=root))
        dbs.append(db)
        authority = SimpleNamespace(db=db, persistence=db, profile_id=str(root),
                                    epoch=db.runtime.begin_epoch(instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        roster = {'default': root, 'two': member_home}
        monkeypatch.setattr(service, 'profile_homes', lambda: dict(roster))
        now = [time.time()]
        service.runtime.clock = lambda: now[0]
        service.runtime.publish_terminal = None
        service.runtime.lease_ttl_seconds, service.runtime.indeterminate_defer_seconds = 1000, 5
        service.insert_room(room_id='room', name='Room', owner='alice', members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'two', 'handle': 'two'}])
        identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db, identity, placement='profile', clock=service.runtime.clock, payload={
            'target_profile': 'two', 'target_member_id': 'two', 'source_event_seq': 1, 'prompt': 'inspect'})
        binding = service.bindings()[0]
        lease = service.runtime._ensure_lease(binding)
        tasks.start_task(db, identity, lease, placement='profile', expected_cancel_generation=0,
                         clock=service.runtime.clock)
        # The member's daemon does not answer: Stop stays pending, then defers the turn.
        assert service.runtime.cancel(identity, cancel_id='stop-1')['status'] == 'stopping'
        now[0] += 10
        assert service.runtime._retry_stopping_tasks(binding, lease) is False
        assert tasks.get_task(db, identity, placement='profile')['status'] == 'deferred'
        # Still served here: only its daemon's confirmation may settle the turn.
        with pytest.raises(RoomStopPendingError):
            service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)
        return service, db, identity, roster

    yield build
    for db in dbs:
        db.close()


def _disband(service):
    service.stop_room('room', cancel_id='room-disbanded', require_acknowledged=True)


def test_disband_settles_the_turn_once_the_profile_is_unserved_and_no_daemon_runs(tmp_path, deferred):
    service, db, identity, roster = deferred(tmp_path.resolve() / 'profiles' / 'two')
    del roster['two']  # unserved: its entry removed (or the profile deleted)
    _disband(service)
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'cancelled'


@pytest.mark.parametrize('layout', ['profiles', 'elsewhere'])
def test_a_live_daemon_that_does_not_answer_keeps_the_turn(tmp_path, deferred, layout):
    home = tmp_path.resolve() / layout / 'two'
    service, db, identity, roster = deferred(home)
    daemon = ProfileOwnership()
    daemon.reserve([home])  # the daemon is alive: it holds its home's gateway lock, its socket is gone
    try:
        del roster['two']
        for _ in range(3):
            with pytest.raises(RoomStopPendingError):
                _disband(service)
        assert tasks.get_task(db, identity, placement='profile')['status'] == 'deferred'
    finally:
        daemon.close()
    _disband(service)  # the daemon exited: nothing can run the turn any more
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'cancelled'


def test_a_deleted_profile_whose_entry_remains_settles_once_its_daemon_is_gone(tmp_path, deferred):
    from hermes_constants import mark_named_profile_deleted
    home = tmp_path.resolve() / 'profiles' / 'two'
    service, db, identity, _roster = deferred(home)
    mark_named_profile_deleted(home)  # `hermes profile delete` tombstones it; the config entry stays
    _disband(service)
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'cancelled'
