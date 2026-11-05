"""A deferred turn this host has no recorded home for (dispatched before homes were recorded), whose
profile's roster entry is gone: the home its daemon ran from is unknown, so an absent conventional
profile directory proves nothing and the turn keeps Disband waiting."""
import pytest

from gateway import hosted_room_driver as tasks
from persistence.gateway.hosted_rooms import RoomStopPendingError
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.test_hosted_room_unserved_member_stop import _disband
from tests.gateway.test_hosted_room_unserved_member_stop import deferred as deferred_fixture

deferred = deferred_fixture


def test_an_unrecorded_custom_home_keeps_the_turn(tmp_path, deferred):
    service, db, identity, roster = deferred(tmp_path.resolve() / 'custom' / 'two')
    raw_holder(db)._execute_write(  # as left by a gateway that never recorded member homes
        lambda conn: conn.execute("DELETE FROM state_meta WHERE key LIKE 'gateway.hosted.member_homes.v1:%'"))
    del roster['two']
    with pytest.raises(RoomStopPendingError):
        _disband(service)
    assert tasks.get_task(db, identity, placement='profile')['status'] == 'deferred'
