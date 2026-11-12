"""Disband drops the room's recorded member homes with its tombstone: they never pile up per room."""
from hermes_agent.gateway import hosted_rooms
from hermes_agent.gateway.session_hosted_member_homes import _recorded_homes, record_member_home
from hermes_agent.hermes_state import SessionDB


def test_disband_forgets_the_rooms_member_homes(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        gateway_id = hosted_rooms.local_authority_gateway_id()
        for room_id in ('room', 'other'):
            hosted_rooms.create_room(db.db_path, room_id=room_id, name=room_id, authority_gateway_id=gateway_id,
                                     members=[{'member_id': 'two', 'profile': 'two', 'handle': 'two'}])
            record_member_home(db, room_id, 'two', tmp_path / 'profiles' / 'two')
        hosted_rooms.disband_room(db.db_path, room_id='room', expected_gateway_id=gateway_id, expected_epoch=1)
        assert _recorded_homes(db, 'room', 'two') == []
        assert _recorded_homes(db, 'other', 'two') == [tmp_path / 'profiles' / 'two']
