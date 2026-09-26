"""Regression coverage for URI-safe legacy hosted-room imports."""

import pytest

from gateway import hosted_rooms as rooms

USER = {"kind": "user", "id": "desktop-user", "display_name": "User"}


@pytest.mark.parametrize("home_name", ["plain", "work#notes", "work%23notes"])
def test_legacy_import_preserves_uri_significant_home_names(tmp_path, home_name):
    home = tmp_path / home_name
    home.mkdir()
    legacy = home / "state.db"

    rooms.create_room(
        legacy,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=10,
    )
    rooms.append_event(
        legacy,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "before the upgrade"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        now=11,
    )

    store = home / "shared-state.db"

    assert [room["room_id"] for room in rooms.list_rooms(store)] == ["room-1"]
    assert rooms.room_state(store, room_id="room-1")["latest_seq"] == 1
    assert [
        event["event_id"] for event in rooms.read_events(store, room_id="room-1")["events"]
    ] == ["event-1"]
    assert rooms.room_state(legacy, room_id="room-1")["latest_seq"] == 1
