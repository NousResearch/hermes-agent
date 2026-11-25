"""The SQLite ``HostedRoomPeerRepository`` leaves the file that holds a room link's grant owner-only, on both
placements, whatever mode the file had before the write."""

from __future__ import annotations

import stat

import pytest

from persistence.gateway.hosted_rooms import RoomLinkRecord
from persistence_assembly import open_persistence, profile_scope

LINK = RoomLinkRecord(
    room_id="room", member_id="m1", target_url="https://peer.example/rooms", target_profile="p1", grant="secret-grant",
    catalog_json="[]", cancellation_scope_id="scope-1", trace_id="trace-1", transport_security="tls", status="ready",
    updated_at=100.0)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_saving_a_room_link_leaves_its_file_owner_only(tmp_path, placement):
    home = tmp_path / "profiles" / "p1"
    home.mkdir(parents=True)
    store = open_persistence(profile_scope(home=home))
    try:
        path = home / "state.db" if placement == "profile" else tmp_path / "shared-state.db"
        assert store.hosted_room_peers.list_room_link_records(placement=placement) == []
        path.chmod(0o644)
        store.hosted_room_peers.upsert_room_link_record(placement=placement, record=LINK, max_links=8)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()
