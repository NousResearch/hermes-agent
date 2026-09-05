"""Replica bytes belong to admission capacity, not an authority-only prune budget."""

from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replicas import AUTH_A, MEMBERS, _seed_room


def test_authority_pruning_does_not_charge_replica_bytes_twice(tmp_path):
    page = _seed_room(tmp_path / "authority.db")
    db = tmp_path / "shared.db"
    replicas.ingest_page(
        db, room_id="room-1", room_name="Replica", members=MEMBERS, page=page
    )
    _seed_room(db, room_id="local-room", n_events=1)
    rooms.disband_room(
        db, room_id="local-room", expected_gateway_id=AUTH_A, expected_epoch=1
    )
    with rooms._transaction(db, immediate=True) as conn:
        authority_bytes = conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
        ).fetchone()[0]
        assert rooms._prune_disbanded_rooms_locked(
            conn, now=None, max_gateway_event_bytes=authority_bytes
        ) == 0
    assert rooms.read_events(db, room_id="local-room", include_disbanded=True)["events"]
    assert replicas.replica_state(db, room_id="room-1")["last_seq"] == page["cursor"]
