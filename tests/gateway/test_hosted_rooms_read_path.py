"""Focused lock-behavior coverage for hosted Group Chat reads."""

from gateway import hosted_rooms


def test_list_rooms_does_not_enter_a_write_transaction_or_prune(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )

    def reject_write_transaction(*_args, **_kwargs):
        raise AssertionError("list_rooms must not enter a write transaction")

    def reject_prune(*_args, **_kwargs):
        raise AssertionError("list_rooms must not prune retention state")

    monkeypatch.setattr(hosted_rooms, "_transaction", reject_write_transaction)
    monkeypatch.setattr(
        hosted_rooms,
        "_prune_disbanded_rooms_locked",
        reject_prune,
    )

    rows = hosted_rooms.list_rooms(db)

    assert [row["room_id"] for row in rows] == ["room-1"]
