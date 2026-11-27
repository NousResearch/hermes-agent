"""Focused lock-behavior coverage for hosted Group Chat reads."""

from gateway import hosted_rooms
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_list_rooms_does_not_enter_a_write_transaction_or_prune(tmp_path, monkeypatch):
    db = storage_handle(tmp_path)
    hosted_rooms.create_room(
        db,
        placement="profile",
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )

    def reject_write_transaction(*_args, **_kwargs):
        raise AssertionError("list_rooms must not enter a write transaction")

    def reject_prune(*_args, **_kwargs):
        raise AssertionError("list_rooms must not prune retention state")

    monkeypatch.setattr(
        "persistence.sqlite.gateway._hosted_store.HostedFiles._hosted_write", reject_write_transaction)
    monkeypatch.setattr(
        "persistence.sqlite.gateway.hosted_rooms._prune_disbanded_rooms_locked",
        reject_prune,
    )

    rows = hosted_rooms.list_rooms(db, placement="profile")

    assert [row["room_id"] for row in rows] == ["room-1"]
