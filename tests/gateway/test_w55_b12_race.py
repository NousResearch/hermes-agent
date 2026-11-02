"""Wave 55 battery B12 — recency-guarded room-link upsert (finding W54-F015).

Lost-update regression: two writers upsert a RoomLink grant for the same
(room_id, member_id). The ON CONFLICT clause of ``upsert_room_link_record``
was unconditional last-writer-wins, so a stale writer (older ``updated_at``)
committing after a fresh one overwrote the newer grant.

The fix guards the DO UPDATE with
``WHERE excluded.updated_at >= hosted_room_links.updated_at`` (the same
recency idiom the reservation upsert uses via MAX), turning a stale writer's
upsert into a no-op so the row keeps the newest grant.

Tests here control ``updated_at`` values explicitly (frozen-clock style, zero
sleeps) so the lost update is deterministic: the stale record arrives second.
"""

import threading

from gateway import hosted_rooms


def _record(*, grant: str, target_url: str, updated_at: float) -> dict:
    return {
        "room_id": "room-1",
        "member_id": "member-1",
        "target_url": target_url,
        "target_profile": "profile-1",
        "grant": grant,
        "catalog_json": "{}",
        "cancellation_scope_id": "",
        "trace_id": "",
        "transport_security": "direct",
        "status": "ready",
        "updated_at": updated_at,
    }


def _new_db(tmp_path):
    path = tmp_path / "state.db"
    hosted_rooms.create_room(
        path,
        room_id="room-1",
        name="B12 room",
        members=[{"profile": "profile-1", "handle": "member-1"}],
        authority_gateway_id="gateway-a",
        now=90,
    )
    return path


def _upsert(db_path, record: dict) -> None:
    hosted_rooms.upsert_room_link_record(db_path, record=record, max_links=16)


def _final_row(db_path) -> dict:
    rows = hosted_rooms.list_room_link_records(db_path)
    assert len(rows) == 1
    return rows[0]


def test_stale_upsert_does_not_overwrite_newer_grant(tmp_path):
    """A stale writer committing second must not clobber the newer grant."""
    db = _new_db(tmp_path)
    _upsert(db, _record(grant="grant-new", target_url="https://new.example/room", updated_at=200.0))
    _upsert(db, _record(grant="grant-stale", target_url="https://stale.example/room", updated_at=100.0))
    row = _final_row(db)
    assert row["grant"] == "grant-new"
    assert row["target_url"] == "https://new.example/room"
    assert row["status"] == "ready"
    assert row["updated_at"] == 200.0


def test_newer_upsert_still_replaces_older_grant(tmp_path):
    """The recency guard must not block forward progress for a fresh writer."""
    db = _new_db(tmp_path)
    _upsert(db, _record(grant="grant-old", target_url="https://old.example/room", updated_at=100.0))
    _upsert(db, _record(grant="grant-fresh", target_url="https://fresh.example/room", updated_at=200.0))
    row = _final_row(db)
    assert row["grant"] == "grant-fresh"
    assert row["target_url"] == "https://fresh.example/room"
    assert row["updated_at"] == 200.0


def test_concurrent_upserts_highest_updated_at_wins(tmp_path):
    """Two racing writers: the newest grant survives regardless of commit order."""
    db = _new_db(tmp_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def writer(record: dict) -> None:
        try:
            barrier.wait(timeout=5)
            _upsert(db, record)
        except BaseException as exc:
            errors.append(exc)

    newer = _record(grant="race-newer", target_url="https://race-newer.example/room", updated_at=300.0)
    stale = _record(grant="race-stale", target_url="https://race-stale.example/room", updated_at=250.0)
    threads = [
        threading.Thread(target=writer, args=(newer,)),
        threading.Thread(target=writer, args=(stale,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    row = _final_row(db)
    assert row["grant"] == "race-newer"
    assert row["target_url"] == "https://race-newer.example/room"
    assert row["updated_at"] == 300.0


def test_equal_timestamp_upsert_still_applies(tmp_path):
    """Pin ``>=`` (not ``>``): an equal-timestamp writer must APPLY, not be dropped.

    The recency guard uses ``WHERE excluded.updated_at >= hosted_room_links.updated_at``.
    A ``>`` would silently no-op equal-timestamp upserts (idempotent retries carrying the
    same generation); ``>=`` must let them through. This test distinguishes the two.
    """
    db = _new_db(tmp_path)
    _upsert(db, _record(grant="grant-eq-first", target_url="https://eq.example/first", updated_at=200.0))
    # Same updated_at, different payload: >= applies it, > would drop it.
    _upsert(db, _record(grant="grant-eq-second", target_url="https://eq.example/second", updated_at=200.0))
    row = _final_row(db)
    assert row["grant"] == "grant-eq-second"
    assert row["target_url"] == "https://eq.example/second"
    assert row["updated_at"] == 200.0