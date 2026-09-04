"""Concurrent checkpoint readers must observe one monotonic policy projection."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


def _append(db, index):
    return hosted_rooms.append_event(
        db,
        room_id="checkpoint",
        event_id=f"user-{index}",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        authority_gateway_id="home",
        authority_epoch=1,
        payload={"text": f"request {index}", "thread_id": "work"},
    )


def _checkpoint(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="checkpoint",
        name="Checkpoint",
        members=[],
        authority_gateway_id="home",
    )
    _append(db, 1)
    _append(db, 2)
    checkpoint = HostedRoomPolicyCheckpoint(db)
    checkpoint.sync(room_id="checkpoint", latest_seq=2)
    return db, checkpoint


def test_migration_discards_page_fetched_before_another_worker_advanced(
    tmp_path, monkeypatch
):
    db, first = _checkpoint(tmp_path)
    second = HostedRoomPolicyCheckpoint(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=1")
    fetched = threading.Event()
    release = threading.Event()
    original = hosted_rooms.read_events

    def delayed(*args, **kwargs):
        page = original(*args, **kwargs)
        if (
            threading.current_thread().name.startswith("stale-migration")
            and kwargs["since_seq"] == 0
        ):
            fetched.set()
            assert release.wait(10)
        return page

    monkeypatch.setattr(hosted_rooms, "read_events", delayed)
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="stale-migration"
    ) as pool:
        pending = pool.submit(first.snapshot, room_id="checkpoint", latest_seq=2)
        try:
            assert fetched.wait(10)
            _append(db, 3)
            _append(db, 4)
            newer = second.snapshot(room_id="checkpoint", latest_seq=4)
        finally:
            release.set()
        resumed = pending.result(timeout=10)
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT through_seq FROM hosted_room_policy_cursors"
            ).fetchone()[0]
            == 4
        )
        assert (
            conn.execute(
                "SELECT latest_user_seq FROM hosted_room_policy_threads"
            ).fetchone()[0]
            == 4
        )
    assert resumed == newer
    assert resumed.through_seq == max(event["seq"] for event in resumed.events) == 4


def test_snapshot_reads_cursor_from_the_same_projection_as_its_rows(
    tmp_path, monkeypatch
):
    db, first = _checkpoint(tmp_path)
    second = HostedRoomPolicyCheckpoint(db)
    original = first.sync

    def advance_after_sync(**kwargs):
        cursor = original(**kwargs)
        _append(db, 3)
        _append(db, 4)
        second.snapshot(room_id="checkpoint", latest_seq=4)
        return cursor

    monkeypatch.setattr(first, "sync", advance_after_sync)
    snapshot = first.snapshot(room_id="checkpoint", latest_seq=2)
    assert snapshot.through_seq == 4
    assert max(event["seq"] for event in snapshot.events) <= snapshot.through_seq


def test_stale_latest_hint_is_not_a_corrupt_checkpoint(tmp_path):
    db, checkpoint = _checkpoint(tmp_path)
    _append(db, 3)
    _append(db, 4)
    expected = checkpoint.snapshot(room_id="checkpoint", latest_seq=4)
    assert checkpoint.snapshot(room_id="checkpoint", latest_seq=2) == expected
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=5")
    with pytest.raises(RuntimeError, match="ahead of the durable log"):
        checkpoint.snapshot(room_id="checkpoint", latest_seq=4)


def test_guarded_event_append_keeps_immutable_replay_after_cursor_changes(tmp_path):
    db, _ = _checkpoint(tmp_path)
    fields = dict(
        room_id="checkpoint",
        event_id="guarded",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        authority_gateway_id="home",
        authority_epoch=1,
        payload={"text": "guarded", "thread_id": "work"},
    )
    appended = hosted_rooms.append_event(db, **fields, expected_latest_seq=2)
    assert appended["seq"] == 3
    _append(db, 4)
    replay = hosted_rooms.append_event(db, **fields, expected_latest_seq=2)
    assert replay["seq"] == 3 and replay["idempotent"]
    with pytest.raises(hosted_rooms.EventCursorConflictError):
        hosted_rooms.append_event(
            db, **{**fields, "event_id": "stale"}, expected_latest_seq=2
        )
    with pytest.raises(hosted_rooms.EventConflictError):
        hosted_rooms.append_event(
            db,
            **{**fields, "payload": {"text": "changed", "thread_id": "work"}},
            expected_latest_seq=4,
        )
    assert hosted_rooms.room_state(db, room_id="checkpoint")["latest_seq"] == 4


@pytest.mark.parametrize("expected", [True, -1, "2", 2.5])
def test_guarded_append_rejects_invalid_cursor_before_writing(tmp_path, expected):
    db, _ = _checkpoint(tmp_path)
    with pytest.raises(hosted_rooms.HostedRoomError, match="nonnegative integer"):
        hosted_rooms.append_event(
            db,
            room_id="checkpoint",
            event_id="bad",
            kind="message.user",
            actor={"kind": "user", "id": "owner"},
            payload={"text": "bad"},
            authority_gateway_id="home",
            authority_epoch=1,
            expected_latest_seq=expected,
        )
    assert hosted_rooms.room_state(db, room_id="checkpoint")["latest_seq"] == 2
