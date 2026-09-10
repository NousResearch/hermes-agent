"""Upgrade caches written before #104919 without changing admitted work or policy."""

import sqlite3

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_policy_checkpoint as policy
from gateway import hosted_rooms as rooms


PROFILES = ("writer", "reviewer")
CACHE_TABLES = ("hosted_room_policy_publications", "hosted_room_policy_watermarks",
                "hosted_room_policy_transcript")
PRESERVED_TABLES = ("hosted_room_policy_threads", "hosted_room_policy_events",
                    "hosted_room_policy_cursors", "hosted_room_driver_tasks", "hosted_room_events")


class _PreLateReceiptCheckpoint(policy.HostedRoomPolicyCheckpoint):
    """Reproduce 83467c28f789's projection-only receipt indexing."""

    def _apply_event(self, conn, event):
        if event["kind"] in {"message.member", *policy._TERMINAL_KINDS}:
            source = conn.execute(
                "SELECT 1 FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? LIMIT 1",
                (event["room_id"], event["payload"].get("discussion_event_id"))).fetchone()
            if source is None:
                return
        super()._apply_event(conn, event)


def _rows(db, tables, room_id="room"):
    with sqlite3.connect(db) as conn:
        return {table: conn.execute(f"SELECT * FROM {table} WHERE room_id=? ORDER BY rowid", (room_id,)).fetchall()
                for table in tables}


def _user(db, event_id, room_id="room"):
    return rooms.append_event(db, room_id=room_id, event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "owner"}, authority_gateway_id="home", authority_epoch=1,
        payload={"text": "@writer Prepare the accepted plan.", "thread_id": "thread"})


def _legacy_store(tmp_path, status="settled", text="Committed old answer.", gate="newer"):
    db = tmp_path / "state.db"
    room = rooms.create_room(db, room_id="room", name="Review", authority_gateway_id="home",
        members=[{"member_id": p, "profile": p, "handle": p} for p in PROFILES])
    source = _user(db, "source")
    task = discussion.plan_next_task(room, [source], local_profiles=PROFILES).task
    deferred = discussion.plan_publication(room, [source], task, status="deferred",
        execution_generation=1, result={"reason": "member_unavailable"}, local_profiles=PROFILES)
    rooms.append_event(db, **deferred.events[0].append_kwargs("room"))
    rooms.append_event(db, room_id="room", event_id="completed", kind="room.activity",
        actor={"kind": "gateway", "id": "home"}, authority_gateway_id="home", authority_epoch=1,
        payload={"status": "bounded" if gate == "bounded" else "settled", "reason_code": "silent_round",
                 "thread_id": "thread", "discussion_event_id": "source"})
    late = discussion.plan_publication(room, rooms.read_events(db, room_id="room")["events"], task,
        status=status, execution_generation=2 if status == "deferred" else None,
        result={"text": text, "reason": "unavailable", "error": "failed"}, local_profiles=PROFILES)
    visible = None
    if len(late.events) == 2:
        visible = rooms.append_event(db, **late.events[0].append_kwargs("room"))
    if gate == "newer":
        _user(db, "newer")
    elif gate == "stopped":
        rooms.request_room_stop(db, room_id="room", cancel_id="stop",
                                expected_gateway_id="home", expected_epoch=1)
    terminal = rooms.append_event(db, **late.events[-1].append_kwargs("room"))
    old = _PreLateReceiptCheckpoint(db)
    old.snapshot(room_id="room", latest_seq=terminal["seq"])
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=1 WHERE room_id='room'")
    # An existing task is not re-derived from newly repaired context.
    identity = driver.TaskIdentity("room", "already-admitted", "thread", "old-turn")
    driver.admit_task(db, identity, payload={"target_profile": "reviewer",
        "target_member_id": "reviewer", "source_event_seq": source["seq"],
        "prompt": "The exact previously admitted prompt."}, clock=lambda: 100)
    rooms.create_room(db, room_id="other", name="Other", members=[], authority_gateway_id="home")
    other = _user(db, "other-user", "other")
    policy.HostedRoomPolicyCheckpoint(db).snapshot(room_id="other", latest_seq=other["seq"])
    return db, task, visible, terminal


@pytest.mark.parametrize("status,text", [
    ("settled", "Committed old answer."), ("settled", "(pass)"),
    ("failed", ""), ("cancelled", ""), ("deferred", ""),
])
@pytest.mark.parametrize("gate", ["newer", "stopped", "bounded", "closed"])
def test_v1_upgrade_repairs_all_outcome_caches_without_restarting_work(tmp_path, status, text, gate):
    db, task, visible, terminal = _legacy_store(tmp_path, status, text, gate)
    preserved = _rows(db, PRESERVED_TABLES)
    other = _rows(db, CACHE_TABLES, "other")
    upgraded = policy.HostedRoomPolicyCheckpoint(db)
    snapshot = upgraded.snapshot(room_id="room", latest_seq=terminal["seq"])
    assert upgraded.publication_exists(room_id="room", task_id=task.identity.task_id,
                                       status=status, execution_generation=2)
    assert _rows(db, PRESERVED_TABLES) == preserved
    assert _rows(db, CACHE_TABLES, "other") == other
    with sqlite3.connect(db) as conn:
        watermark = conn.execute("SELECT seen_through_seq FROM hosted_room_policy_watermarks "
                                 "WHERE room_id='room' AND member_id='writer'").fetchone()[0]
        assert watermark == (visible["seq"] if visible else task.seen_through_seq)
        if visible:
            assert conn.execute("SELECT settled_seq FROM hosted_room_policy_transcript "
                                "WHERE room_id='room' AND seq=?", (visible["seq"],)).fetchone()[0] == terminal["seq"]
    assert bool(snapshot.events) is (gate == "newer")
    after = _rows(db, (*CACHE_TABLES, *PRESERVED_TABLES))
    assert upgraded.snapshot(room_id="room", latest_seq=terminal["seq"]) == snapshot
    assert _rows(db, (*CACHE_TABLES, *PRESERVED_TABLES)) == after


def test_upgrade_failure_rolls_back_cache_rebuild_and_version(tmp_path, monkeypatch):
    db, task, visible, terminal = _legacy_store(tmp_path)
    before = _rows(db, (*CACHE_TABLES, *PRESERVED_TABLES, "hosted_room_policy_transcript_state"))
    upgraded = policy.HostedRoomPolicyCheckpoint(db)
    original = upgraded._store_transcript_event

    def fail_on_late_reply(conn, *, event, **kwargs):
        if event["seq"] == visible["seq"]:
            raise RuntimeError("interrupted cache migration")
        return original(conn, event=event, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(upgraded, "_store_transcript_event", fail_on_late_reply)
        with pytest.raises(RuntimeError, match="interrupted cache migration"):
            upgraded.snapshot(room_id="room", latest_seq=terminal["seq"])
    assert _rows(db, (*CACHE_TABLES, *PRESERVED_TABLES, "hosted_room_policy_transcript_state")) == before
    upgraded.snapshot(room_id="room", latest_seq=terminal["seq"])
    assert upgraded.publication_exists(room_id="room", task_id=task.identity.task_id,
                                       status="settled", execution_generation=2)
