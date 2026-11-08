"""The work-record capture of the SQLite hosted-room backend against main's (gateway/hosted_room_work_records.py,
gateway/hosted_room_work_storage.py, read from the pinned checkout).

Each file gets its evidence schema from its own first capture, inside the savepoint of the driver transition that captures, so the schema, the rows each driver transition captures and the authority-transition trigger are compared across independently built files.
"""

from __future__ import annotations

import pytest

from persistence.gateway import hosted_rooms as contract
from persistence.sqlite.gateway.hosted_room_driver import SQLiteHostedRoomDriverRepository
from tests.persistence_sqlite.hosted._task_fixtures import (
    GATEWAY, ROOM, WORK_TABLES, Hosted, dump, m1_event, m1_room, make_hosted, raw, schema, work_records,
)
from tests.fixtures.storage_holder import raw_holder
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted.test_driver_parity import Ident, Ref, Seed, Step, _admit, _lease, _start, run_chain

__all__ = ["legacy_store"]

pytestmark = pytest.mark.usefixtures("legacy_store")


@pytest.fixture
def hosted() -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def _m1():
    """Main's room driver from the pinned checkout."""
    return legacy_import("gateway.hosted_room_driver")


def _payload(n: int) -> dict:
    return {"target_profile": "p1", "target_member_id": "m1", "prompt": "hello", "source_event_seq": n}


def test_work_record_schema_matches_m1(hosted, tmp_path):
    # CONTRACT: the evidence schema appears with the first capture, inside the driver transition that captures, as
    # main's capture creates it; the room verbs never create it.
    path = hosted.prepare("profile")
    legacy = tmp_path / "m1" / "state.db"
    legacy.parent.mkdir()
    for target in (path, legacy):
        m1_room(target)
        m1_event(target, event_id="e-1", kind="message.user", payload={"text": "q", "thread_id": "thread-1"}, now=100.1)
    hosted.store.hosted_rooms.room_state(placement="profile", room_id=ROOM)
    assert schema(path, "*hosted_room_work_records*") == []
    m1 = _m1()
    m1.admit_task(legacy, m1.TaskIdentity(ROOM, "task-1", "thread-1", "turn-1"), payload=_payload(1), clock=lambda: 101.0)
    SQLiteHostedRoomDriverRepository(hosted.store).admit_task(
        contract.HostedTaskIdentity(ROOM, "task-1", "thread-1", "turn-1"), placement="profile", payload=_payload(1),
        now=101.0)
    for pattern in ("*hosted_room_work_records*", "trg_work_*"):
        assert schema(path, pattern) == schema(legacy, pattern)
        assert schema(path, pattern)


def test_capture_follows_each_transition_and_freezes_on_an_authority_change(hosted, tmp_path):
    path = hosted.prepare("profile")
    legacy = tmp_path / "m1" / "state.db"
    legacy.parent.mkdir()
    for target in (path, legacy):
        m1_room(target)
        for n in (1, 2):
            m1_event(target, event_id=f"e-{n}", kind="message.user", payload={"text": "q", "thread_id": f"thread-{n}"},
                     now=100.0 + n / 10)
    repo = SQLiteHostedRoomDriverRepository(hosted.store)
    m1 = _m1()
    identities = {n: (m1.TaskIdentity(ROOM, f"task-{n}", f"thread-{n}", f"turn-{n}"),
                      contract.HostedTaskIdentity(ROOM, f"task-{n}", f"thread-{n}", f"turn-{n}")) for n in (1, 2)}
    for n, (legacy_identity, identity) in identities.items():
        m1.admit_task(legacy, legacy_identity, payload=_payload(n), clock=lambda: 101.0)
        repo.admit_task(identity, placement="profile", payload=_payload(n), now=101.0)
    lease_args = dict(room_id=ROOM, gateway_id=GATEWAY, authority_epoch=1, process_generation="proc-a", ttl_seconds=60.0)
    legacy_lease = m1.acquire_lease(legacy, **lease_args, clock=lambda: 102.0)
    lease = repo.acquire_lease(placement="profile", **lease_args, now=102.0)
    legacy_attempt = m1.start_task(legacy, identities[1][0], legacy_lease, expected_cancel_generation=0,
                                   clock=lambda: 103.0)
    attempt = repo.start_task(identities[1][1], lease, placement="profile", expected_cancel_generation=0, now=103.0)
    m1.settle_task(legacy, legacy_attempt, settlement_id="settle-1", status="settled", result={"text": "done"},
                   clock=lambda: 104.0)
    repo.settle_task(attempt, placement="profile", settlement_id="settle-1", status="settled",
                     result={"text": "done"}, now=104.0)
    assert dump(path, WORK_TABLES) == dump(legacy, WORK_TABLES)
    (record,) = work_records(path)
    # CONTRACT: one current record per producer scope, revised by every transition that changed the task view.
    assert (record["disposition"], record["availability"], record["revision"]) == ("current", "available", 4)
    assert [(task["task_id"], task["phase"]) for task in record["tasks"]] == [("task-1", "settled"), ("task-2", "queued")]
    for target in (path, legacy):
        raw(target, "UPDATE hosted_rooms SET authority_gateway_id='gw2', authority_epoch=2 WHERE room_id=?", (ROOM,))
    assert dump(path, WORK_TABLES) == dump(legacy, WORK_TABLES)
    # CONTRACT: an authority change freezes the old producer's evidence as historical.
    assert [record["disposition"] for record in work_records(path)] == ["historical"]


def test_invalid_stored_evidence_is_classified_in_place_and_the_transition_commits(hosted, tmp_path):
    run_chain(hosted, tmp_path, "profile", [
        _admit(1, now=101.0),
        Seed(lambda path: raw(path, "UPDATE hosted_room_work_records_source SET record_json='{}'")),
        _admit(2, now=102.0),
        _admit(3, now=103.0),
        _lease(now=104.0, save="lease"),
        _start(1, "lease", now=105.0, save="attempt"),
        Step("list_tasks", kwargs=dict(room_id=ROOM)),
        Step("get_task", (Ident(2),)),
        Step("requeue_not_admitted_task", (Ref("attempt"),), now=106.0),
    ])
    # CONTRACT: the unreadable record is marked invalid, never replaced, and the driver writes still committed.
    with raw_holder(hosted.store)._read_ctx() as conn:
        assert [tuple(row) for row in conn.execute("SELECT disposition FROM hosted_room_work_records_source")] == [
            ("invalid",)]
