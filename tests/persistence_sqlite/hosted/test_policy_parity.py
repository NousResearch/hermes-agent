"""The SQLite ``HostedRoomPolicyRepository`` against main's own checkpoint (gateway/hosted_room_policy_checkpoint.py),
read from the pinned main checkout.

The room is created through the room verb and its file copied to a twin; the room log is then seeded with the room verb
``append_event`` on the placement's file and main's ``append_event`` on the twin; each step runs
``HostedRoomPolicyCheckpoint(twin).<verb>`` and the backend verb, and the outcomes and every policy table must be
equal after each step. Main's bare ``RuntimeError`` refusals are the contract's ``PolicyProjectionError`` with the same
message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from persistence.sqlite.gateway.hosted_room_policy import SQLiteHostedRoomPolicyRepository
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted._task_fixtures import (
    GATEWAY, MEMBERS, POLICY_TABLES, ROOM, Hosted, dump, m1_event, m1_room, make_hosted, outcome, raw, schema, status,
    twin,
)

__all__ = ["legacy_store"]


@pytest.fixture
def hosted(legacy_store) -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


def _m1_checkpoint(path: Path, **budget: int) -> Any:
    """Main's ``HostedRoomPolicyCheckpoint`` on ``path``, from the pinned checkout."""
    return legacy_import("gateway.hosted_room_policy_checkpoint").HostedRoomPolicyCheckpoint(path, **budget)


def _create_room(hosted: Hosted, placement: str) -> None:
    """The room ``m1_room`` creates, through the room verb on the placement's file."""
    hosted.store.hosted_rooms.create_room(
        placement=placement, room_id=ROOM, name="Room", members=MEMBERS, authority_gateway_id=GATEWAY, now=100.0)


def _actor(kind: str) -> dict:
    return ({"kind": "user", "id": "desktop"} if kind == "message.user"
            else {"kind": "member", "id": "m1"} if kind == "message.member" else {"kind": "gateway", "id": GATEWAY})


@dataclass
class Step:
    verb: str
    kwargs: dict = field(default_factory=dict)
    expect: str = "ok"
    # The configured active-event budget: main's checkpoint takes it at construction, the verb as max_active_events.
    budget: int | None = None


@dataclass
class Seed:
    fn: Callable[[Path], Any]


@dataclass
class Log:
    """Room events appended on both files, the n-th at time ``100 + n``: the room verb on the placement's file, main's
    ``append_event`` on the twin."""
    events: list[tuple[str, str, dict]]


def _log() -> Log:
    """Two discussions, a deferral, a stop, an activity that closes a third thread, and one task's published pair."""
    return Log([
        ("e-u1", "message.user", {"text": "q1", "thread_id": "th1"}),
        ("e-m1", "message.member", {"text": "a1", "thread_id": "th1", "discussion_event_id": "e-u1"}),
        ("e-t1", "turn.settled", {"thread_id": "th1", "discussion_event_id": "e-u1", "task_id": "dtask:abc",
                                  "member_id": "m1", "message_event_id": "e-m1", "seen_through_seq": 2}),
        ("e-u2", "message.user", {"text": "q2", "thread_id": "th2"}),
        ("e-d2", "turn.deferred", {"thread_id": "th2", "discussion_event_id": "e-u2", "task_id": "t2",
                                   "execution_generation": 1, "member_id": "m2", "seen_through_seq": 4}),
        ("e-s", "room.stop_requested", {"cancel_id": "c1"}),
        ("e-u3", "message.user", {"text": "q3", "thread_id": "th3"}),
        ("e-a3", "room.activity", {"thread_id": "th3", "discussion_event_id": "e-u3"}),
        ("e-u4", "message.user", {"text": "q4", "thread_id": "th4"}),
        ("dmessage:abc", "message.member", {"text": "a", "thread_id": "th4", "discussion_event_id": "e-u4",
                                            "task_id": "dtask:abc"}),
        ("dterminal:abc", "turn.failed", {"thread_id": "th4", "discussion_event_id": "e-u4", "task_id": "dtask:abc"}),
    ])


def _legacy_outcome(result: tuple) -> tuple:
    """Main's bare ``RuntimeError`` is ``PolicyProjectionError`` under the contract, with main's message."""
    if result[0] == "error" and result[1] == "RuntimeError":
        return ("error", "PolicyProjectionError", result[2])
    return result


def run_chain(hosted: Hosted, tmp_path: Path, placement: str, steps: list[Step | Seed | Log]) -> list[tuple]:
    path = hosted.prepare(placement)
    _create_room(hosted, placement)
    legacy_path = twin(path, tmp_path / "m1")
    repo = SQLiteHostedRoomPolicyRepository(hosted.store)
    results = []
    for index, step in enumerate(steps):
        if isinstance(step, Seed):
            step.fn(legacy_path)
            step.fn(path)
            continue
        if isinstance(step, Log):
            for number, (event_id, kind, payload) in enumerate(step.events, start=1):
                m1_event(legacy_path, event_id=event_id, kind=kind, payload=payload, actor=_actor(kind),
                         now=100.0 + number)
                hosted.store.hosted_rooms.append_event(
                    placement=placement, room_id=ROOM, event_id=event_id, kind=kind, actor=_actor(kind),
                    payload=payload, authority_gateway_id=GATEWAY, authority_epoch=1, now=100.0 + number)
            continue
        budget = {} if step.budget is None else {"max_active_events": step.budget}
        legacy = _legacy_outcome(outcome(
            lambda: getattr(_m1_checkpoint(legacy_path, **budget), step.verb)(room_id=ROOM, **step.kwargs)))
        backend = outcome(lambda: getattr(repo, step.verb)(placement=placement, room_id=ROOM, **step.kwargs, **budget))
        label = f"step {index} {step.verb}"
        assert backend == legacy, label
        assert status(backend) == step.expect, (label, backend)
        assert dump(path, POLICY_TABLES) == dump(legacy_path, POLICY_TABLES), label
        results.append(backend)
    return results


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_snapshot_publications_task_events_and_compaction(hosted, tmp_path, placement):
    results = run_chain(hosted, tmp_path, placement, [
        _log(),
        Step("snapshot", dict(latest_seq=3)),
        Step("snapshot", dict(latest_seq=11)),
        Step("snapshot", dict(latest_seq=2)),
        Step("publication_exists", dict(task_id="dtask:abc", status="settled", execution_generation=0)),
        Step("publication_exists", dict(task_id="t2", status="deferred", execution_generation=1)),
        Step("publication_exists", dict(task_id="t2", status="deferred", execution_generation=2)),
        Step("publication_exists", dict(task_id="t2", status="settled", execution_generation=0)),
        Step("events_for_task", dict(source_event_seq=1, task_id="dtask:abc")),
        Step("events_for_task", dict(source_event_seq=4, input_context={"watermark": 0, "event_seqs": [1, 4]})),
        Step("events_for_task", dict(source_event_seq=9, task_id="dtask:abc",
                                     input_context={"watermark": 0, "event_seqs": [9]})),
        Step("events_for_task", dict(source_event_seq=99)),
        Step("events_for_task", dict(source_event_seq=2)),
        Step("events_for_task", dict(source_event_seq=4, input_context={"watermark": 0, "event_seqs": [50]}),
             expect="PolicyProjectionError"),
        Step("events_for_task", dict(source_event_seq=2, input_context={"watermark": 0, "event_seqs": [2]}),
             expect="PolicyProjectionError"),
        Step("events_for_task", dict(source_event_seq=1, input_context={"watermark": 3, "event_seqs": [2]}),
             expect="ValueError"),
        Step("events_for_task", dict(source_event_seq=1, task_id="dtask:other")),
        Seed(lambda path: raw(path, "UPDATE hosted_room_policy_threads SET completed=1 WHERE thread_id='th1'")),
        Step("compact_completed"),
        Step("snapshot", dict(latest_seq=11)),
        Seed(lambda path: raw(path, "UPDATE hosted_room_policy_transcript_state SET schema_version=3")),
        Step("snapshot", dict(latest_seq=11)),
        Seed(lambda path: raw(path, "UPDATE hosted_room_policy_cursors SET through_seq=40")),
        Step("snapshot", dict(latest_seq=11), expect="PolicyProjectionError"),
    ])
    assert results[1][1]["through_seq"] == 11 and results[1][1]["stopped_through_seq"] == 6


def test_snapshot_refusals_for_a_missing_room_and_a_cursor_that_disappears(hosted, tmp_path):
    run_chain(hosted, tmp_path, "profile", [
        _log(),
        Step("snapshot", dict(latest_seq=1)),
        Seed(lambda path: raw(path, "DELETE FROM hosted_room_policy_cursors")),
        Step("snapshot", dict(latest_seq=11)),
        Seed(lambda path: raw(path, "UPDATE hosted_room_events SET room_id='gone' WHERE room_id=?", (ROOM,))),
        Seed(lambda path: raw(path, "UPDATE hosted_rooms SET room_id='gone' WHERE room_id=?", (ROOM,))),
        Step("snapshot", dict(latest_seq=11), expect="RoomNotFoundError"),
        Step("compact_completed"),
        Step("publication_exists", dict(task_id="t2", status="deferred", execution_generation=1)),
    ])



def _discussion(members: int, receipts: int) -> Log:
    """One user message, ``members`` member messages and ``receipts`` retry receipts in one discussion."""
    events = [("e-u1", "message.user", {"text": "q", "thread_id": "th1"})]
    events += [(f"e-m{n}", "message.member", {"text": f"a{n}", "thread_id": "th1", "discussion_event_id": "e-u1"})
               for n in range(members)]
    events += [(f"e-d{n}", "turn.deferred", {"thread_id": "th1", "discussion_event_id": "e-u1", "task_id": f"t{n}",
                                            "execution_generation": 1, "member_id": "m1", "seen_through_seq": 1})
               for n in range(receipts)]
    return Log(events)


@pytest.mark.parametrize("members, receipts, expect", [
    (63, 64, "ok"), (64, 1, "PolicyProjectionError"), (1, 65, "PolicyProjectionError")])
def test_retry_receipts_are_bounded_apart_from_the_discussion(hosted, tmp_path, members, receipts, expect):
    # CONTRACT: the active discussion and its retry receipts each have their own bound.
    run_chain(hosted, tmp_path, "profile", [
        _discussion(members, receipts),
        Step("snapshot", dict(latest_seq=1 + members + receipts), expect=expect),
    ])


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_a_configured_budget_is_recorded_and_read_back_as_m1(hosted, tmp_path, monkeypatch, placement):
    # CONTRACT: a configured budget above the default raises the file's capacity; a later default budget, in this
    # process or after a restart, reads the larger capacity back, as main's checkpoint does.
    from persistence.sqlite.gateway import hosted_room_policy

    run_chain(hosted, tmp_path, placement, [
        _discussion(100, 0),
        Step("snapshot", dict(latest_seq=101), expect="PolicyProjectionError"),
        Step("snapshot", dict(latest_seq=101), budget=101),
        Step("snapshot", dict(latest_seq=101)),
        Seed(lambda path: monkeypatch.setattr(hosted_room_policy, "_PREPARED", {})),
        Step("events_for_task", dict(source_event_seq=1)),
        Step("snapshot", dict(latest_seq=101), budget=0, expect="ValueError"),
    ])
    path = hosted.store.hosted_room_policy._files.path(placement)
    legacy = tmp_path / "m1" / path.name
    assert dump(path, ("hosted_room_policy_capacity",)) == dump(legacy, ("hosted_room_policy_capacity",))
    assert dump(path, ("hosted_room_policy_capacity",)) == {"hosted_room_policy_capacity": [(1, 101)]}


def test_policy_schema_matches_m1(hosted, tmp_path):
    # CONTRACT: the checkpoint tables appear with the first checkpoint verb on the file, as main creates them when it
    # constructs the checkpoint; the room verbs never create them.
    path = hosted.prepare("profile")
    _create_room(hosted, "profile")
    hosted.store.hosted_rooms.room_state(placement="profile", room_id=ROOM)
    assert schema(path, "*hosted_room_policy*") == []
    legacy = tmp_path / "m1" / "state.db"
    legacy.parent.mkdir()
    m1_room(legacy)
    _m1_checkpoint(legacy)
    repo = SQLiteHostedRoomPolicyRepository(hosted.store)
    assert repo.publication_exists(placement="profile", room_id=ROOM, task_id="t1", status="settled",
                                   execution_generation=0) is False
    for pattern in ("*hosted_room_policy*", "idx_hosted_room_user_thread"):
        assert schema(path, pattern) == schema(legacy, pattern)
        assert schema(path, pattern)
