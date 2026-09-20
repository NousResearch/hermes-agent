"""Owned policy-selection contracts with real SQLite and no Output consumer.

The callback supplies held thread identities, not execution or publication authority.
Only room-log/checkpoint storage is initialized; no service, owner or worker starts.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "policy.db"
    rooms.create_room(path, room_id="room", name="Policy", members=[],
                      authority_gateway_id="home", now=1)
    events = []
    # A later discussion in the same causal thread must be held as a unit.
    for event_id, thread in (("old", "held"), ("followup", "held"),
                             ("independent", "free"), ("later", "later")):
        events.append(rooms.append_event(
            path, room_id="room", event_id=event_id, kind="message.user",
            actor={"kind": "user", "id": "alice"},
            payload={"thread_id": thread, "text": event_id},
            authority_gateway_id="home", authority_epoch=1, now=2))
    policy = HostedRoomPolicyCheckpoint(path)
    return policy, events


@pytest.mark.parametrize("mode", ["default", "empty", "held", "all", "refusal"])
def test_snapshot_preserves_fifo_and_excludes_whole_held_threads(checkpoint, mode):
    policy, events = checkpoint
    calls = []

    def held(conn):
        assert conn.in_transaction
        calls.append(conn)
        if mode == "refusal":
            raise RuntimeError("held-thread evidence unavailable")
        return frozenset({"held", "free", "later"} if mode == "all"
                         else {"held"} if mode == "held" else ())

    kwargs = {} if mode == "default" else {"held_output_threads": held}
    if mode == "refusal":
        with pytest.raises(RuntimeError, match="held-thread evidence unavailable"):
            policy.snapshot(room_id="room", latest_seq=events[-1]["seq"], **kwargs)
    else:
        selected = policy.snapshot(room_id="room", latest_seq=events[-1]["seq"], **kwargs)
        expected = (events[2],) if mode == "held" else () if mode == "all" else tuple(events[:2])
        assert selected.events == expected
        assert selected.through_seq == events[-1]["seq"]
        assert selected.stopped_through_seq == 0
        assert selected.watermarks == {}
    assert len(calls) == (0 if mode == "default" else 1)
    # A prior selection/hold/failure never marks the discussion completed.
    again = policy.snapshot(room_id="room", latest_seq=events[-1]["seq"])
    assert again.events == tuple(events[:2])


@pytest.mark.parametrize("outcome", ["selected", "empty", "refusal"])
def test_snapshot_uses_one_view_and_ends_transaction_before_connection_exit(checkpoint, outcome):
    policy, events = checkpoint
    latest = events[-1]["seq"]
    exited = []
    callbacks = []
    with closing(sqlite3.connect(policy.db_path, timeout=5)) as reader:
        reader.row_factory = sqlite3.Row

        @contextmanager
        def supplied():
            assert not reader.in_transaction
            # Actual sync must finish before entering the supplied read lifetime.
            assert reader.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id='room'").fetchone()[0] == latest
            try:
                yield reader
            finally:
                assert not reader.in_transaction
                # This fresh read must see the writer, not the old selection view.
                exited.append(reader.execute("SELECT stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id='room'").fetchone()[0])

        def commit_different_view():
            with closing(sqlite3.connect(policy.db_path, timeout=5)) as writer, writer:
                writer.execute("UPDATE hosted_room_policy_cursors SET stopped_through_seq=through_seq WHERE room_id='room'")
                writer.execute("UPDATE hosted_room_policy_threads SET completed=1 WHERE room_id='room'")
                writer.execute("UPDATE hosted_room_policy_events SET event_json=? WHERE room_id='room' AND discussion_event_id='independent'",
                               (json.dumps(dict(events[2], payload={"text": "new view"})),))
                writer.execute("INSERT INTO hosted_room_policy_watermarks VALUES ('room','free','reader',999)")

        def held(conn):
            assert conn is reader and conn.in_transaction
            callbacks.append(conn)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(commit_different_view).result(timeout=5)
            assert conn.execute("SELECT completed FROM hosted_room_policy_threads WHERE room_id='room' AND thread_id='free'").fetchone()[0] == 0
            if outcome == "refusal":
                raise RuntimeError("held-thread evidence unavailable")
            return frozenset({"held", "free", "later"} if outcome == "empty" else {"held"})

        kwargs = dict(room_id="room", latest_seq=latest, held_output_threads=held,
                      read_connection=supplied)
        if outcome == "refusal":
            with pytest.raises(RuntimeError, match="held-thread evidence unavailable"):
                policy.snapshot(**kwargs)
        else:
            selected = policy.snapshot(**kwargs)
            assert selected.through_seq == latest
            assert selected.stopped_through_seq == 0
            assert selected.events == ((events[2],) if outcome == "selected" else ())
            assert selected.watermarks == {}
        assert callbacks == [reader]
        assert exited == [latest]
        assert not reader.in_transaction
        assert reader.execute("SELECT seen_through_seq FROM hosted_room_policy_watermarks WHERE room_id='room'").fetchone()[0] == 999
