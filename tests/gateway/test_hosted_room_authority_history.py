"""Historical context keeps its original authority; it cannot authorize new work."""

import copy
import json
import sqlite3

import pytest

from gateway import hosted_room_authority_history as lineage
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_discussion import (
    GATEWAY_ID, LOCAL_PROFILES, ROOM_ID, _append_user, _events, _settle_next, room_db as room_db,
)


def claim(db, source=GATEWAY_ID, epoch=1, target="gateway-b", event_id="claim-b"):
    return rooms.claim_authority(db, room_id=ROOM_ID, expected_gateway_id=source,
        expected_epoch=epoch, new_gateway_id=target, event_id=event_id)


def test_committed_claims_explain_boundaries_and_idempotent_failback(room_db):
    db, _ = room_db
    assert "authority_history" not in rooms.room_state(db, room_id=ROOM_ID)
    _append_user(db, event_id="user-before", text="Context")
    moved = claim(db)
    first = rooms.room_state(db, room_id=ROOM_ID)
    assert first["authority_history"] == [
        {"gateway_id": GATEWAY_ID, "epoch": 1, "from_seq": 0},
        {"gateway_id": "gateway-b", "epoch": 2, "from_seq": moved["claim_event"]["seq"]},
    ]
    assert claim(db)["idempotent"] is True
    assert rooms.room_state(db, room_id=ROOM_ID)["authority_history"] == first["authority_history"]
    returned = claim(db, "gateway-b", 2, GATEWAY_ID, "claim-return")
    state = rooms.room_state(db, room_id=ROOM_ID)
    history = lineage.validate_history(state["authority_history"], gateway_id=GATEWAY_ID, epoch=3)
    assert lineage.at_sequence(history, 1).epoch == 1
    assert lineage.at_sequence(history, moved["claim_event"]["seq"]).epoch == 2
    assert lineage.at_sequence(history, returned["claim_event"]["seq"]).epoch == 3


def test_old_replies_are_context_not_restarted_tasks_after_explicit_stop_boundary(room_db):
    db, original = room_db
    _append_user(db, event_id="user-before", text="@research Report briefly.")
    old_task = _settle_next(original, db, text="Earlier answer")
    claim(db)
    rooms.request_room_stop(db, room_id=ROOM_ID, cancel_id="manual-boundary", expected_gateway_id="gateway-b", expected_epoch=2)
    state = rooms.room_state(db, room_id=ROOM_ID)
    stopped = discussion.plan_next_task(state, _events(db), local_profiles=LOCAL_PROFILES)
    assert stopped.task is None
    fresh = rooms.append_event(db, room_id=ROOM_ID, event_id="fresh-user", kind="message.user",
        actor={"kind": "user", "id": "local-user"}, payload={"text": "@build Summarize the earlier answer.", "thread_id": "thread-1"},
        authority_gateway_id="gateway-b", authority_epoch=2)
    state = rooms.room_state(db, room_id=ROOM_ID)
    result = discussion.plan_next_task(state, _events(db), local_profiles=LOCAL_PROFILES)
    assert result.task is not None
    assert result.task.identity != old_task.identity
    assert result.task.payload["source_event_seq"] == fresh["seq"]
    # Build has not consumed this reply yet; Research's own seen reply is
    # intentionally omitted from its delta to preserve the existing cache contract.
    assert "Earlier answer" in result.task.payload["prompt"]
    without_history = {key: value for key, value in state.items() if key != "authority_history"}
    with pytest.raises(discussion.DiscussionValidationError):
        discussion.plan_next_task(without_history, _events(db), local_profiles=LOCAL_PROFILES)


@pytest.mark.parametrize("change", ["old_epoch_after_change", "new_epoch_before_change", "wrong_gateway"])
def test_known_epoch_does_not_allow_events_on_the_wrong_side_of_its_boundary(room_db, change):
    db, original = room_db
    _append_user(db, event_id="user-before", text="@research Report.")
    _settle_next(original, db, text="Earlier answer")
    claim(db)
    rooms.request_room_stop(db, room_id=ROOM_ID, cancel_id="stop", expected_gateway_id="gateway-b", expected_epoch=2)
    state = rooms.room_state(db, room_id=ROOM_ID)
    events = copy.deepcopy(_events(db))
    if change == "old_epoch_after_change":
        events[-1]["authority_epoch"] = 1
    elif change == "new_epoch_before_change":
        events[0]["authority_epoch"] = 2
    else:
        events[-1]["actor"]["id"] = GATEWAY_ID
    with pytest.raises(discussion.DiscussionValidationError):
        discussion.plan_next_task(state, events, local_profiles=LOCAL_PROFILES)


@pytest.mark.parametrize("invalid", [
    [], [{"gateway_id": "gateway-b", "epoch": 2, "from_seq": 0}],
    [{"gateway_id": "gateway-a", "epoch": True, "from_seq": 0}],
    [{"gateway_id": "gateway-a", "epoch": 1, "from_seq": False}],
    [{"gateway_id": "gateway-a", "epoch": 1, "from_seq": 0}, {"gateway_id": "gateway-b", "epoch": 3, "from_seq": 2}],
    [{"gateway_id": "gateway-a", "epoch": 1, "from_seq": 0}, {"gateway_id": "gateway-b", "epoch": 2, "from_seq": 0}],
])
def test_invalid_context_does_not_bless_a_higher_epoch(invalid):
    with pytest.raises(lineage.AuthorityHistoryError):
        lineage.validate_history(invalid, gateway_id="gateway-b", epoch=2)


def test_capacity_refuses_new_claim_before_the_current_room_becomes_unreadable(room_db, monkeypatch):
    db, _ = room_db
    monkeypatch.setattr(lineage, "MAX_AUTHORITY_SPANS", 2)
    claim(db)
    before = rooms.room_state(db, room_id=ROOM_ID)
    with pytest.raises(rooms.AuthorityConflictError, match="history is full"):
        claim(db, "gateway-b", 2, "gateway-c", "claim-c")
    assert rooms.room_state(db, room_id=ROOM_ID) == before
    assert claim(db)["idempotent"] is True


def test_persistent_capacity_guard_also_covers_an_older_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage, "MAX_AUTHORITY_SPANS", 2)
    db = tmp_path / "state.db"
    rooms.create_room(db, room_id=ROOM_ID, name="Group", members=[{"member_id": "one"}, {"member_id": "two"}],
        authority_gateway_id=GATEWAY_ID)
    claim(db)
    with sqlite3.connect(db) as old_writer:
        with pytest.raises(sqlite3.IntegrityError, match="authority history capacity"):
            old_writer.execute("""INSERT INTO hosted_room_events
                (room_id,seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at)
                VALUES (?,2,'old-claim','authority.claimed',?,3,?,2)""",
                (ROOM_ID, json.dumps({"kind": "system", "id": "authority-control"}),
                 json.dumps({"previous_gateway_id": "gateway-b", "authority_gateway_id": "gateway-c", "authority_epoch": 3})))
    assert rooms.room_state(db, room_id=ROOM_ID)["authority_epoch"] == 2
