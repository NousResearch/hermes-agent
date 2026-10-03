"""Accepted work across a move: classification, inherited runs at the participant, member sessions, the
policy's earlier hosts, and turns that wait for the old host."""

import hashlib

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_fence as fence
from gateway import hosted_room_succession as succession
from gateway import hosted_room_succession_move as move
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_run_scope import room_run_scope_key

H, S, T, P = "install:" + "a" * 32, "install:" + "b" * 32, "install:" + "c" * 32, "install:" + "d" * 32


def admission(task_id, member, target, generation=1):
    return {"task": {"room_id": "room", "task_id": task_id, "thread_id": "main", "turn_id": f"turn-{task_id}"},
            "execution_generation": generation, "target_member_id": member, "target_install_id": target,
            "source_event_seq": 3}


def event(kind, payload, seq):
    return {"kind": kind, "payload": payload, "seq": seq}


def test_pending_admissions_are_classified_from_the_backups_run_evidence():
    events = [event("task.admitted", admission("done", "peer-bot", P), 4),
              event("task.admitted", admission("running", "peer-bot", P), 5),
              event("task.admitted", admission("nobody-knows", "peer-bot", P), 6),
              event("task.admitted", admission("local", "writer", H), 7),
              event("task.admitted", admission("settled", "peer-bot", P), 8),
              event("turn.settled", {"task_id": "settled"}, 9),
              event("task.admitted", admission("copy-only", "custody:installation", P), 10)]
    pending = move.pending_admissions(events)
    assert [item["task"]["task_id"] for item in pending] == ["done", "running", "nobody-knows", "local"]
    index = move.evidence_index({P: {"runs": [
        {"run_id": "run-1", "task_id": "done", "execution_generation": 1, "target_install_id": P,
         "status": "completed"},
        {"run_id": "run-2", "task_id": "running", "execution_generation": 1, "target_install_id": P,
         "status": "running"}]}})
    states = [move.classify(item, index, host_install_id=H, host_members={"writer"})[0] for item in pending]
    # A turn for the old host's own Bot may have run there: unknown, never run again by itself.
    assert states == ["completed", "elsewhere", "unknown", "unknown"]
    assert move.work_counts([{"state": state} for state in states]) == {
        "completed": 1, "elsewhere": 1, "unknown": 2, "waiting_for_host": 0}


def identity(gateway, epoch):
    return {"room_id": "room", "home_install_id": gateway, "authority_gateway_id": gateway, "authority_epoch": epoch,
            "member_id": "peer-bot", "target_install_id": P, "target_profile": "default"}


@pytest.fixture
def runs(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs_idempotency.db"))
    yield store
    store.close()


def reserve(store, scope_identity, key, run_id):
    return store.reserve(room_run_scope_key(scope_identity), key, "fingerprint-" + run_id, run_id,
                         {"status": "queued"}, identity=scope_identity)


def test_a_later_host_re_dispatching_admitted_work_reattaches_to_the_existing_run(runs):
    assert reserve(runs, identity(H, 1), "room:task-1:1", "run-old")[0] == "created"
    fence.fence_and_promise(runs.path, room_id="room", fence_epoch=1, promise_epoch=2, candidate_install_id=S)
    outcome, record = reserve(runs, identity(S, 2), "room:task-1:1", "run-new")
    assert outcome == "inherited" and record["run_id"] == "run-old"
    # Two moves later, the same task still runs once.
    fence.fence_and_promise(runs.path, room_id="room", fence_epoch=2, promise_epoch=3, candidate_install_id=T)
    assert reserve(runs, identity(T, 3), "room:task-1:1", "run-newer") == ("inherited", record)
    # A later generation of a task that finished is the same work too; one that failed may run again.
    runs.update_status("run-old", {"status": "completed"})
    assert reserve(runs, identity(T, 3), "room:task-1:2", "run-retry")[1]["run_id"] == "run-old"
    runs.update_status("run-old", {"status": "failed"})
    assert reserve(runs, identity(T, 3), "room:task-1:1", "run-same")[1]["run_id"] == "run-old"
    assert reserve(runs, identity(T, 3), "room:task-1:2", "run-retry")[0] == "created"


def test_only_the_computer_an_epoch_was_promised_to_dispatches_at_it(runs):
    reserve(runs, identity(H, 1), "room:task-1:1", "run-old")
    fence.fence_and_promise(runs.path, room_id="room", fence_epoch=1, promise_epoch=2, candidate_install_id=S)
    with pytest.raises(fence.RoomAuthorityPromised):
        reserve(runs, identity(T, 2), "room:task-1:1", "run-rival")
    assert reserve(runs, identity(S, 2), "room:task-1:1", "run-new")[1]["run_id"] == "run-old"
    with pytest.raises(fence.RoomAuthorityFenced):
        reserve(runs, identity(H, 1), "room:task-2:1", "run-late")
    # A room that never moved admits as before.
    other = {**identity(H, 1), "room_id": "other-room"}
    assert reserve(runs, other, "room:task-1:1", "run-other")[0] == "created"


def test_a_member_session_is_keyed_to_the_rooms_original_home(tmp_path):
    db = tmp_path / "state.db"

    def legacy(home):
        return "room_" + hashlib.sha256(f"{home}\0room\0peer-bot\0default".encode()).hexdigest()[:32]

    from gateway import hosted_rooms as rooms
    with rooms._transaction(db, immediate=True):
        pass
    # A room whose host never changed keeps exactly today's session.
    assert succession.member_session_id(db, home_install_id=H, room_id="room", member_id="peer-bot",
                                        target_profile="default") == legacy(H)
    with rooms._transaction(db, immediate=True) as conn:
        succession.record_lineage_locked(conn, "room", origin_install_id=H, gateway_id=S, epoch=2, role="promised")
    assert succession.member_session_id(db, home_install_id=S, room_id="room", member_id="peer-bot",
                                        target_profile="default") == legacy(H)
    assert succession.member_session_for(H, "room", "peer-bot", "default") == legacy(H)


ROOM_VALUE = {"room_id": "room", "name": "Room", "authority_gateway_id": S, "authority_epoch": 2,
              "members": [{"member_id": "writer", "handle": "writer", "profile": "default"},
                          {"member_id": "reviewer", "handle": "reviewer", "profile": "reviewer"}]}


def test_the_policy_checks_history_against_the_host_that_wrote_it():
    settled = {"room_id": "room", "seq": 2, "event_id": "dterminal:1", "kind": "turn.deferred",
               "actor": {"kind": "gateway", "id": H}, "authority_epoch": 1,
               "payload": {"discussion_event_id": "user:0", "member_id": "writer", "member_index": 0,
                           "round_index": 0, "task_id": "dtask:1", "thread_id": "main", "turn_id": "turn-1",
                           "seen_through_seq": 1, "execution_generation": 1, "reason": "member_unavailable"}}
    profiles = ("default", "reviewer")
    with pytest.raises(discussion.DiscussionValidationError):
        discussion.derive_member_watermarks(ROOM_VALUE, [settled], local_profiles=profiles)
    moved = {**ROOM_VALUE, "authority_lineage": {"1": H}}
    assert discussion.derive_member_watermarks(moved, [settled], local_profiles=profiles) == {("main", "writer"): 1}
    forged = {**settled, "actor": {"kind": "gateway", "id": T}}
    with pytest.raises(discussion.DiscussionValidationError):
        discussion.derive_member_watermarks(moved, [forged], local_profiles=profiles)


def test_a_turn_waiting_for_the_old_host_names_the_missing_bot_and_its_computer():
    waiting = {"reason": "waiting_for_host", "resource": "bot", "host_name": "Mac mini", "retryable": False}
    extra, effects = discussion._deferred_effects(waiting, execution_generation=1)
    assert extra == {"execution_generation": 1, "reason": "waiting_for_host", "resource": "bot",
                     "host_name": "Mac mini"} and effects == []
    assert discussion._deferred_effects({"reason": "member_unavailable"}, execution_generation=1)[0] == {
        "execution_generation": 1, "reason": "member_unavailable"}
    payload = {"discussion_event_id": "user:0", "member_id": "writer", "member_index": 0, "round_index": 0,
               "task_id": "dtask:1", "thread_id": "main", "turn_id": "turn-1", "seen_through_seq": 1, **extra}
    room = discussion.validate_room(ROOM_VALUE, local_profiles=("default", "reviewer"))
    actor = {"kind": "gateway", "id": S}
    assert discussion._validate_terminal_event("turn.deferred", payload, actor, room)["resource"] == "bot"
    with pytest.raises(discussion.DiscussionValidationError):
        discussion._validate_terminal_event("turn.deferred", {**payload, "reason": "member_unavailable"}, actor, room)
    error = succession.WaitingForHostError(resource="bot", host_name="Mac mini")
    assert (error.not_admitted, error.ambiguous, error.defer_reason) == (True, False, "waiting_for_host")
    assert error.defer_detail == {"resource": "bot", "host_name": "Mac mini"}
    assert set(driver.defer_not_admitted_task.__code__.co_varnames) >= {"detail"}
