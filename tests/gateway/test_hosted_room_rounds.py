"""Cross-engine round traces: the hosted policy against the native Desktop round driver.

The scenarios in ``tests/fixtures/hosted_room_round_traces.json`` are executed by the real
native loop in ``apps/desktop/src/plugins/hermes-bots/round-trace-parity.test.ts`` and by the
real planner and policy checkpoint here, on an isolated store, driven exactly the way
``HostedRoomService.prepare_room`` drives them: sync, snapshot, plan, publish, repeat.

Whole ordered speaker traces and the input delta a released member actually receives are
compared, because equal caps and equal rotation helpers do not make equal conversations.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


ROOM_ID = "trace-room"
GATEWAY_ID = "gateway-traces"
THREAD_ID = "thread-1"
# One scenario cannot exceed the message and round caps by more than its own budget; a run that
# reaches this has stopped converging and must fail loudly instead of looping.
MAX_TURNS = 40
VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "hosted_room_round_traces.json").read_text())
ALL_MEMBERS = {
    member["handle"]: {
        "member_id": member["member_id"], "profile": member["handle"], "handle": member["handle"]}
    for member in VECTORS["members"]}
LOCAL_PROFILES = tuple(ALL_MEMBERS)


def _events(db: Path) -> list[dict]:
    return hosted_rooms.read_events(
        db, room_id=ROOM_ID, since_seq=0, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def _user(db: Path, *, event_id: str, text: str) -> dict:
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "local-user"},
        payload={"text": text, "thread_id": THREAD_ID},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1, now=time.time())


def _activity(db: Path, decision: discussion.DiscussionDecision) -> dict:
    """Complete one discussion exactly as ``_append_room_status`` does."""
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID,
        event_id=f"dactivity:{decision.discussion_event_id}:{decision.reason}",
        kind="room.activity", actor={"kind": "gateway", "id": GATEWAY_ID},
        payload={
            "status": decision.status, "reason_code": decision.reason,
            "thread_id": decision.thread_id, "discussion_event_id": decision.discussion_event_id},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1, now=time.time())


def _drive(db: Path, room: dict, scenario: dict) -> dict:
    """Send each scripted user message and drive its discussion to a terminal decision."""
    checkpoint = HostedRoomPolicyCheckpoint(db)
    spoken: dict[str, int] = {}
    speakers: list[str] = []
    deltas: list[tuple[str, str]] = []
    reasons: list[str] = []
    turns: list[dict] = []
    consumptions: list[dict[str, int]] = []
    for index, text in enumerate(scenario["user"], start=1):
        _user(db, event_id=f"user-{index}", text=text)
        for _ in range(MAX_TURNS):
            latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
            checkpoint.sync(room_id=ROOM_ID, latest_seq=latest)
            snapshot = checkpoint.snapshot(room_id=ROOM_ID, latest_seq=latest)
            decision = discussion.plan_next_task(
                room, list(snapshot.events), local_profiles=LOCAL_PROFILES,
                initial_watermarks=snapshot.watermarks,
                held_member_ids=snapshot.held_member_ids,
                initial_citations=snapshot.citations)
            if decision.held_consumptions:
                consumptions.append(dict(decision.held_consumptions))
            if decision.thread_id and decision.held_consumptions:
                assert checkpoint.apply_held_consumptions(
                    room_id=ROOM_ID, thread_id=decision.thread_id,
                    consumptions=decision.held_consumptions,
                    expected_through_seq=snapshot.through_seq), "the fenced write refused a fresh snapshot"
            if decision.status != "task" or decision.task is None:
                reasons.append(f"{decision.status}:{decision.reason}")
                if decision.status in {"settled", "bounded"}:
                    _activity(db, decision)
                break
            task = decision.task
            handle = task.member.handle
            turns.append({
                "handle": handle, "round": task.round_index, "reason": decision.reason,
                "turn_id": task.identity.turn_id, "task_id": task.identity.task_id,
                "held": tuple(snapshot.held_member_ids)})
            deltas.append((handle, str(task.payload["prompt"])))
            # A flat list is one continuous script for the whole scenario; a per-discussion map
            # restarts with each user message, so a member's answer belongs to the discussion
            # that asked for it however many turns an earlier one took.
            scripted = scenario["replies"].get(handle) or []
            key = handle
            if isinstance(scripted, Mapping):
                key = f"{handle}:{index}"
                scripted = scripted.get(str(index)) or []
            turn = spoken.get(key, 0)
            spoken[key] = turn + 1
            reply = scripted[turn] if turn < len(scripted) else "(pass)"
            for event in discussion.plan_publication(
                room, _events(db), task, status="settled", result={"text": reply},
                local_profiles=LOCAL_PROFILES,
            ).events:
                hosted_rooms.append_event(db, **event.append_kwargs(ROOM_ID), now=time.time())
            if not discussion.is_pass_text(reply):
                speakers.append(handle)
        else:
            raise AssertionError(f"scenario did not reach a terminal decision: {speakers}")
    return {
        "speakers": speakers, "deltas": deltas, "reasons": reasons, "turns": turns,
        # Each walk's own held skips, in order, so a test can pin the bounded value a slot
        # recorded instead of the aggregate later rounds may raise.
        "consumptions": consumptions}


def _room(db: Path, *handles: str) -> dict:
    return hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Traces", members=[ALL_MEMBERS[handle] for handle in handles],
        authority_gateway_id=GATEWAY_ID, now=1)


def _watermark(db: Path, member_id: str) -> int:
    """The durable thread watermark, read straight from the projection.

    Not through ``snapshot``: a completed discussion exposes no active thread, so its snapshot
    carries no watermarks at all, while the consumption it recorded is still durable.
    """
    latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
    HostedRoomPolicyCheckpoint(db).sync(room_id=ROOM_ID, latest_seq=latest)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""SELECT seen_through_seq FROM hosted_room_policy_watermarks
               WHERE room_id=? AND thread_id=? AND member_id=?""",
            (ROOM_ID, THREAD_ID, member_id)).fetchone()
    return 0 if row is None else int(row["seen_through_seq"])


def _seq_of(db: Path, text: str) -> int:
    """The sequence of the one committed message carrying this text."""
    matches = [
        int(event["seq"]) for event in _events(db)
        if str((event.get("payload") or {}).get("text") or "") == text]
    assert len(matches) == 1, f"expected exactly one {text!r} message, found {matches}"
    return matches[0]


def _slots(result: dict) -> list[tuple[int, str]]:
    """Ordered ``(round, phase coordinate)`` of every dispatched turn."""
    return [(turn["round"], turn["turn_id"].split(".s")[0]) for turn in result["turns"]]


def test_a_held_member_owed_a_recovery_turn_keeps_the_input_it_never_read(tmp_path: Path):
    """Holds apply to a recovery turn, and a turn nobody took costs its member no context.

    b is paused, then cited by a peer, then left owed a recovery turn when a later round goes
    quiet. Its ordinary slot already charged it the citing reply; the recovery it is skipped for
    must not charge it the newer message it has still never read.
    """
    db = tmp_path / "state.db"
    result = _drive(db, _room(db, "a", "b"), {
        "user": ["@b stop", "@a lead this", "@a status?"],
        "replies": {"a": {"2": ["@b please confirm"], "3": ["(pass)"]}, "b": {}}})

    assert "b" not in {turn["handle"] for turn in result["turns"]}, result["turns"]
    assert not [turn for turn in result["turns"] if turn["reason"] == "continuation_turn"]
    assert result["reasons"][-1].startswith("settled"), result["reasons"]
    assert _watermark(db, "member-b") == _seq_of(db, "@b please confirm")
    assert _watermark(db, "member-b") < _seq_of(db, "@a status?")


def test_a_broadcast_reply_owes_nobody_a_recovery_turn(tmp_path: Path):
    """`@all` addresses the room, not a peer, so it can leave no handoff outstanding.

    Same shape as the accepted old-citation recovery vector, with the citation replaced by a
    broadcast: if `@all` were expanded into citations, the quiet second discussion would dispatch
    a recovery turn exactly where the explicit-citation vector does.
    """
    db = tmp_path / "state.db"
    result = _drive(db, _room(db, "a", "b"), {
        "user": ["@a lead this", "@a status?"],
        "replies": {"a": {"1": ["@all thanks everyone"], "2": ["(pass)"]}, "b": {}}})

    assert not [turn for turn in result["turns"] if turn["reason"] == "continuation_turn"], (
        result["turns"])
    assert result["reasons"][-1].startswith("settled"), result["reasons"]


def test_an_earlier_continuation_reconstructs_after_its_citation_is_resolved(tmp_path: Path):
    """A recovery turn planned at an old frontier keeps its exact identity as the room moves on.

    Its own reply both answers the citation that earned it and cites somebody new, so today's
    citation state genuinely differs from the state that task was planned against. After a cold
    rebuild it must still reconstruct byte for byte, never be re-minted under the new frontier.
    """
    db = tmp_path / "state.db"
    room = _room(db, "a", "b", "c")
    checkpoint = HostedRoomPolicyCheckpoint(db)

    def plan():
        latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
        checkpoint.sync(room_id=ROOM_ID, latest_seq=latest)
        snapshot = checkpoint.snapshot(room_id=ROOM_ID, latest_seq=latest)
        return snapshot, discussion.plan_next_task(
            room, list(snapshot.events), local_profiles=LOCAL_PROFILES,
            initial_watermarks=snapshot.watermarks, held_member_ids=snapshot.held_member_ids,
            initial_citations=snapshot.citations, freeze_input_context=True)

    def settle(task, text):
        for event in discussion.plan_publication(
            room, _events(db), task, status="settled", result={"text": text},
            local_profiles=LOCAL_PROFILES,
        ).events:
            hosted_rooms.append_event(db, **event.append_kwargs(ROOM_ID), now=time.time())

    _user(db, event_id="user-1", text="@a lead this")
    _snapshot, decision = plan()
    settle(decision.task, "@b please confirm")
    for _ in range(MAX_TURNS):  # b says nothing, so its citation stays owed and the room settles
        _snapshot, decision = plan()
        if decision.status != "task":
            _activity(db, decision)
            break
        settle(decision.task, "(pass)")
    else:
        raise AssertionError("the first discussion never reached a terminal decision")

    _user(db, event_id="user-2", text="@a status?")
    _snapshot, decision = plan()
    assert decision.reason == "member_turn" and decision.task.member.handle == "a"
    settle(decision.task, "(pass)")
    frontier, decision = plan()
    assert decision.reason == "continuation_turn" and decision.task.member.handle == "b"
    earlier = decision.task

    # The recovery reply resolves b's own citation and hands off to c.
    settle(earlier, "@c can you take it from here?")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=3")
    cold = HostedRoomPolicyCheckpoint(db)
    latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
    cold.sync(room_id=ROOM_ID, latest_seq=latest)

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        live = {
            str(row["member_id"]): (int(row["cited_at_seq"]), int(row["last_post_seq"]))
            for row in conn.execute(
                """SELECT member_id, cited_at_seq, last_post_seq FROM hosted_room_policy_citations
                   WHERE room_id=? AND thread_id=?""", (ROOM_ID, THREAD_ID))}

    # Not a vacuous comparison: today's citation state is genuinely past the earlier frontier.
    assert live != dict(frontier.citations)
    assert live["member-c"][0] > 0, live
    assert live["member-b"][1] > live["member-b"][0], live

    reconstructed = discussion.reconstruct_task_plan(
        hosted_rooms.room_state(db, room_id=ROOM_ID),
        cold.events_for_task(
            room_id=ROOM_ID,
            source_event_seq=earlier.payload["source_event_seq"],
            input_context=earlier.payload.get("input_context"),
            task_id=earlier.identity.task_id),
        {"identity": earlier.identity, "payload": earlier.payload},
        local_profiles=LOCAL_PROFILES)

    assert reconstructed.identity == earlier.identity
    assert dict(reconstructed.payload) == dict(earlier.payload)
    assert reconstructed.round_index == earlier.round_index
    assert ".c1" in reconstructed.identity.turn_id

    # Reconstruction reads the STORED coordinate and prompt, so it would pass even with a broken
    # baseline. Re-plan instead: the rebuilt as-of baseline, the historical watermarks and holds,
    # and the events as they stood at that frontier must select the same member for the same
    # recovery phase and mint the same task.
    cold_snapshot = cold.snapshot(room_id=ROOM_ID, latest_seq=latest)

    assert dict(cold_snapshot.citations) == dict(frontier.citations)
    assert dict(cold_snapshot.citations) != live

    replanned = discussion.plan_next_task(
        room,
        [event for event in cold_snapshot.events if int(event["seq"]) <= frontier.through_seq],
        local_profiles=LOCAL_PROFILES,
        initial_watermarks=frontier.watermarks,
        held_member_ids=frontier.held_member_ids,
        initial_citations=cold_snapshot.citations,
        freeze_input_context=True)

    assert replanned.reason == "continuation_turn"
    assert replanned.task is not None and replanned.task.member.handle == "b"
    assert replanned.task.identity == earlier.identity
    assert dict(replanned.task.payload) == dict(earlier.payload)


@pytest.mark.parametrize("scenario", VECTORS["scenarios"], ids=lambda scenario: scenario["name"])
def test_hosted_rounds_match_the_native_traces(tmp_path: Path, scenario: dict):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Traces",
        members=[ALL_MEMBERS[handle] for handle in scenario["members"]],
        authority_gateway_id=GATEWAY_ID, now=1)

    actual = _drive(db, room, scenario)

    assert actual["speakers"] == scenario["speakers"], actual["reasons"]

    if scenario.get("dispatch"):
        # Who was asked for a turn, passes included: a silent recovery turn is invisible in the
        # committed log, so the speaker trace alone cannot tell the two engines apart.
        assert [turn["handle"] for turn in actual["turns"]] == scenario["dispatch"], actual["turns"]

    if scenario.get("phases"):
        # The same coordinates the native drive reported for each dispatch: which member, whether
        # it was an ordinary slot or the silent round's recovery, and the round it belonged to.
        phase_of = {"member_turn": "ordinary", "continuation_turn": "continuation"}
        assert [
            [turn["handle"], phase_of[turn["reason"]], turn["round"]]
            for turn in actual["turns"]
        ] == [list(entry) for entry in scenario["phases"]], actual["turns"]

    if scenario.get("unread"):
        # The same claim the native room's own watermarks carry: the held member's skips
        # consumed the earlier entry, and the later one is still waiting for it.
        unread = scenario["unread"]
        member_id = ALL_MEMBERS[unread["member"]]["member_id"]
        assert _watermark(db, member_id) >= _seq_of(db, unread["read"])
        assert _watermark(db, member_id) < _seq_of(db, unread["waiting"])

    if scenario.get("outcome"):
        status = actual["reasons"][-1].split(":")[0]
        assert {"settled": "settled", "bounded": "capped"}[status] == scenario["outcome"], (
            actual["reasons"])

    # Weaker than it looks, and deliberately so: this says every dispatched turn had its own
    # identity and that each coordinate agrees with the round it was planned in, recovery turns
    # included. Whether an EARLIER coordinate survives a later citation is a reconstruction
    # question, covered where the projection is rebuilt.
    assert len({turn["task_id"] for turn in actual["turns"]}) == len(actual["turns"])
    for turn in actual["turns"]:
        assert f".r{turn['round']}." in turn["turn_id"], turn

    resumed = scenario.get("resumed_delta")
    if resumed:
        delivered = [prompt for handle, prompt in actual["deltas"] if handle == resumed["member"]]
        assert delivered, "the released member never received a turn"
        for fragment in resumed["must_contain"]:
            assert fragment in delivered[-1]
