"""Group Chat purpose/notes and scheduled-job posts, through the real room store and Discussion policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway import hosted_room_context as context
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms

ROOM_ID = "room-ctx"
PROFILES = ("research", "build")
MEMBERS = [{"member_id": f"member-{p}", "profile": p, "handle": p} for p in PROFILES]


@pytest.fixture
def room_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Ops", members=MEMBERS,
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), now=1)
    return db


def _plan(db: Path, *, before_seq: int | None = None) -> discussion.DiscussionDecision:
    room = hosted_rooms.room_state(db, room_id=ROOM_ID)
    room["members"] = MEMBERS
    events = hosted_rooms.read_events(db, room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]
    return discussion.plan_next_task(
        room, events, local_profiles=PROFILES,
        room_context=None if before_seq is None else context.room_context(db, room_id=ROOM_ID, before_seq=before_seq))


def _post(db: Path, text: str, *, run_key: str = "run-1", notice: bool = False) -> dict:
    return context.post_scheduled_message(
        db, room_id=ROOM_ID, job_id="job42", job_name="Nightly", run_key=run_key, text=text, notice=notice)


def test_scheduled_post_without_mention_is_only_a_message(room_db: Path):
    event = _post(room_db, "All backups OK")

    assert event["actor"] == {"kind": "user", "id": "cron:job42", "display_name": "Nightly"}
    decision = _plan(room_db)
    assert (decision.status, decision.reason, decision.task) == ("settled", "silent_round", None)


def test_scheduled_post_mention_pulls_in_only_that_member(room_db: Path):
    _post(room_db, "Disk at 91% @build please look")

    decision = _plan(room_db)
    assert decision.status == "task"
    assert decision.task.member.profile == "build"
    assert 'Scheduled job "Nightly" (automated post, not the user): Disk at 91%' in decision.task.payload["prompt"]
    assert "User (user):" not in decision.task.payload["prompt"]


def test_failure_notice_never_starts_a_round(room_db: Path):
    _post(room_db, "job failed: @build @all", notice=True)

    assert _plan(room_db).status == "settled"


def test_redelivering_the_same_run_posts_once(room_db: Path):
    first, again = _post(room_db, "report"), _post(room_db, "report")

    assert again["idempotent"] and again["seq"] == first["seq"]


def test_purpose_and_notes_ride_the_user_turn_prompt_and_freeze_per_discussion(room_db: Path):
    context.set_room_context(
        room_db, room_id=ROOM_ID, event_id="ctx-1", purpose="Ship the 2.0 release",
        notes="Freeze is Friday\nOwner: @build")
    message = hosted_rooms.append_event(
        room_db, room_id=ROOM_ID, event_id="u1", kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": "status?", "thread_id": "t1"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)

    first = _plan(room_db, before_seq=message["seq"])
    prompt = first.task.payload["prompt"]
    assert "Room purpose (set by the user): Ship the 2.0 release" in prompt
    assert "  Freeze is Friday\n  Owner: @build" in prompt
    assert prompt.index("Room purpose") < prompt.index("User (user): status?")

    # An edit after the discussion opened applies from the next message: the planned turn is unchanged.
    context.set_room_context(room_db, room_id=ROOM_ID, event_id="ctx-2", purpose="Something else", notes="")
    assert _plan(room_db, before_seq=message["seq"]).task.identity == first.task.identity
    assert context.room_context(room_db, room_id=ROOM_ID) == {"purpose": "Something else", "notes": ""}


def test_room_without_context_keeps_the_existing_prompt(room_db: Path):
    message = hosted_rooms.append_event(
        room_db, room_id=ROOM_ID, event_id="u1", kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": "hi", "thread_id": "t1"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)

    assert _plan(room_db, before_seq=message["seq"]).task.identity == _plan(room_db).task.identity


def test_context_is_bounded(room_db: Path):
    with pytest.raises(hosted_rooms.HostedRoomError):
        context.set_room_context(room_db, room_id=ROOM_ID, event_id="c", purpose="x" * 501, notes="")
    with pytest.raises(hosted_rooms.HostedRoomError):
        context.set_room_context(
            room_db, room_id=ROOM_ID, event_id="c", purpose="", notes="x" * (context.MAX_NOTES_BYTES + 1))
