"""Direct user addresses must not disappear as ordinary Discussion silence."""

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from tests.gateway.test_hosted_room_discussion import (
    GATEWAY_ID,
    LOCAL_PROFILES,
    MEMBERS,
    ROOM_ID,
    _append_publication,
    _append_user,
    _events,
    _next_task,
    _settle_next,
)


@pytest.fixture
def room_db(tmp_path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db,
        room_id=ROOM_ID,
        name="Release",
        members=MEMBERS,
        authority_gateway_id=GATEWAY_ID,
        now=1,
    )
    return db, room


@pytest.mark.parametrize("address", ["@research", "@RESEARCH", "@everyone", "@all"])
@pytest.mark.parametrize("reply", ["(pass)", "", "pass."])
def test_addressed_pass_is_visible_and_does_not_schedule_a_second_attempt(
    room_db, address, reply
):
    db, room = room_db
    _append_user(db, event_id="user-1", text=f"{address} continue the task")
    task = _next_task(room, db)
    assert "concrete blocker" in task.payload["prompt"]
    assert 'reply with exactly "(pass)"' not in task.payload["prompt"]
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": reply},
        local_profiles=LOCAL_PROFILES,
    )
    assert publication.terminal_kind == "turn.failed"
    assert len(publication.events) == 1
    assert "directly addressed" in publication.events[0].payload["error"]
    _append_publication(db, publication)
    next_decision = discussion.plan_next_task(
        room, _events(db), local_profiles=LOCAL_PROFILES
    )
    assert next_decision.task is None or next_decision.task.member != task.member


@pytest.mark.parametrize("message", ["Continue the task", "@unknown continue"])
def test_unaddressed_collaborative_pass_stays_silent(room_db, message):
    db, room = room_db
    _append_user(db, event_id="user-1", text=message)
    task = _next_task(room, db)
    assert 'reply with exactly "(pass)"' in task.payload["prompt"]
    result = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    assert result.terminal_kind == "turn.settled"
    assert result.events[0].payload["passed"] is True


def test_answered_member_can_pass_on_a_later_peer_handoff(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@research check the release")
    _settle_next(room, db, text="Checked. @build please confirm.")
    _settle_next(room, db, text="Confirmed. @research anything else?")
    task = _next_task(room, db)
    assert task.member.handle == "research" and task.round_index > 0
    result = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    assert result.terminal_kind == "turn.settled"
    assert result.events[0].payload["passed"] is True


def test_superseded_addressed_pass_remains_cancelled(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@research inspect")
    task = _next_task(room, db)
    _append_user(db, event_id="user-2", text="Never mind")
    result = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    assert result.terminal_kind == "turn.cancelled"
    assert result.events[0].payload["reason"] == "superseded_by_newer_user_event"


def test_peer_member_address_is_checked_against_handle_not_profile(room_db):
    db, room = room_db
    room["members"][0] = {
        "member_id": "remote-member",
        "profile": "default",
        "handle": "remote",
        "target": {
            "kind": "peer",
            "profile": "default",
            "peer_id": "peer-1",
            "installation_id": "remote-install",
            "capability_digest": "a" * 64,
        },
    }
    _append_user(db, event_id="user-1", text="@remote report your blocker")
    task = _next_task(room, db)
    assert task.member.member_id == "remote-member"
    result = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    assert result.terminal_kind == "turn.failed"
    assert "@remote" in result.events[0].payload["error"]
