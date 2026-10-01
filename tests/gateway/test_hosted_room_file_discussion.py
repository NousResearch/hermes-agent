"""Discussion policy for shared files: the turn's rule, its one member message, and the next turn."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms

ROOM_ID = "room-1"
GATEWAY_ID = "gateway-a"
LOCAL_PROFILES = ("research", "build", "review")
MEMBERS = [{"member_id": f"member-{p}", "profile": p, "handle": p} for p in LOCAL_PROFILES]
RULE = "- To hand off a local file, call share_group_file; never paste a local path into chat."
BASE_RULES = [
    "Rules for this Discussion:",
    "- Reply with one conversational message only when you have something new worth adding.",
    '- If you have nothing new to add, reply with exactly "(pass)".',
    "- Mention a teammate by handle to pull them into the next round; do not repeat points already made.",
    "- Never reveal content from private conversations. Your reply is published verbatim.",
]


@pytest.fixture
def room_db(tmp_path: Path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(db, room_id=ROOM_ID, name="Release", members=MEMBERS,
                                    authority_gateway_id=GATEWAY_ID, now=1)
    return db, room


def _events(db):
    return hosted_rooms.read_events(db, room_id=ROOM_ID, since_seq=0, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def _user(db, text, event_id="user-1"):
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="message.user", actor={"kind": "user", "id": "local-user"},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1, payload={"text": text, "thread_id": "thread-1"},
        now=time.time())


def _task(room, db):
    decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert decision.status == "task"
    return decision.task


def _attachments(db, names):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    store = HostedRoomAttachmentStore(db)
    staged = []
    for index, name in enumerate(names):
        saved = store.put(room_id=ROOM_ID, upload_id=f"upload-{index}", kind="file", name=name,
                          mime="text/plain", data=f"{name} body\n".encode())
        staged.append({key: saved[key] for key in ("attachment_id", "kind", "name", "size", "mime")})
    return staged


def _rules(prompt):
    lines = prompt.splitlines()
    return lines[lines.index("Rules for this Discussion:"):]


def test_the_file_rule_is_added_for_local_members_and_nothing_else_changes(room_db):
    db, room = room_db
    _user(db, "@build Draft the plan.")
    rules = _rules(_task(room, db).payload["prompt"])
    assert rules == [*BASE_RULES[:4], RULE, BASE_RULES[4]]


def test_a_member_served_by_another_gateway_gets_no_file_rule(tmp_path):
    member = dict(member_id="member-peer", profile="review", handle="peer", target={
        "kind": "peer", "peer_id": "peer-1", "installation_id": "install-peer", "profile": "review",
        "capability_digest": "a" * 64})
    room = discussion.validate_room(
        {"room_id": ROOM_ID, "name": "Release", "authority_gateway_id": GATEWAY_ID, "authority_epoch": 1,
         "members": [MEMBERS[0], member]}, local_profiles=LOCAL_PROFILES)
    prompt = discussion._build_prompt(room=room, member=room.members[1], messages=[], watermark=0,
                                      seen_through_seq=0)
    assert _rules(prompt) == BASE_RULES


@pytest.mark.parametrize(("text", "expected"), [
    ("Here is the plan.", "Here is the plan."),
    ("(pass)", "Shared plan.md, notes.txt."),
    ("", "Shared plan.md, notes.txt."),
])
def test_shared_files_ride_on_the_turns_one_member_message(room_db, text, expected):
    db, room = room_db
    _user(db, "@build Draft the plan.")
    task = _task(room, db)
    attachments = _attachments(db, ["plan.md", "notes.txt"])
    recipients = [member["member_id"] for member in MEMBERS]
    publication = discussion.plan_publication(
        room, _events(db), task, status="settled", local_profiles=LOCAL_PROFILES,
        result={"text": text, "attachments": attachments, "recipient_member_ids": recipients})
    message, terminal = publication.events
    assert message.kind == "message.member" and terminal.kind == "turn.settled"
    assert message.payload["text"] == expected
    assert message.payload["attachments"] == attachments
    assert message.payload["recipient_member_ids"] == recipients
    assert terminal.payload["passed"] is False


def test_without_files_a_settled_reply_is_planned_exactly_as_before(room_db):
    db, room = room_db
    _user(db, "@build Draft the plan.")
    task = _task(room, db)
    message, _terminal = discussion.plan_publication(
        room, _events(db), task, status="settled", result={"text": "Plan ready."},
        local_profiles=LOCAL_PROFILES).events
    assert set(message.payload) == {"discussion_event_id", "member_id", "member_index", "round_index", "task_id",
                                    "thread_id", "turn_id", "text"}
    passed = discussion.plan_publication(room, _events(db), task, status="settled", result={"text": "(pass)"},
                                         local_profiles=LOCAL_PROFILES)
    assert [event.kind for event in passed.events] == ["turn.settled"]


def test_the_next_mentioned_bot_receives_the_shared_files_but_the_author_does_not(room_db):
    db, room = room_db
    _user(db, "@build Draft the plan.")
    task = _task(room, db)
    attachments = _attachments(db, ["plan.md"])
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    publication = discussion.plan_publication(
        room, _events(db), task, status="settled", local_profiles=LOCAL_PROFILES,
        result={"text": "@review Please check the plan.", "attachments": attachments,
                "recipient_member_ids": [member["member_id"] for member in MEMBERS]})
    HostedRoomAttachmentStore(db).commit_message(
        room_id=ROOM_ID, event_id=publication.events[0].event_id, manifest=attachments,
        recipient_member_ids=[member["member_id"] for member in MEMBERS], viewer_access=True, hold_until_event=True)
    for event in publication.events:
        hosted_rooms.append_event(db, **event.append_kwargs(ROOM_ID), now=time.time())
    following = _task(room, db)
    assert following.member.member_id == "member-review"
    assert [item["name"] for item in following.payload["attachments"]] == ["plan.md"]
    events = discussion._validated_events(_events(db), room=discussion.validate_room(
        room, local_profiles=LOCAL_PROFILES))
    message = next(event for event in events if event.kind == "message.member")
    assert discussion._event_attachments(message, task.member) == []


def test_a_bots_own_replies_never_appear_in_its_next_prompt(room_db):
    """The watermark normally covers them; the prompt also drops any that sit past it."""
    db, room = room_db
    _user(db, "@build Draft the plan.")
    task = _task(room, db)
    for event in discussion.plan_publication(room, _events(db), task, status="settled",
                                             result={"text": "My own earlier draft."},
                                             local_profiles=LOCAL_PROFILES).events:
        hosted_rooms.append_event(db, **event.append_kwargs(ROOM_ID), now=time.time())
    _user(db, "@build Now the details.", event_id="user-2")
    checked = discussion.validate_room(room, local_profiles=LOCAL_PROFILES)
    messages = [e for e in discussion._validated_events(_events(db), room=checked)
                if e.kind in {"message.user", "message.member"}]
    member = next(m for m in checked.members if m.member_id == "member-build")
    prompt = discussion._build_prompt(room=checked, member=member, messages=messages, watermark=0,
                                      seen_through_seq=messages[-1].seq)
    assert "Now the details." in prompt and "Draft the plan." in prompt
    assert "My own earlier draft." not in prompt
    other = next(m for m in checked.members if m.member_id == "member-review")
    assert "My own earlier draft." in discussion._build_prompt(
        room=checked, member=other, messages=messages, watermark=0, seen_through_seq=messages[-1].seq)
