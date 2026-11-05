"""Opt-in lead routing in hosted Group Chats, through the real room store and Discussion policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway import hosted_room_context as context
from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from persistence.sqlite.gateway import hosted_rooms as backend_rooms
from persistence_assembly import open_persistence, profile_scope

ROOM_ID = "room-lead"
PROFILES = ("ada", "linus", "grace")
MEMBERS = [{"member_id": f"member-{p}", "profile": p, "handle": p} for p in PROFILES]


@pytest.fixture
def room_db(tmp_path: Path):
    db = open_persistence(profile_scope(home=tmp_path))
    hosted_rooms.create_room(db, placement="profile", room_id=ROOM_ID, name="Kernel", members=MEMBERS, authority_gateway_id=hosted_rooms.local_authority_gateway_id(), now=1)
    yield db
    db.close()


def _room(db) -> dict:
    room = hosted_rooms.room_state(db, placement="profile", room_id=ROOM_ID)
    room["members"] = MEMBERS
    return room


def _events(db) -> list[dict]:
    return hosted_rooms.read_events(db, placement="profile", room_id=ROOM_ID, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def _plan(db, source_seq: int) -> discussion.DiscussionDecision:
    return discussion.plan_next_task(
        _room(db), _events(db), local_profiles=PROFILES,
        room_context=context.room_context(db, placement="profile", room_id=ROOM_ID, before_seq=source_seq))


def _say(db, text: str, *, event_id: str = "u1", thread_id: str = "t1") -> int:
    return hosted_rooms.append_event(
        db, placement="profile", room_id=ROOM_ID, event_id=event_id, kind="message.user", actor={"kind": "user", "id": "desktop"},
        payload={"text": text, "thread_id": thread_id}, authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)["seq"]


def _finish(db, decision: discussion.DiscussionDecision, text: str, status: str = "settled") -> list[dict]:
    result = {"text": text} if status == "settled" else {"error": text}
    plan = discussion.plan_publication(
        _room(db), _events(db), decision.task, status=status, result=result, local_profiles=PROFILES)
    return [hosted_rooms.append_event(db, placement="profile", **event.append_kwargs(ROOM_ID)) for event in plan.events]


def _lead_room(db) -> None:
    context.set_room_context(
        db, placement="profile", room_id=ROOM_ID, event_id="ctx", purpose="Kernel work", notes="linus: scheduler; grace: compiler",
        routing="lead", lead_member_id="member-ada")


def test_unaddressed_message_wakes_only_who_the_lead_picks(room_db):
    _lead_room(room_db)
    seq = _say(room_db, "The build is slow, can someone look?")

    routing = _plan(room_db, seq)
    assert (routing.status, routing.reason, routing.task.member.handle) == ("task", "lead_routing", "ada")
    assert routing.task.seen_through_seq < seq
    assert '{"assign": ["handle"]}' in routing.task.payload["prompt"]
    assert "linus: scheduler" in routing.task.payload["prompt"]

    published = _finish(room_db, routing, 'Sure: {"assign": ["@Linus"]}')
    assignment = published[0]
    assert assignment["kind"] == "room.assignment"
    assert assignment["payload"]["member_ids"] == ["member-linus"]
    assert assignment["payload"]["lead_member_id"] == "member-ada" and assignment["payload"]["fallback"] is False
    # The routing answer itself is never a visible reply.
    assert [event["kind"] for event in published] == ["room.assignment", "turn.settled"]
    assert published[1]["payload"]["passed"] is True

    turn = _plan(room_db, seq)
    assert (turn.reason, turn.task.member.handle) == ("member_turn", "linus")
    assert "The room lead @ada gave the user's message to you." in turn.task.payload["prompt"]
    assert "User (user): The build is slow" in turn.task.payload["prompt"]
    _finish(room_db, turn, "On it: the scheduler is thrashing.")
    assert (_plan(room_db, seq).status, _plan(room_db, seq).reason) == ("settled", "silent_round")


@pytest.mark.parametrize("answer", ["Linus should take it", '{"assign": ["@all"]}', '{"assign": []}', "(pass)"])
def test_a_routing_answer_naming_no_member_wakes_the_lead_alone_never_everyone(room_db, answer: str):
    _lead_room(room_db)
    seq = _say(room_db, "Anyone?")

    assignment = _finish(room_db, _plan(room_db, seq), answer)[0]
    assert assignment["payload"]["member_ids"] == ["member-ada"] and assignment["payload"]["fallback"] is True
    # The lead's routing receipt does not consume its own round-0 turn.
    turn = _plan(room_db, seq)
    assert turn.task.member.handle == "ada" and turn.task.seen_through_seq >= seq
    _finish(room_db, turn, "(pass)")
    assert _plan(room_db, seq).status == "settled"


def test_a_failed_routing_turn_falls_back_to_the_lead(room_db):
    _lead_room(room_db)
    seq = _say(room_db, "Anyone?")

    published = _finish(room_db, _plan(room_db, seq), "provider down", status="failed")
    assert [event["kind"] for event in published] == ["room.assignment", "turn.failed"]
    assert _plan(room_db, seq).task.member.handle == "ada"


@pytest.mark.parametrize("answer", [
    'Pick: {"assign": ["linus"]} (scheduler work, see {notes})',
    'Thinking {not json} then {"assign": ["@linus"]}',
    '{"reason": "x"} {"assign": ["linus"]}'])
def test_a_routing_answer_wrapped_in_prose_with_braces_still_assigns(room_db, answer: str):
    _lead_room(room_db)
    seq = _say(room_db, "The build is slow")

    assignment = _finish(room_db, _plan(room_db, seq), answer)[0]
    assert assignment["payload"]["member_ids"] == ["member-linus"] and assignment["payload"]["fallback"] is False


def test_a_discarded_routing_turn_records_no_assignment_and_the_lead_still_answers(room_db):
    _lead_room(room_db)
    seq = _say(room_db, "Anyone?")

    published = _finish(room_db, _plan(room_db, seq), "discarded by the user", status="cancelled")
    assert [event["kind"] for event in published] == ["turn.cancelled"]
    # Discard gives up the attempt, not the user's message: the lead takes it alone, exactly once.
    turn = _plan(room_db, seq)
    assert (turn.reason, turn.task.member.handle) == ("member_turn", "ada") and turn.task.seen_through_seq >= seq
    _finish(room_db, turn, "Looking now.")
    assert _plan(room_db, seq).status == "settled"


def test_a_purpose_only_save_racing_a_routing_change_never_restores_the_old_routing(room_db, monkeypatch):
    import threading

    _lead_room(room_db)
    real_validate = backend_rooms.validate_context
    first_read, routing_saved = threading.Event(), threading.Event()

    def validate(*args, **kwargs):
        if threading.current_thread().name == "older-client" and not first_read.is_set():
            first_read.set()
            # Hold the older client right after it read the current routing. Unless that read sits
            # under the write lock, the routing change commits meanwhile and would be written back stale.
            routing_saved.wait(timeout=1.0)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(backend_rooms, "validate_context", validate)
    older = threading.Thread(name="older-client", target=lambda: context.set_room_context(
        room_db, placement="profile", room_id=ROOM_ID, event_id="c-older", purpose="Older client purpose", notes=""))
    older.start()
    assert first_read.wait(timeout=5)
    context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-routing", purpose="Kernel work", notes="",
                             routing="everyone")
    routing_saved.set()
    older.join(timeout=10)

    assert context.room_context(room_db, placement="profile", room_id=ROOM_ID)["routing"] == "everyone"


@pytest.mark.parametrize(("text", "first"), [("@grace what broke?", "grace"), ("@everyone status", "ada")])
def test_mentions_and_everyone_bypass_the_lead(room_db, text: str, first: str):
    _lead_room(room_db)
    seq = _say(room_db, text)

    decision = _plan(room_db, seq)
    assert (decision.reason, decision.task.member.handle) == ("member_turn", first)
    assert decision.task.seen_through_seq >= seq


def test_scheduled_posts_and_thread_replies_keep_their_resolution(room_db):
    _lead_room(room_db)
    post = context.post_scheduled_message(
        room_db, placement="profile", room_id=ROOM_ID, job_id="job1", job_name="Nightly", run_key="r1", text="all green")
    settled = _plan(room_db, post["seq"])
    assert settled.status == "settled"
    hosted_rooms.append_event(  # what the room service records for a settled discussion
        room_db, placement="profile", room_id=ROOM_ID, event_id="activity-1", kind="room.activity",
        actor={"kind": "gateway", "id": hosted_rooms.local_authority_gateway_id()},
        payload={"status": "settled", "reason_code": settled.reason, "thread_id": settled.thread_id,
                 "discussion_event_id": settled.discussion_event_id},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)

    first = _say(room_db, "@linus kick it", event_id="u1", thread_id="t2")
    _finish(room_db, _plan(room_db, first), "Kicked.")
    reply = _say(room_db, "thanks, anything else?", event_id="u2", thread_id="t2")
    decision = _plan(room_db, reply)
    # A reply inside a thread addresses everyone, as before: no routing turn.
    assert (decision.reason, decision.task.member.handle) == ("member_turn", "ada")
    assert decision.task.seen_through_seq >= reply


def test_everyone_routing_plans_exactly_as_a_room_without_settings(room_db):
    _say(room_db, "hello")
    plans = [discussion.plan_next_task(_room(room_db), _events(room_db), local_profiles=PROFILES, room_context=value)
             for value in (None, {"purpose": "", "notes": "", "routing": "everyone", "lead_member_id": ""})]

    assert plans[0].task.identity == plans[1].task.identity and plans[0].task.member.handle == "ada"


def test_room_settings_validate_the_lead_and_keep_routing_a_client_omits(room_db):
    with pytest.raises(hosted_rooms.HostedRoomError):
        context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c1", purpose="", notes="", routing="lead",
                                 lead_member_id="member-nobody")
    with pytest.raises(hosted_rooms.HostedRoomError):
        context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c1", purpose="", notes="", routing="lead")
    _lead_room(room_db)

    # An older Desktop saves purpose/notes only: the room stays a lead room.
    context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c2", purpose="New purpose", notes="")
    assert context.room_context(room_db, placement="profile", room_id=ROOM_ID) == {
        "purpose": "New purpose", "notes": "", "routing": "lead", "lead_member_id": "member-ada"}


def test_attachment_only_message_tells_the_lead_what_arrived():
    room = {"room_id": "room", "name": "Files", "authority_gateway_id": "gw", "authority_epoch": 1,
            "members": [{"member_id": p, "profile": p, "handle": p} for p in ("alice", "bob")]}
    files = [{"attachment_id": f"att_{n:032x}", "kind": kind, "name": name, "size": 1, "mime": mime}
             for n, (kind, name, mime) in enumerate([("file", "crash.log", "text/plain"), ("image", "trace.png", "image/png")])]
    events = [{"room_id": "room", "seq": 1, "event_id": "event-1", "kind": "message.user", "authority_epoch": 1,
               "actor": {"kind": "user", "id": "user"}, "payload": {"text": "", "thread_id": "t", "attachments": files}}]

    routing = discussion.plan_next_task(room, events, local_profiles=("alice", "bob"),
                                        room_context={"routing": "lead", "lead_member_id": "alice"})

    assert (routing.reason, routing.task.member.member_id) == ("lead_routing", "alice")
    prompt = routing.task.payload["prompt"]
    assert "crash.log (file, text/plain)" in prompt and "trace.png (image, image/png)" in prompt


def test_retried_settings_save_returns_its_first_event_after_a_routing_edit(room_db):
    first = context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-notes", purpose="Notes only", notes="")
    _lead_room(room_db)  # another client turns on a lead meanwhile

    retry = context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-notes", purpose="Notes only", notes="")

    assert retry["event"]["idempotent"] and retry["event"]["seq"] == first["event"]["seq"]
    assert context.room_context(room_db, placement="profile", room_id=ROOM_ID)["routing"] == "lead"
    with pytest.raises(hosted_rooms.EventConflictError):  # a different save under the same id still conflicts
        context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-notes", purpose="Other", notes="")



def test_a_settings_retry_of_an_event_saved_before_routing_existed_returns_it(room_db):
    """A ``room.context_set`` written before lead routing carries only purpose and notes."""
    first = hosted_rooms.append_event(
        room_db, placement="profile", room_id=ROOM_ID, event_id="c-old", kind=context.CONTEXT_EVENT_KIND,
        actor={"kind": "system", "id": "room-control"}, payload={"purpose": "Old", "notes": ""},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)

    retry = context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-old", purpose="Old", notes="")

    assert retry["event"]["idempotent"] and retry["event"]["seq"] == first["seq"]
    with pytest.raises(hosted_rooms.EventConflictError):  # asking for a lead under that id is a different save
        context.set_room_context(room_db, placement="profile", room_id=ROOM_ID, event_id="c-old", purpose="Old", notes="",
                                 routing="lead", lead_member_id="member-ada")


@pytest.mark.parametrize(("answer", "status"), [('{"assign": ["ada"]}', "settled"), ("discarded", "cancelled")])
def test_a_rename_after_routing_never_consumes_the_leads_answer(room_db, answer: str, status: str):
    """The routing receipt is matched by its durable identity, not by a prompt that names the room."""
    _lead_room(room_db)
    seq = _say(room_db, "Anyone?")
    _finish(room_db, _plan(room_db, seq), answer, status=status)
    hosted_rooms.rename_room(room_db, placement="profile", room_id=ROOM_ID, event_id="rename", name="Renamed")

    turn = _plan(room_db, seq)
    assert (turn.status, turn.reason, turn.task.member.handle) == ("task", "member_turn", "ada")
    assert "lead_routing" not in turn.task.payload

@pytest.mark.parametrize("routing", ["everyone", "lead"])
def test_batch_bounded_member_turn_publishes_its_real_reply(routing: str):
    """A member turn whose frozen batch stops below its source message is still a member turn, not routing."""
    room = {"room_id": "room", "name": "Files", "authority_gateway_id": "gw", "authority_epoch": 1,
            "members": [{"member_id": p, "profile": p, "handle": p} for p in ("alice", "bob")]}
    files = [[{"attachment_id": f"att_{i * 10 + n:032x}", "kind": "file", "name": "f.txt", "size": 1,
               "mime": "text/plain"} for n in range(8)] for i in range(1, 4)]
    events = [{"room_id": "room", "seq": i, "event_id": f"event-{i}", "kind": "message.user", "authority_epoch": 1,
               "actor": {"kind": "user", "id": "user"},
               "payload": {"text": "@alice", "thread_id": "thread", "attachments": files[i - 1]}} for i in range(1, 4)]
    settings = {"routing": routing, "lead_member_id": "alice" if routing == "lead" else ""}
    task = discussion.plan_next_task(room, events, local_profiles=("alice", "bob"), freeze_input_context=True,
                                     room_context=settings).task
    assert (task.member.member_id, task.seen_through_seq) == ("alice", 2)
    publication = discussion.plan_publication(
        room, events, task, status="settled", result={"text": "Review @bob"}, local_profiles=("alice", "bob"))
    kinds = [effect.kind for effect in publication.events]
    assert "room.assignment" not in kinds
    message = next(effect for effect in publication.events if effect.kind == "message.member")
    assert message.payload["text"] == "Review @bob"
