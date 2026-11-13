"""Friendly @mention aliases in hosted Group Chats (#89720).

A member's renamed tag is its frozen ``display_name`` in the room descriptor: the immutable
technical ``handle`` (``default-device-a1b2c3d4``) stays the canonical tag and the actor id,
while ``@ash`` addresses that member with no roster rewrite.
"""

from __future__ import annotations

import time

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms

PROFILES = ("default", "ops", "review")
TECHNICAL_HANDLE = "default-device-a1b2c3d4"

ROSTER = [
    {"member_id": "member-ash", "profile": "default", "handle": TECHNICAL_HANDLE, "display_name": "Ash"},
    {"member_id": "member-ops", "profile": "ops", "handle": "ops", "display_name": "Ops"},
    {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
]


def _roster(rows=None) -> tuple[discussion.DiscussionMember, ...]:
    return discussion.validate_roster(rows or ROSTER, local_profiles=PROFILES)


def test_a_member_is_tagged_by_its_frozen_display_name():
    members = _roster()

    assert [m.member_id for m in discussion.resolve_mentions(("@ash",), members, default_all=False)] == [
        "member-ash"
    ]
    assert [m.member_id for m in discussion.resolve_mentions(("@Ash",), members, default_all=False)] == [
        "member-ash"
    ]


def test_the_canonical_handle_still_wins_over_another_members_alias():
    # ``review`` is review's handle; Ash's display name "Review" would otherwise steal it.
    members = _roster([
        {"member_id": "member-ash", "profile": "default", "handle": TECHNICAL_HANDLE, "display_name": "Review"},
        {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
    ])

    assert [m.member_id for m in discussion.resolve_mentions(("@review",), members, default_all=False)] == [
        "member-review"
    ]


def test_reserved_tags_are_never_somebodys_alias():
    members = _roster([
        {"member_id": "member-ash", "profile": "default", "handle": TECHNICAL_HANDLE, "display_name": "Everyone"},
        {"member_id": "member-ops", "profile": "ops", "handle": "ops", "display_name": "Hermes"},
        {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
    ])

    assert discussion.resolve_mentions(("@everyone",), members, default_all=False) == members
    assert discussion.resolve_mentions(("@hermes",), members, default_all=False) == ()


def test_an_ambiguous_alias_neither_picks_one_member_nor_broadcasts_to_all():
    members = _roster([
        {"member_id": "member-one", "profile": "default", "handle": TECHNICAL_HANDLE, "display_name": "Ash"},
        {"member_id": "member-two", "profile": "ops", "handle": "ops", "display_name": "Ash"},
        {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
    ])

    resolved = discussion.resolve_mention_tags(("@ash",), members, default_all=True)

    assert resolved.members == ()
    assert resolved.ambiguous == ("ash",)
    assert discussion.resolve_mentions(("@ash",), members, default_all=True) == ()


def test_a_full_display_name_keeps_its_host_label_in_the_tag():
    # "Ash · Homelab" is not "Ash": the frozen full name must not be stripped down to a
    # different bot's tag.
    members = _roster([
        {"member_id": "member-ash", "profile": "default", "handle": TECHNICAL_HANDLE, "display_name": "Ash"},
        {"member_id": "member-hosted", "profile": "ops", "handle": "ops", "display_name": "Ash · Homelab"},
        {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
    ])

    assert [m.member_id for m in discussion.resolve_mentions(("@ash-homelab",), members, default_all=False)] == [
        "member-hosted"
    ]
    # The collapsed form is the same frozen name, and it is not the other member either.
    assert [m.member_id for m in discussion.resolve_mentions(("@ashhomelab",), members, default_all=False)] == [
        "member-hosted"
    ]
    assert [m.member_id for m in discussion.resolve_mentions(("@ash",), members, default_all=False)] == [
        "member-ash"
    ]


def test_an_unknown_tag_keeps_the_existing_default():
    members = _roster()

    assert discussion.resolve_mentions(("@nobody",), members, default_all=True) == members
    assert discussion.resolve_mentions(("@nobody",), members, default_all=False) == ()


# -- real policy dispatch over a newly created room --------------------------------------------

GATEWAY_ID = "gateway-a"


@pytest.fixture
def created_room(tmp_path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db, room_id="room-1", name="Release", members=ROSTER,
        authority_gateway_id=GATEWAY_ID, now=1)
    return db, room


def _send(db, *, event_id: str, text: str) -> None:
    hosted_rooms.append_event(
        db, room_id="room-1", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "local-user"}, authority_gateway_id=GATEWAY_ID,
        authority_epoch=1, payload={"text": text, "thread_id": "thread-1"}, now=time.time())


def _events(db):
    return hosted_rooms.read_events(db, room_id="room-1", since_seq=0,
                                    limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def test_a_created_room_routes_the_friendly_tag_to_exactly_that_member(created_room):
    db, room = created_room
    _send(db, event_id="user-1", text="@ash what is the status?")

    decision = discussion.plan_next_task(room, _events(db), local_profiles=PROFILES)

    assert decision.status == "task", decision
    assert decision.task.member.member_id == "member-ash"
    # The technical handle and actor id are untouched by the friendly tag.
    assert decision.task.member.handle == TECHNICAL_HANDLE
    assert f"@{TECHNICAL_HANDLE}" in decision.task.payload["prompt"]


def test_the_prompt_tells_the_model_its_own_friendly_tag(created_room):
    db, room = created_room
    _send(db, event_id="user-1", text="@ash and @ops, status?")

    decision = discussion.plan_next_task(room, _events(db), local_profiles=PROFILES)

    assert decision.status == "task", decision
    prompt = decision.task.payload["prompt"]
    if decision.task.member.member_id == "member-ash":
        assert f"@{TECHNICAL_HANDLE} (also @ash)" in prompt
    else:
        assert "You are @ops" in prompt
        assert "(also @ops)" not in prompt  # a name that is already the handle stays one tag


def test_an_everyone_room_does_not_broadcast_a_friendly_tag(created_room):
    db, room = created_room
    _send(db, event_id="user-1", text="@ops take this one")

    decision = discussion.plan_next_task(room, _events(db), local_profiles=PROFILES)

    assert decision.status == "task", decision
    assert decision.task.member.member_id == "member-ops"


# -- lead routing and the auto router read the same tags ---------------------------------------


def _validated(room, db):
    _send(db, event_id="user-1", text="who owns this?")
    return discussion._validated_events(_events(db), room=discussion.validate_room(room, local_profiles=PROFILES))


def test_a_friendly_tag_counts_as_explicit_addressing_for_lead_routing(created_room):
    from gateway import hosted_room_lead

    db, room = created_room
    _send(db, event_id="user-1", text="@ash who owns this?")
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)
    events = discussion._validated_events(_events(db), room=discussion_room)

    assert hosted_room_lead.routes_through_lead(events[0], discussion_room, events) is False


def test_a_lead_can_assign_by_friendly_tag(created_room):
    from gateway import hosted_room_lead

    _, room = created_room
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)

    assert hosted_room_lead.parse_assignment('{"assign": ["ash"]}', discussion_room) == ("member-ash",)
    assert hosted_room_lead.parse_assignment('{"assign": ["@ops"]}', discussion_room) == ("member-ops",)


def test_the_lead_routing_prompt_lists_each_members_friendly_tag(created_room):
    from gateway import hosted_room_lead

    db, room = created_room
    _send(db, event_id="user-1", text="who owns this?")
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)
    events = discussion._validated_events(_events(db), room=discussion_room)
    lead = discussion_room.members[0]

    prompt = hosted_room_lead.routing_prompt(
        room=discussion_room, lead=lead, discussion=events[0], context_lines=[])

    assert f"@{TECHNICAL_HANDLE} (Ash, also @ash)" in prompt
    assert "@ops (Ops)" in prompt


def test_the_auto_router_answers_by_friendly_tag(created_room):
    from gateway import hosted_room_router as router

    _, room = created_room
    handles = router.router_handles(room["members"])

    assert router.parse_router_answer('{"members": ["ash"]}', handles) == ("member-ash",)
    assert router.parse_router_answer('{"members": ["@ash"]}', handles) == ("member-ash",)
    assert router.parse_router_answer('{"members": ["nobody"]}', handles) is None


def test_the_auto_router_prompt_offers_the_friendly_tag(created_room):
    from gateway import hosted_room_router as router

    _db, room = created_room
    roster = [{"handle": m["handle"], "name": m["display_name"], "description": ""} for m in room["members"]]

    prompt = router.router_prompt(
        room_name=room["name"], purpose="", notes="", roster=roster, recent=[], message="status?")

    assert f"- @{TECHNICAL_HANDLE} (Ash, also @ash)" in prompt


# -- adjudicated review regressions ---------------------------------------------------------------

RESERVED_HANDLE_ROSTER = [
    {"member_id": "default", "profile": "default", "handle": "default", "display_name": "Default Bot"},
    {"member_id": "member-hermes", "profile": "ops", "handle": "hermes"},
    {"member_id": "member-user", "profile": "review", "handle": "user"},
]

COLLIDING_ROSTER = [
    {"member_id": "member-a", "profile": "default", "handle": "ops-a", "display_name": "Same"},
    {"member_id": "member-b", "profile": "ops", "handle": "rev-b", "display_name": "Same"},
    {"member_id": "member-c", "profile": "review", "handle": "review", "display_name": "Review"},
]

CAPTURED_ROSTER = [
    {"member_id": "member-ash", "profile": "default", "handle": "ash", "display_name": "Review"},
    {"member_id": "member-review", "profile": "review", "handle": "review", "display_name": "Review"},
    {"member_id": "member-ops", "profile": "ops", "handle": "ops"},
]


def _room_with(tmp_path, rows):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db, room_id="room-1", name="Reserved", members=rows,
        authority_gateway_id=GATEWAY_ID, now=1)
    return db, room


def _validated_message(db, text):
    _send(db, event_id="user-1", text=text)
    return _events(db)


def _router_prompt(room, message="status?"):
    from gateway import hosted_room_router as router

    roster = [{"handle": m["handle"], "name": m.get("display_name") or "", "description": ""}
              for m in room["members"]]
    return router.router_prompt(
        room_name=room["name"], purpose="", notes="", roster=roster, recent=[], message=message)


def _discussion_prompt(discussion_room, member):
    return discussion._build_prompt(
        room=discussion_room, member=member, messages=(), watermark=0, seen_through_seq=0)


def test_a_reserved_word_that_is_an_actual_handle_still_addresses_that_member():
    from gateway.hosted_room_mentions import mention_index

    members = _roster(RESERVED_HANDLE_ROSTER)

    for tag, member_id in (("default", "default"), ("hermes", "member-hermes"), ("user", "member-user")):
        assert [m.member_id for m in discussion.resolve_mentions((f"@{tag}",), members, default_all=False)] == [
            member_id]
        # An explicit tag is not "no mention": the old everyone default must not fire.
        assert [m.member_id for m in discussion.resolve_mentions((f"@{tag}",), members, default_all=True)] == [
            member_id]
    index = mention_index(members)
    assert index.member("@default").member_id == "default"
    assert index.member("hermes").member_id == "member-hermes"
    assert index.member("user").member_id == "member-user"


def test_a_reserved_display_name_is_still_never_an_alias():
    members = _roster([
        {"member_id": "member-ash", "profile": "default", "handle": TECHNICAL_HANDLE,
         "display_name": "Default"},
        {"member_id": "member-ops", "profile": "ops", "handle": "ops", "display_name": "Hermes"},
        {"member_id": "member-review", "profile": "review", "handle": "review"},
    ])

    assert discussion.resolve_mentions(("@default",), members, default_all=False) == ()
    assert discussion.resolve_mentions(("@hermes",), members, default_all=False) == ()


def test_a_lead_can_assign_reserved_handle_rooms(tmp_path):
    from gateway import hosted_room_lead

    _db, room = _room_with(tmp_path, RESERVED_HANDLE_ROSTER)
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)

    assert hosted_room_lead.parse_assignment('{"assign": ["hermes"]}', discussion_room) == ("member-hermes",)
    assert hosted_room_lead.parse_assignment('{"assign": ["@default"]}', discussion_room) == ("default",)
    assert hosted_room_lead.parse_assignment('{"assign": ["@user"]}', discussion_room) == ("member-user",)


def test_colliding_titles_are_never_advertised_as_aliases(tmp_path):
    from gateway import hosted_room_lead
    from gateway import hosted_room_router as router

    _db, room = _room_with(tmp_path, COLLIDING_ROSTER)
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)
    events = discussion._validated_events(_validated_message(_db, "@ops-a status?"), room=discussion_room)

    prompts = [
        hosted_room_lead.routing_prompt(
            room=discussion_room, lead=discussion_room.members[0], discussion=events[0], context_lines=[]),
        _router_prompt(room),
    ] + [_discussion_prompt(discussion_room, member) for member in discussion_room.members]
    for prompt in prompts:
        assert "also @same" not in prompt

    handles = router.router_handles(room["members"])
    assert "same" not in handles
    assert router.parse_router_answer('{"members": ["ops-a"]}', handles) == ("member-a",)
    assert router.parse_router_answer('{"members": ["same"]}', handles) is None
    assert hosted_room_lead.parse_assignment('{"assign": ["rev-b"]}', discussion_room) == ("member-b",)
    assert hosted_room_lead.parse_assignment('{"assign": ["same"]}', discussion_room) == ()


def test_a_display_name_that_spells_a_peer_handle_is_not_advertised(tmp_path):
    from gateway import hosted_room_lead

    _db, room = _room_with(tmp_path, CAPTURED_ROSTER)
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)
    events = discussion._validated_events(_validated_message(_db, "@ash status?"), room=discussion_room)

    prompts = [
        hosted_room_lead.routing_prompt(
            room=discussion_room, lead=discussion_room.members[0], discussion=events[0], context_lines=[]),
        _router_prompt(room),
    ] + [_discussion_prompt(discussion_room, member) for member in discussion_room.members]
    for prompt in prompts:
        assert "@ash (Review, also @review)" not in prompt
    # The roster line still shows the frozen human name, just without an unsafe tag.
    assert "- @ash (Review)" in prompts[1]


def test_an_ambiguous_user_tag_bypasses_auto_and_lead_and_fans_out_to_nobody(tmp_path):
    from gateway import hosted_room_lead
    from gateway import hosted_room_router as router

    _db, room = _room_with(tmp_path, COLLIDING_ROSTER)
    discussion_room = discussion.validate_room(room, local_profiles=PROFILES)
    events = discussion._validated_events(_validated_message(_db, "@same status?"), room=discussion_room)

    assert discussion._first_round_responders(events[0], discussion_room) == ()
    assert hosted_room_lead.routes_through_lead(events[0], discussion_room, events) is False
    # The pre-existing malformed/unknown-answer router fallback is unchanged by this fix.
    assert router.routed_payload(discussion_event_id="discussion-1", thread_id="thread-1", member_ids=None) == {
        "discussion_event_id": "discussion-1", "thread_id": "thread-1",
        "member_ids": [], "everyone": True, "fallback": True}
