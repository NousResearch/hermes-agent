"""Behavior tests for deterministic same-gateway Discussion policy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms


ROOM_ID = "room-1"
GATEWAY_ID = "gateway-a"
LOCAL_PROFILES = ("research", "build", "review", "ops", "qa", "docs")
MEMBERS = [
    {
        "member_id": f"member-{profile}",
        "profile": profile,
        "handle": profile,
        "display_name": profile.title(),
    }
    for profile in LOCAL_PROFILES[:3]
]


@pytest.fixture
def room_db(tmp_path: Path) -> tuple[Path, dict]:
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


def _events(db: Path) -> list[dict]:
    return hosted_rooms.read_events(
        db,
        room_id=ROOM_ID,
        since_seq=0,
        limit=hosted_rooms.MAX_LOG_LIMIT,
    )["events"]


def _append_user(
    db: Path,
    *,
    event_id: str,
    text: str,
    thread_id: str = "thread-1",
) -> dict:
    return hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "local-user"},
        authority_gateway_id=GATEWAY_ID,
        authority_epoch=1,
        payload={"text": text, "thread_id": thread_id},
        now=time.time(),
    )


def _append_publication(
    db: Path,
    plan: discussion.PublicationPlan,
) -> list[dict]:
    return [
        hosted_rooms.append_event(
            db,
            **event.append_kwargs(ROOM_ID),
            now=time.time(),
        )
        for event in plan.events
    ]


def _append_activity(
    db: Path,
    *,
    event_id: str,
    discussion_event_id: str,
    thread_id: str,
) -> dict:
    return hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id=event_id,
        kind="room.activity",
        actor={"kind": "gateway", "id": GATEWAY_ID},
        payload={
            "status": "settled",
            "reason_code": "silent_round",
            "thread_id": thread_id,
            "discussion_event_id": discussion_event_id,
        },
        authority_gateway_id=GATEWAY_ID,
        authority_epoch=1,
    )


def _next_task(room: dict, db: Path) -> discussion.DiscussionTaskPlan:
    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "task", decision
    assert decision.task is not None
    return decision.task


def _settle_next(
    room: dict,
    db: Path,
    *,
    text: str,
) -> discussion.DiscussionTaskPlan:
    task = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": text},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, publication)
    return task


def test_deferred_member_allows_next_mentioned_member_and_later_terminal_result(
    room_db,
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Report.")
    first = _next_task(room, db)
    deferred = discussion.plan_publication(
        room,
        _events(db),
        first,
        status="deferred",
        result={"reason": "member_unavailable"},
        execution_generation=1,
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, deferred)

    second = _next_task(room, db)
    assert second.member.member_id != first.member.member_id

    settled = discussion.plan_publication(
        room,
        _events(db),
        first,
        status="settled",
        result={"text": "Recovered on explicit retry."},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, settled)
    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "task"
    assert decision.task is not None
    assert decision.task.member.member_id == second.member.member_id


def test_distinct_threads_are_planned_fifo_without_skipping(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="First", thread_id="thread-1")
    _append_user(db, event_id="user-2", text="Second", thread_id="thread-2")

    first = _next_task(room, db)
    assert first.discussion_event_id == "user-1"
    _append_activity(
        db,
        event_id="activity-1",
        discussion_event_id="user-1",
        thread_id="thread-1",
    )
    second = _next_task(room, db)
    assert second.discussion_event_id == "user-2"


def test_room_stop_fences_old_work_but_allows_a_later_message(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="First", thread_id="thread-1")
    stop = hosted_rooms.request_room_stop(
        db,
        room_id=ROOM_ID,
        cancel_id="user-stop-1",
        expected_gateway_id=str(room["authority_gateway_id"]),
        expected_epoch=int(room["authority_epoch"]),
    )
    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "idle"
    assert stop["kind"] == "room.stop_requested"

    _append_user(db, event_id="user-2", text="Continue", thread_id="thread-2")
    resumed = _next_task(room, db)
    assert resumed.discussion_event_id == "user-2"


def test_deterministic_task_fits_existing_driver_and_reconstructs_after_restart(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    user = _append_user(db, event_id="user-1", text="Check the release.")

    first = _next_task(room, db)
    repeated = _next_task(room, db)
    assert first == repeated
    assert first.identity.thread_id == "thread-1"
    assert first.payload == {
        "target_member_id": "member-research",
        "target_profile": "research",
        "prompt": first.payload["prompt"],
        "source_event_seq": user["seq"],
    }
    assert set(first.payload) == {
        "target_member_id",
        "target_profile",
        "prompt",
        "source_event_seq",
    }

    admitted = driver.admit_task(
        db,
        first.identity,
        payload=first.payload,
        clock=time.time,
    )
    stored = driver.get_task(db, first.identity)
    reconstructed = discussion.reconstruct_task_plan(
        room,
        _events(db),
        stored,
        local_profiles=LOCAL_PROFILES,
    )
    assert admitted["status"] == "queued"
    assert reconstructed == first

    reopened_events = _events(db)
    assert (
        discussion.reconstruct_task_plan(
            room,
            reopened_events,
            driver.get_task(db, first.identity),
            local_profiles=LOCAL_PROFILES,
        )
        == first
    )


@pytest.mark.parametrize(
    ("text", "expected_profile"),
    [
        ("@build please inspect this", "build"),
        ("@all inspect this", "research"),
        ("@everyone inspect this", "research"),
        ("inspect this", "research"),
        ("@unknown inspect this", "research"),
    ],
)
def test_mentions_select_handles_or_everyone(
    room_db: tuple[Path, dict],
    text: str,
    expected_profile: str,
):
    db, room = room_db
    _append_user(db, event_id="user-1", text=text)

    assert _next_task(room, db).member.profile == expected_profile


def test_member_mention_joins_the_next_round_not_the_current_round(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@research lead this")

    first = _settle_next(room, db, text="@build can add the implementation detail.")
    second = _next_task(room, db)

    assert first.member.profile == "research"
    assert first.round_index == 0
    assert second.member.profile == "build"
    assert second.round_index == 1
    assert "@research lead this" in second.payload["prompt"]
    assert "@build can add the implementation detail." in second.payload["prompt"]


def test_plain_member_reply_does_not_wake_another_bot_round(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@research answer the user")
    _settle_next(room, db, text="The answer is ready for the user.")

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )

    assert decision.status == "settled"
    assert decision.reason == "silent_round"


@pytest.mark.parametrize("value", ["", "pass", "pass.", "(pass)", " ( PASS ). "])
def test_pass_detection(value: str):
    assert discussion.is_pass_text(value)


def test_real_text_is_not_a_pass():
    assert not discussion.is_pass_text("I found the issue.")


def test_full_pass_round_settles_without_member_messages(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Any concerns?")

    for _member in MEMBERS:
        _settle_next(room, db, text="(pass)")

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "settled"
    assert decision.reason == "silent_round"
    assert [event["kind"] for event in _events(db)].count("message.member") == 0


def test_failed_members_advance_the_round_as_silence(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Any concerns?")

    for expected in ("research", "build", "review"):
        task = _next_task(room, db)
        assert task.member.profile == expected
        publication = discussion.plan_publication(
            room,
            _events(db),
            task,
            status="failed",
            result={"error": f"{expected} unavailable"},
            local_profiles=LOCAL_PROFILES,
        )
        assert publication.terminal_kind == "turn.failed"
        assert len(publication.events) == 1
        assert publication.events[0].payload["reason_code"] == "unknown"
        _append_publication(db, publication)

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "settled"
    assert decision.reason == "silent_round"


def test_failed_publication_preserves_a_typed_actionable_reason(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Please continue.")
    task = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="failed",
        result={"error": "HTTP 401 authentication failed"},
        local_profiles=LOCAL_PROFILES,
    )
    assert publication.events[0].payload["reason_code"] == "provider_auth_or_access"


def test_failed_publication_rejects_an_untrusted_reason_code(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Please continue.")
    task = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="failed",
        result={"error": "failed", "reason_code": "invented"},
        local_profiles=LOCAL_PROFILES,
    )
    assert publication.events[0].payload["reason_code"] == "unknown"


def test_publication_is_idempotent_and_changed_result_conflicts(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Report.")
    task = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "Ready."},
        local_profiles=LOCAL_PROFILES,
    )

    first = _append_publication(db, publication)
    repeated = _append_publication(db, publication)
    assert [event["seq"] for event in first] == [event["seq"] for event in repeated]
    assert all(event["idempotent"] for event in repeated)

    changed = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "Different."},
        local_profiles=LOCAL_PROFILES,
    )
    with pytest.raises(hosted_rooms.EventConflictError):
        _append_publication(db, changed)


def test_partial_publication_replays_same_effects_before_policy_advances(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Report.")
    task = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "Ready."},
        local_profiles=LOCAL_PROFILES,
    )

    message_effect = publication.events[0]
    hosted_rooms.append_event(
        db,
        **message_effect.append_kwargs(ROOM_ID),
        now=time.time(),
    )
    assert _next_task(room, db).identity == task.identity

    replayed = discussion.plan_publication(
        room,
        _events(db),
        task,
        status="settled",
        result={"text": "Ready."},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, replayed)
    assert _next_task(room, db).member.profile == "build"


def test_watermark_excludes_a_members_old_input_and_own_reply(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Old request.")
    first = _settle_next(room, db, text="Old answer.")
    watermark = discussion.derive_member_watermarks(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )[("thread-1", first.member.member_id)]
    assert watermark == max(
        event["seq"]
        for event in _events(db)
        if event["kind"] == "message.member"
        and event["payload"]["task_id"] == first.identity.task_id
    )

    latest = _append_user(db, event_id="user-2", text="New request.")
    next_task = _next_task(room, db)
    assert next_task.member.profile == "research"
    assert next_task.payload["source_event_seq"] == latest["seq"]
    assert "New request." in next_task.payload["prompt"]
    assert "Old request." not in next_task.payload["prompt"]
    assert "Old answer." not in next_task.payload["prompt"]


def test_newer_same_thread_user_event_cancels_a_late_result(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="First request.")
    stale = _next_task(room, db)
    latest = _append_user(db, event_id="user-2", text="Second request.")

    publication = discussion.plan_publication(
        room,
        _events(db),
        stale,
        status="settled",
        result={"text": "Late stale answer."},
        local_profiles=LOCAL_PROFILES,
    )
    assert publication.terminal_kind == "turn.cancelled"
    assert [event.kind for event in publication.events] == ["turn.cancelled"]
    assert publication.events[0].payload["reason"] == "superseded_by_newer_user_event"
    _append_publication(db, publication)

    current = _next_task(room, db)
    assert current.payload["source_event_seq"] == latest["seq"]
    assert "Second request." in current.payload["prompt"]


def test_cross_thread_newer_user_does_not_discard_completed_old_reply(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="First request.", thread_id="thread-1")
    old = _next_task(room, db)
    _append_user(db, event_id="user-2", text="Other topic.", thread_id="thread-2")

    publication = discussion.plan_publication(
        room,
        _events(db),
        old,
        status="settled",
        result={"text": "Completed first topic."},
        local_profiles=LOCAL_PROFILES,
    )
    assert [event.kind for event in publication.events] == [
        "message.member",
        "turn.settled",
    ]


def test_oversized_member_reply_is_truncated_and_next_turn_stays_serviceable(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-large",
        text="u" * discussion.MAX_USER_TEXT_BYTES,
    )
    first = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        first,
        status="settled",
        result={"text": "é" * (discussion.MAX_MEMBER_TEXT_BYTES + 100)},
        local_profiles=LOCAL_PROFILES,
    )

    member_event = next(event for event in publication.events if event.kind == "message.member")
    member_text = member_event.payload["text"]
    assert len(member_text.encode("utf-8")) <= discussion.MAX_MEMBER_TEXT_BYTES
    assert member_text.endswith("share the full result as a file.]")
    _append_publication(db, publication)

    followup = _next_task(room, db)
    assert len(followup.payload["prompt"].encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert "Earlier content omitted" in followup.payload["prompt"]


def test_omission_marker_stands_in_for_a_dropped_oldest_line(room_db: tuple[Path, dict]):
    """Two maximal user messages: the newest is placed with the marker's bytes reserved, so when the oldest
    cannot take the remainder the marker leads the delta, the newest survives whole, and the prompt fits."""
    db, room = room_db
    _append_user(db, event_id="user-old", text="o" * discussion.MAX_USER_TEXT_BYTES)
    _append_user(db, event_id="user-new", text="n" * discussion.MAX_USER_TEXT_BYTES)

    prompt = _next_task(room, db).payload["prompt"]
    lines = prompt.split("\n")

    assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert "oooo" not in prompt
    marker_at = lines.index("  [Earlier content omitted to fit this turn.]")
    assert lines[marker_at + 1] == f"  User (user): {'n' * discussion.MAX_USER_TEXT_BYTES}"
    assert lines[marker_at + 2] == ""  # nothing else in the delta; the rules follow


def test_three_round_bound(room_db: tuple[Path, dict]):
    db, room = room_db
    room["members"] = MEMBERS[:2]
    _append_user(db, event_id="user-1", text="Discuss.")

    for index in range(6):
        task = _next_task(room, db)
        peer = "build" if task.member.profile == "research" else "research"
        publication = discussion.plan_publication(
            room,
            _events(db),
            task,
            status="settled",
            result={"text": f"Reply {index}. @{peer}"},
            local_profiles=LOCAL_PROFILES,
        )
        _append_publication(db, publication)

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "bounded"
    assert decision.reason == "max_rounds"


def test_ten_message_bound(tmp_path: Path):
    db = tmp_path / "state.db"
    members = [
        {
            "member_id": f"member-{profile}",
            "profile": profile,
            "handle": profile,
        }
        for profile in LOCAL_PROFILES
    ]
    room = hosted_rooms.create_room(
        db,
        room_id=ROOM_ID,
        name="Large",
        members=members,
        authority_gateway_id=GATEWAY_ID,
        now=1,
    )
    _append_user(db, event_id="user-1", text="Discuss.")

    for index in range(discussion.MAX_DISCUSSION_MESSAGES):
        _settle_next(room, db, text=f"Reply {index}. @everyone")

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )
    assert decision.status == "bounded"
    assert decision.reason == "max_messages"


def test_prompt_delta_is_bounded_to_24_message_lines(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    for index in range(30):
        _append_user(
            db,
            event_id=f"user-{index}",
            text=f"Message {index}.",
        )

    task = _next_task(room, db)
    assert task.payload["prompt"].count("User (user):") == 24
    assert "Message 5." not in task.payload["prompt"]
    assert "Message 6." in task.payload["prompt"]
    assert "Message 29." in task.payload["prompt"]


def test_attachment_payload_is_rejected_by_local_text_only_boundary():
    with pytest.raises(discussion.DiscussionValidationError, match="unknown fields"):
        discussion.validate_user_payload({
            "text": "Review.",
            "thread_id": "thread-1",
            "attachments": [{"name": "notes.txt"}],
        })


@pytest.mark.parametrize(
    ("members", "match"),
    [
        (MEMBERS[:1], "between 2 and 6"),
        (MEMBERS + MEMBERS + MEMBERS[:1], "between 2 and 6"),
        (
            [MEMBERS[0], {**MEMBERS[1], "profile": "research"}],
            "profiles must be unique",
        ),
        ([MEMBERS[0], {**MEMBERS[1], "handle": "RESEARCH"}], "handles must be unique"),
        (
            [MEMBERS[0], {**MEMBERS[1], "member_id": "MEMBER-RESEARCH"}],
            "ids must be unique",
        ),
        ([MEMBERS[0], {**MEMBERS[1], "route": {"mode": "ssh"}}], "cross-gateway"),
        ([MEMBERS[0], {**MEMBERS[1], "connectionId": "remote"}], "cross-gateway"),
        ([MEMBERS[0], {**MEMBERS[1], "profile": "missing"}], "not local"),
    ],
)
def test_malformed_or_remote_roster_is_rejected(members: list[dict], match: str):
    with pytest.raises(discussion.DiscussionValidationError, match=match):
        discussion.validate_roster(members, local_profiles=LOCAL_PROFILES)


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "hello"},
        {"text": "hello", "thread_id": "thread-1", "images": []},
        {"text": "", "thread_id": "thread-1"},
        {"text": "hello", "thread_id": "../escape"},
        {"text": ["hello"], "thread_id": "thread-1"},
    ],
)
def test_user_payload_is_exact_and_text_only(payload: dict):
    with pytest.raises(discussion.DiscussionValidationError):
        discussion.validate_user_payload(payload)


def test_malformed_log_and_task_reconstruction_fail_closed(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Report.")
    _append_user(db, event_id="user-2", text="Report again.")
    task = _next_task(room, db)
    events = _events(db)

    with pytest.raises(discussion.DiscussionValidationError, match="sequence order"):
        discussion.plan_next_task(
            room,
            list(reversed(events)),
            local_profiles=LOCAL_PROFILES,
        )

    malformed = {
        "identity": driver.TaskIdentity(
            room_id=task.identity.room_id,
            task_id="dtask:wrong",
            thread_id=task.identity.thread_id,
            turn_id=task.identity.turn_id,
        ),
        "payload": dict(task.payload),
    }
    with pytest.raises(
        discussion.DiscussionReconstructionError,
        match="deterministic reconstruction",
    ):
        discussion.reconstruct_task_plan(
            room,
            events,
            malformed,
            local_profiles=LOCAL_PROFILES,
        )


# -- designated-owner routing: the no-match fallback tier ---------------------
OWNER_PROFILE = "build"
OWNED_MEMBERS = [{**member, "owner": True} if member["profile"] == OWNER_PROFILE else member for member in MEMBERS]
PRECEDENCE_CASES = json.loads(
    (Path(__file__).with_name("hosted_room_responder_precedence.json")).read_text("utf-8"))["cases"]


def _create_room(db: Path, members: list[dict]) -> dict:
    return hosted_rooms.create_room(
        db,
        room_id=ROOM_ID,
        name="Release",
        members=members,
        authority_gateway_id=GATEWAY_ID,
        now=1,
    )


def _owned_members(owner: str | None) -> list[dict]:
    return [{**member, "owner": True} if member["profile"] == owner else member for member in MEMBERS]


def _round_zero_profiles(room: dict, db: Path) -> list[str]:
    """Drain one Discussion with every member passing; return the responders in turn order."""
    profiles: list[str] = []
    while True:
        decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
        if decision.status != "task":
            assert decision.status == "settled", decision
            return profiles
        assert decision.task is not None
        assert decision.task.round_index == 0
        profiles.append(decision.task.member.profile)
        _settle_next(room, db, text="(pass)")


def _fail_or_defer(room: dict, db: Path, task: discussion.DiscussionTaskPlan, status: str) -> None:
    publication = discussion.plan_publication(
        room,
        _events(db),
        task,
        status=status,
        result={"reason": "member_unavailable"} if status == "deferred" else {"error": "owner crashed"},
        execution_generation=1 if status == "deferred" else None,
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, publication)


@pytest.mark.parametrize("case", PRECEDENCE_CASES, ids=[case["name"] for case in PRECEDENCE_CASES])
def test_responder_precedence_matches_the_shared_contract(tmp_path: Path, case: dict):
    db = tmp_path / "state.db"
    room = _create_room(db, _owned_members(case["owner"]))
    _append_user(db, event_id="user-1", text=case["text"])

    assert _round_zero_profiles(room, db) == case["expected"]


def test_owner_flag_is_validated_onto_the_roster():
    members = discussion.validate_roster(OWNED_MEMBERS, local_profiles=LOCAL_PROFILES)
    assert [member.owner for member in members] == [False, True, False]
    room = discussion.validate_room(
        {"room_id": ROOM_ID, "name": "Release", "members": OWNED_MEMBERS,
         "authority_gateway_id": GATEWAY_ID, "authority_epoch": 1},
        local_profiles=LOCAL_PROFILES,
    )
    assert room.owner is not None and room.owner.profile == OWNER_PROFILE
    assert discussion.validate_room(
        {"room_id": ROOM_ID, "name": "Release", "members": MEMBERS,
         "authority_gateway_id": GATEWAY_ID, "authority_epoch": 1},
        local_profiles=LOCAL_PROFILES,
    ).owner is None


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ([{**MEMBERS[0], "owner": True}, {**MEMBERS[1], "owner": True}], "at most one"),
        ([{**MEMBERS[0], "owner": "yes"}, MEMBERS[1]], "owner must be a boolean"),
    ],
)
def test_owner_flag_is_exclusive_and_boolean(members: list[dict], match: str):
    with pytest.raises(discussion.DiscussionValidationError, match=match):
        discussion.validate_roster(members, local_profiles=LOCAL_PROFILES)


@pytest.mark.parametrize("owner", [OWNER_PROFILE, None])
@pytest.mark.parametrize("text", ["hello", "@ everyone hi"])
def test_unmentioned_human_message_never_has_zero_responders(tmp_path: Path, owner: str | None, text: str):
    db = tmp_path / "state.db"
    room = _create_room(db, _owned_members(owner))
    _append_user(db, event_id="user-1", text=text)

    decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert decision.status == "task", decision


@pytest.mark.parametrize("status", ["deferred", "failed"])
def test_unavailable_owner_falls_back_to_the_rest_of_the_room_visibly(tmp_path: Path, status: str):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Ship it?")

    owner_task = _next_task(room, db)
    assert owner_task.member.profile == OWNER_PROFILE
    assert owner_task.member_index == 0
    assert "unavailable" not in owner_task.payload["prompt"]
    _fail_or_defer(room, db, owner_task, status)

    fallback: list[discussion.DiscussionTaskPlan] = []
    for expected_profile, expected_index in (("research", 1), ("review", 2)):
        task = _next_task(room, db)
        assert (task.member.profile, task.round_index, task.member_index) == (expected_profile, 0, expected_index)
        assert f"[Owner @{OWNER_PROFILE} was unavailable for this message; it is routed to you.]" in task.payload["prompt"]
        fallback.append(_settle_next(room, db, text="(pass)"))

    decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert (decision.status, decision.reason) == ("settled", "silent_round")
    kinds = [event["kind"] for event in _events(db)]
    assert kinds.count(f"turn.{status}") == 1  # the owner's fallback trigger stays visible in the log
    assert [event["payload"]["member_id"] for event in _events(db) if event["kind"] == "turn.settled"] == [
        "member-research", "member-review"]


def test_owner_pass_settles_the_round_as_silence(tmp_path: Path):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Anything to add?")

    owner_task = _settle_next(room, db, text="(pass)")
    assert owner_task.member.profile == OWNER_PROFILE

    decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert (decision.status, decision.reason) == ("settled", "silent_round")


def test_owner_reply_hands_off_to_cited_peers_as_before(tmp_path: Path):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Ship it?")

    owner_task = _settle_next(room, db, text="@review please sign off.")
    assert owner_task.member.profile == OWNER_PROFILE

    second = _next_task(room, db)
    assert (second.member.profile, second.round_index) == ("review", 1)


def test_owner_fallback_task_is_deterministic_and_reconstructs_after_restart(tmp_path: Path):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Ship it?")
    _fail_or_defer(room, db, _next_task(room, db), "deferred")

    first = _next_task(room, db)
    assert (first.member.profile, first.member_index) == ("research", 1)
    assert first == _next_task(room, db)
    assert set(first.payload) == {"target_member_id", "target_profile", "prompt", "source_event_seq"}

    driver.admit_task(db, first.identity, payload=first.payload, clock=time.time)
    reconstructed = discussion.reconstruct_task_plan(
        room, _events(db), driver.get_task(db, first.identity), local_profiles=LOCAL_PROFILES)
    assert reconstructed == first
    # A restart re-plans from the same durable log and must land on the same task.
    assert discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES).task == first


@pytest.mark.parametrize("text", [f"@{OWNER_PROFILE} and @review, thoughts?", "@everyone thoughts?"])
def test_explicitly_mentioned_owner_going_offline_triggers_no_owner_fallback(tmp_path: Path, text: str):
    """Mentions win outright: a mentioned owner that defers is silence, not a routing fallback."""
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text=text)

    tasks: list[discussion.DiscussionTaskPlan] = []
    while (decision := discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)).status == "task":
        assert decision.task is not None
        tasks.append(decision.task)
        if decision.task.member.profile == OWNER_PROFILE:
            _fail_or_defer(room, db, decision.task, "deferred")
        else:
            _settle_next(room, db, text="(pass)")

    assert [task.member.profile for task in tasks] == (
        [OWNER_PROFILE, "review"] if text.startswith(f"@{OWNER_PROFILE}") else ["research", OWNER_PROFILE, "review"])
    assert not any("unavailable" in task.payload["prompt"] for task in tasks)


@pytest.mark.parametrize("owner", [OWNER_PROFILE, None])
def test_prompt_stays_serviceable_at_every_delta_boundary(tmp_path: Path, owner: str | None):
    """Any valid transcript yields a prompt within the driver limit, with or without the owner notice."""
    for index, middle in enumerate(range(64_860, 64_980, 2)):
        db = tmp_path / f"state-{index}.db"
        room = _create_room(db, _owned_members(owner))
        for seq, size in enumerate((1_000, middle, discussion.MAX_USER_TEXT_BYTES)):
            _append_user(db, event_id=f"user-{seq}", text="x" * size)
        if owner:
            _fail_or_defer(room, db, _next_task(room, db), "deferred")
        prompt = _next_task(room, db).payload["prompt"]
        assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES, middle


def test_owner_retry_that_defers_again_keeps_the_fallback_going(tmp_path: Path):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Ship it?")
    owner_task = _next_task(room, db)
    _fail_or_defer(room, db, owner_task, "deferred")
    assert _settle_next(room, db, text="(pass)").member.profile == "research"

    retried = discussion.plan_publication(
        room, _events(db), owner_task, status="deferred", result={"reason": "member_unavailable"},
        execution_generation=2, local_profiles=LOCAL_PROFILES)
    _append_publication(db, retried)

    task = _next_task(room, db)
    assert (task.member.profile, task.round_index, task.member_index) == ("review", 0, 2)


def test_owner_retry_that_answers_ends_the_fallback_without_a_second_owner_turn(tmp_path: Path):
    db = tmp_path / "state.db"
    room = _create_room(db, OWNED_MEMBERS)
    _append_user(db, event_id="user-1", text="Ship it?")
    owner_task = _next_task(room, db)
    _fail_or_defer(room, db, owner_task, "deferred")
    assert _settle_next(room, db, text="(pass)").member.profile == "research"

    recovered = discussion.plan_publication(
        room, _events(db), owner_task, status="settled", result={"text": "(pass)"}, local_profiles=LOCAL_PROFILES)
    _append_publication(db, recovered)

    decision = discussion.plan_next_task(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert (decision.status, decision.reason) == ("settled", "silent_round")
    owner_turns = [e for e in _events(db) if e["kind"].startswith("turn.") and e["payload"]["member_id"] == "member-build"]
    assert [e["kind"] for e in owner_turns] == ["turn.deferred", "turn.settled"]
