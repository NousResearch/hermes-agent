"""Behavior tests for deterministic same-gateway Discussion policy."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore


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
MEMBER_IDS = tuple(member["member_id"] for member in MEMBERS)


def _attachment(
    kind: str,
    name: str,
    mime: str,
    *,
    size: int = 128,
    attachment_id: str = "att_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
) -> dict:
    return {
        "attachment_id": attachment_id,
        "kind": kind,
        "name": name,
        "size": size,
        "mime": mime,
    }


def _attachment_manifest() -> list[dict]:
    return [
        _attachment(
            "image",
            "diagram.png",
            "image/png",
            size=2048,
            attachment_id="att_11111111111111111111111111111111",
        ),
        _attachment(
            "pdf",
            "release.pdf",
            "application/pdf",
            size=4096,
            attachment_id="att_22222222222222222222222222222222",
        ),
        _attachment(
            "file",
            "notes.txt",
            "text/plain",
            size=512,
            attachment_id="att_33333333333333333333333333333333",
        ),
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
    attachments: list[dict] | None = None,
) -> dict:
    if attachments:
        store = HostedRoomAttachmentStore(db)
        for entry in attachments:
            header = {"image": b"\x89PNG\r\n\x1a\n", "pdf": b"%PDF-"}.get(
                entry["kind"], b""
            )
            data = header + b"x" * (entry["size"] - len(header))
            # Keep stable planner fixtures, but require real verified commitments.
            with patch(
                "gateway.hosted_room_attachments.secrets.token_hex",
                return_value=entry["attachment_id"].removeprefix("att_"),
            ):
                uploaded = store.put(
                    room_id=ROOM_ID,
                    upload_id=entry["attachment_id"],
                    kind=entry["kind"],
                    name=entry["name"],
                    mime=entry["mime"],
                    data=data,
                )
            assert {key: uploaded[key] for key in entry} == entry
        store.commit_message(
            room_id=ROOM_ID,
            event_id=event_id,
            manifest=attachments,
            recipient_member_ids=MEMBER_IDS,
            viewer_access=True,
            hold_until_event=True,
        )
    return hosted_rooms.append_event(
        db,
        room_id=ROOM_ID,
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "local-user"},
        authority_gateway_id=GATEWAY_ID,
        authority_epoch=1,
        payload={
            "text": text,
            "thread_id": thread_id,
            **({"attachments": attachments} if attachments is not None else {}),
        },
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


def test_attachment_only_user_payload_is_accepted():
    payload = discussion.validate_user_payload(
        {
            "text": "",
            "thread_id": "thread-1",
            "attachments": [_attachment("file", "notes.txt", "text/plain")],
        },
        member_ids=MEMBER_IDS,
    )

    assert payload["text"] == ""
    assert len(payload["attachments"]) == 1


def test_attachment_manifest_total_bytes_are_bounded():
    attachments = [
        _attachment(
            "file",
            f"attachment-{index}.bin",
            "application/octet-stream",
            size=4_000_000,
            attachment_id=f"att_{index:032x}",
        )
        for index in range(discussion.MAX_ATTACHMENTS)
    ]

    with pytest.raises(
        discussion.DiscussionValidationError, match="message byte limit"
    ):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": attachments,
            },
            member_ids=MEMBER_IDS,
        )


def test_attachment_count_is_bounded():
    attachments = [
        _attachment(
            "file",
            f"notes-{index}.txt",
            "text/plain",
            attachment_id=f"att_{index:032x}",
        )
        for index in range(discussion.MAX_ATTACHMENTS + 1)
    ]

    with pytest.raises(discussion.DiscussionValidationError, match="at most 8"):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": attachments,
            },
            member_ids=MEMBER_IDS,
        )


@pytest.mark.parametrize(
    ("attachment", "match"),
    [
        (
            _attachment("file", "x" * 256, "text/plain"),
            "bounded basename",
        ),
        (
            _attachment("file", "../secret", "text/plain"),
            "bounded basename",
        ),
        (
            _attachment(
                "file",
                "notes.txt",
                "text/plain",
                size=discussion.MAX_ATTACHMENT_SIZE_BYTES + 1,
            ),
            "size must be between",
        ),
        (
            _attachment("image", "image.bin", "application/octet-stream"),
            "image kind requires image mime",
        ),
        (
            _attachment("pdf", "release.pdf", "application/octet-stream"),
            "pdf kind requires application/pdf",
        ),
    ],
)
def test_attachment_metadata_is_bounded(attachment: dict, match: str):
    with pytest.raises(discussion.DiscussionValidationError, match=match):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": [attachment],
            },
            member_ids=MEMBER_IDS,
        )


@pytest.mark.parametrize("attachment_id", ["", "att_short", "file:secret", "../secret"])
def test_attachment_manifest_rejects_non_opaque_ids(attachment_id: str):
    with pytest.raises(discussion.DiscussionValidationError, match="attachment_id"):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": [
                    _attachment(
                        "file",
                        "notes.txt",
                        "text/plain",
                        attachment_id=attachment_id,
                    )
                ],
            },
            member_ids=MEMBER_IDS,
        )


def test_attachment_manifest_requires_every_exact_metadata_field():
    attachment = _attachment("file", "notes.txt", "text/plain")
    attachment.pop("mime")

    with pytest.raises(discussion.DiscussionValidationError, match="missing fields"):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": [attachment],
            },
            member_ids=MEMBER_IDS,
        )


@pytest.mark.parametrize("field", ["data", "data_url", "base64", "path"])
def test_attachment_manifest_rejects_raw_data_and_path_fields(field: str):
    attachment = _attachment("file", "notes.txt", "text/plain")
    attachment[field] = "/tmp/raw" if field == "path" else "AAAA"

    with pytest.raises(discussion.DiscussionValidationError, match="unknown fields"):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": [attachment],
            },
            member_ids=MEMBER_IDS,
        )


def test_attachment_manifest_requires_frozen_member_ids():
    with pytest.raises(discussion.DiscussionValidationError, match="frozen room member"):
        discussion.validate_user_payload(
            {
                "text": "Review.",
                "thread_id": "thread-1",
                "attachments": [_attachment("file", "notes.txt", "text/plain")],
            },
        )


def test_attachment_task_reconstructs_after_its_terminal_publication(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-attachments-terminal",
        text="Review the upload.",
        attachments=_attachment_manifest(),
    )
    first = _next_task(room, db)
    publication = discussion.plan_publication(
        room,
        _events(db),
        first,
        status="settled",
        result={"text": "First review complete."},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, publication)

    reconstructed = discussion.reconstruct_task_plan(
        room,
        _events(db),
        {"identity": first.identity, "payload": first.payload},
        local_profiles=LOCAL_PROFILES,
    )
    assert reconstructed == first

    second = _next_task(room, db)
    assert second.member.member_id != first.member.member_id
    assert second.payload["attachments"] == _attachment_manifest()


def test_attachment_task_reconstructs_after_driver_reopen(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-attachments",
        text="Review the upload.",
        attachments=_attachment_manifest(),
    )
    task = _next_task(room, db)
    driver.admit_task(db, task.identity, payload=task.payload, clock=time.time)

    reconstructed = discussion.reconstruct_task_plan(
        room,
        _events(db),
        driver.get_task(db, task.identity),
        local_profiles=LOCAL_PROFILES,
    )
    assert reconstructed == task
    assert reconstructed.payload["attachments"] == _attachment_manifest()
    assert "att_11111111111111111111111111111111" not in reconstructed.payload["prompt"]


def test_attachment_manifest_changes_the_deterministic_task_id(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-attachments",
        text="Review the upload.",
        attachments=_attachment_manifest(),
    )
    with_attachments = _next_task(room, db)
    without_events = [
        {
            **event,
            "payload": {
                "text": event["payload"]["text"],
                "thread_id": event["payload"]["thread_id"],
            },
        }
        if event["kind"] == "message.user"
        else event
        for event in _events(db)
    ]
    without_attachments = discussion.plan_next_task(
        room,
        without_events,
        local_profiles=LOCAL_PROFILES,
    ).task

    assert without_attachments is not None
    assert with_attachments.identity.turn_id == without_attachments.identity.turn_id
    assert with_attachments.identity.task_id != without_attachments.identity.task_id
    assert with_attachments.payload["prompt"] != without_attachments.payload["prompt"]


def test_attachment_backlog_is_also_chunked_by_task_bytes(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    for message_index in range(3):
        manifest = [
            _attachment(
                "file",
                f"chunk-{message_index}-{file_index}.bin",
                "application/octet-stream",
                size=10_000_000,
                attachment_id=f"att_{message_index * 2 + file_index:032x}",
            )
            for file_index in range(2)
        ]
        _append_user(
            db,
            event_id=f"user-bytes-{message_index}",
            text=f"Byte batch {message_index}",
            attachments=manifest,
        )

    first = _next_task(room, db)
    assert sum(item["size"] for item in first.payload["attachments"]) == 40_000_000
    publication = discussion.plan_publication(
        room,
        _events(db),
        first,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, publication)

    second = _next_task(room, db)
    assert second.member.profile == "research"
    assert sum(item["size"] for item in second.payload["attachments"]) == 20_000_000


def test_large_attachment_backlog_advances_in_bounded_lossless_batches(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    expected_ids: list[str] = []
    for message_index in range(25):
        manifest = [
            _attachment(
                "file",
                f"part-{message_index}-{file_index}.bin",
                "application/octet-stream",
                size=1,
                attachment_id=f"att_{message_index * 8 + file_index:032x}",
            )
            for file_index in range(8)
        ]
        expected_ids.extend(item["attachment_id"] for item in manifest)
        _append_user(
            db,
            event_id=f"user-batch-{message_index}",
            text=f"Batch {message_index}",
            attachments=manifest,
        )

    observed_ids: list[str] = []
    tasks: list[discussion.DiscussionTaskPlan] = []
    while len(observed_ids) < len(expected_ids):
        task = _next_task(room, db)
        assert task.member.profile == "research"
        attachments = task.payload["attachments"]
        assert len(attachments) <= 16
        assert sum(item["size"] for item in attachments) <= 50_000_000
        observed_ids.extend(item["attachment_id"] for item in attachments)
        tasks.append(task)
        publication = discussion.plan_publication(
            room,
            _events(db),
            task,
            status="settled",
            result={"text": "First batch reviewed." if len(tasks) == 1 else "(pass)"},
            local_profiles=LOCAL_PROFILES,
        )
        assert publication.terminal_kind == "turn.settled"
        _append_publication(db, publication)

    assert observed_ids == expected_ids
    assert len(tasks) == 13
    assert len({task.identity.task_id for task in tasks}) == len(tasks)
    assert tasks[0].seen_through_seq < tasks[-1].seen_through_seq
    assert "First batch reviewed." not in tasks[1].payload["prompt"]

    next_member = _next_task(room, db)
    assert next_member.member.profile == "build"
    assert len(next_member.payload["attachments"]) == 16


def test_prompts_include_safe_metadata_and_tasks_carry_attachment_ids(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-attachments",
        text="Review the upload.",
        attachments=_attachment_manifest(),
    )

    research = _next_task(room, db)
    research_prompt = research.payload["prompt"]
    assert research.member.member_id == "member-research"
    assert "User (user): Review the upload." in research_prompt
    assert "User (user): Review the upload. diagram.png" not in research_prompt
    assert "att_11111111111111111111111111111111" not in research_prompt
    assert research.payload["attachments"] == _attachment_manifest()
    assert "Queued image/PDF attachments are staged separately" in research_prompt
    assert 'Queued image "diagram.png"' in research_prompt
    assert 'Queued PDF "release.pdf"' in research_prompt
    assert 'Staged file "notes.txt"' in research_prompt

    publication = discussion.plan_publication(
        room,
        _events(db),
        research,
        status="settled",
        result={"text": "(pass)"},
        local_profiles=LOCAL_PROFILES,
    )
    _append_publication(db, publication)
    build = _next_task(room, db)
    build_prompt = build.payload["prompt"]
    assert build.member.member_id == "member-build"
    assert "att_11111111111111111111111111111111" not in build_prompt
    assert build.payload["attachments"] == _attachment_manifest()
    assert build.identity.task_id != research.identity.task_id


def test_valid_image_pdf_and_file_manifest_is_normalized():
    payload = discussion.validate_user_payload(
        {
            "text": "Review the attached material.",
            "thread_id": "thread-1",
            "attachments": _attachment_manifest(),
        },
        member_ids=MEMBER_IDS,
    )

    assert [attachment["kind"] for attachment in payload["attachments"]] == [
        "image",
        "pdf",
        "file",
    ]
    assert payload["attachments"][0] == {
        "attachment_id": "att_11111111111111111111111111111111",
        "kind": "image",
        "name": "diagram.png",
        "size": 2048,
        "mime": "image/png",
    }
