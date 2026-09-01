"""Behavior tests for deterministic same-gateway Discussion policy."""

from __future__ import annotations

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
        ("Email ops@example.com, then inspect this", "research"),
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


def test_punctuated_user_mentions_reach_every_addressed_member(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(
        db,
        event_id="user-1",
        text="@research, @build: and @review. Reply independently.",
    )

    assert _settle_next(room, db, text="Research ready.").member.profile == (
        "research"
    )
    assert _settle_next(room, db, text="Build ready.").member.profile == "build"
    assert _settle_next(room, db, text="Review ready.").member.profile == "review"


def test_unknown_explicit_mention_does_not_broadcast(
    room_db: tuple[Path, dict],
):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@unknown inspect this")

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )

    assert decision.status == "settled"
    assert decision.reason == "unresolved_mention"


def test_mentions_use_longest_exact_legacy_handle_before_trailing_punctuation():
    members = (
        discussion.DiscussionMember(
            member_id="member-build",
            profile="build",
            handle="build",
            display_name="Build",
        ),
        discussion.DiscussionMember(
            member_id="member-build-colon",
            profile="build-colon",
            handle="build:",
            display_name="Build Colon",
        ),
        discussion.DiscussionMember(
            member_id="member-all-dot",
            profile="all-dot",
            handle="all.",
            display_name="All Dot",
        ),
    )

    assert [member.handle for member in discussion.resolve_mentions(
        ("@build:. reply",), members, default_all=False
    )] == ["build:"]
    assert [member.handle for member in discussion.resolve_mentions(
        ("@all.. reply",), members, default_all=False
    )] == ["all."]


def test_code_email_escaped_and_unicode_adjacent_tokens_are_not_mentions():
    members = (
        discussion.DiscussionMember(
            member_id="member-build",
            profile="build",
            handle="build",
            display_name="Build",
        ),
        discussion.DiscussionMember(
            member_id="member-review",
            profile="review",
            handle="review",
            display_name="Review",
        ),
    )

    text = (
        "Email ops@example.com and 用户@build. "
        "Document `@everyone.` and \\@review literally. "
        "```text\n@build: example only\n```"
    )
    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        "@`ignored`everyone.",
        "````text\n@everyone.\n```\nstill code",
        "~~~text\n@everyone.\n",
        "`unterminated @everyone.",
    ],
)
def test_code_masking_never_synthesizes_or_leaks_mentions(text: str):
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )

    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


def test_code_masking_preserves_a_boundary_before_a_real_mention():
    members = (
        discussion.DiscussionMember(
            member_id="member-build",
            profile="build",
            handle="build",
            display_name="Build",
        ),
    )

    assert discussion.resolve_mentions(
        ("word`ignored`@build",), members, default_all=False
    ) == members


@pytest.mark.parametrize(
    "text",
    [
        "```text\n```not-a-close\n@build\n```",
        "~~~text\n~~~not-a-close\n@build\n~~~",
    ],
)
def test_fence_closers_with_trailing_text_do_not_expose_mentions(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


def test_escaped_backtick_does_not_hide_a_later_real_mention():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("Use \\` literally, then @build",),
        (member,),
        default_all=False,
    ) == (member,)


def test_unknown_token_never_narrows_or_overrides_a_broadcast():
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )

    resolved, unresolved = discussion._mention_resolution(
        ("@all @build @typo",), members, default_all=True
    )
    assert resolved == ()
    assert unresolved is True


def test_combining_mark_email_and_link_destination_are_not_mentions():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )
    text = "cafe\u0301@build [docs](https://example.test/@build)"

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        '[docs](url "note ) @build")',
        "[docs](<mailto:@build(foo>)",
        "https://example.test/?target=@build",
        "https://example.test/#@build",
    ],
)
def test_link_titles_angle_destinations_and_bare_urls_do_not_mention(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        "https://example.test/(@build)",
        "https://example.test/~@build",
        "https://example.test/;@build",
        "mailto:person@example.test?cc=@build",
    ],
)
def test_bare_uri_spans_do_not_mention_bots(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        "查看https://example.test/;@build",
        "éhttps://example.test/(~@build)",
    ],
)
def test_unicode_adjacent_bare_uri_is_still_masked(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        "(https://example.test)@build",
        "https://example.test,@build",
        "https://example.test!@build",
    ],
)
def test_visible_mention_after_bare_uri_remains_actionable(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == (member,)


@pytest.mark.parametrize(
    "text",
    [
        "http://[::1]/;@build",
        "http://[2001:db8::1]/(~@build)",
    ],
)
def test_ipv6_bare_uri_is_fully_masked(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


@pytest.mark.parametrize(
    "text",
    [
        "<!-- @everyone. -->",
        '<a title="@everyone">docs</a>',
        "<script>@everyone.</script>",
        "<style>.x{@everyone.}</style>",
    ],
)
def test_hidden_html_does_not_mention_bots(text: str):
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )

    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


def test_visible_text_inside_html_element_can_mention_a_bot():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("<span>@build</span>",), (member,), default_all=False
    ) == (member,)


@pytest.mark.parametrize(
    "text",
    [
        '<script>const x="</scripture>"; @build</script>',
        "<template><template>x</template>@build</template>",
        "<script>ß</script> @build",
        "<!DOCTYPE @build>",
        "<?xml @build?>",
    ],
)
def test_standard_html_visibility_handles_raw_blocks_and_declarations(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )
    expected = (member,) if text == "<script>ß</script> @build" else ()

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == expected


def test_hidden_html_cannot_mask_a_later_visible_mention():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("<!-- [x](unterminated --> @build",),
        (member,),
        default_all=False,
    ) == (member,)


@pytest.mark.parametrize(
    "text",
    [
        "Intro<br>@build",
        "<p>Intro</p><p>@build</p>",
        "&#64;build",
        "<span>&#64;build</span>",
    ],
)
def test_rendered_html_boundaries_and_entities_preserve_visible_mentions(text: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == (member,)


def test_mismatched_hidden_end_tag_does_not_expose_template_content():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("<template></script>@build</template>",),
        (member,),
        default_all=False,
    ) == ()


@pytest.mark.parametrize("tag", ["script", "style", "template"])
def test_self_closing_hidden_tag_remains_hidden_like_html_rendering(tag: str):
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        (f"<{tag}/>@build",),
        (member,),
        default_all=False,
    ) == ()


def test_self_closing_script_enters_raw_text_until_matching_close():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("<script/><style/></script>@build",),
        (member,),
        default_all=False,
    ) == (member,)


def test_inline_html_does_not_invent_a_text_boundary():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("word<span></span>@build",),
        (member,),
        default_all=False,
    ) == ()


@pytest.mark.parametrize(
    "tag", ["caption", "details", "dialog", "legend", "menu", "summary"]
)
def test_standard_block_elements_keep_visible_mentions_separate(tag: str):
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )
    text = f"<{tag}>@</{tag}><{tag}>everyone</{tag}>"

    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


def test_url_apostrophe_does_not_hide_a_later_visible_mention():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("[docs](https://example.test/O'Brien) @build",),
        (member,),
        default_all=False,
    ) == (member,)


def test_unclosed_link_destination_masks_the_remaining_text_once():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )
    text = "[broken](unterminated " + "](" * 10_000 + " @build"

    assert discussion.resolve_mentions((text,), (member,), default_all=False) == ()


def test_literal_unmatched_link_syntax_keeps_its_visible_mention():
    member = discussion.DiscussionMember(
        member_id="member-build",
        profile="build",
        handle="build",
        display_name="Build",
    )

    assert discussion.resolve_mentions(
        ("](@build)",), (member,), default_all=False
    ) == (member,)


def test_malformed_link_syntax_is_linear_and_does_not_fan_out():
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )
    text = "](" * (64 * 1024 // 2)

    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


def test_bot_handoff_with_unknown_peer_surfaces_unresolved_mention(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="@research lead this")
    _settle_next(room, db, text="@build @typo please continue")

    decision = discussion.plan_next_task(
        room,
        _events(db),
        local_profiles=LOCAL_PROFILES,
    )

    assert decision.status == "settled"
    assert decision.reason == "unresolved_mention"


def test_long_unknown_mention_is_bounded():
    members = tuple(
        discussion.DiscussionMember(
            member_id=f"member-{handle}",
            profile=handle,
            handle=handle,
            display_name=handle.title(),
        )
        for handle in ("build", "review")
    )
    text = "@unknown" + "." * (64 * 1024 - len("@unknown"))

    assert discussion.resolve_mentions((text,), members, default_all=False) == ()


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
