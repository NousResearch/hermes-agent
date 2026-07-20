from gateway.group_runtime import (
    GroupBatchItem,
    GroupDispatchThresholds,
    GroupTriggerState,
    decide_project_group_dispatch,
    resolve_group_trigger_reason,
)


def test_resolve_group_trigger_reason_returns_disabled_when_explicit_trigger_not_required():
    state = GroupTriggerState(require_explicit_trigger=False)

    assert resolve_group_trigger_reason(state) == "require_mention_disabled"


def test_resolve_group_trigger_reason_respects_priority_order():
    state = GroupTriggerState(
        require_explicit_trigger=True,
        slash_command=True,
        mentioned_bot=True,
        replied_to_bot=True,
        shared_followup=True,
        user_followup=True,
        recent_session_followup=True,
        name_trigger=True,
    )

    assert resolve_group_trigger_reason(state) == "slash_command"


def test_resolve_group_trigger_reason_returns_first_matching_followup_reason():
    state = GroupTriggerState(
        require_explicit_trigger=True,
        shared_followup=True,
        user_followup=True,
        recent_session_followup=True,
    )

    assert resolve_group_trigger_reason(state) == "group_followup_window"


def test_decide_project_group_dispatch_uses_direct_trigger_before_scores():
    batch = [
        GroupBatchItem(
            speaker_id="u1",
            text="看一下这个",
            direct_trigger_reason="bot_mention",
        )
    ]

    should_dispatch, reason = decide_project_group_dispatch(batch)

    assert should_dispatch is True
    assert reason == "direct_trigger:bot_mention"


def test_decide_project_group_dispatch_uses_admin_and_media_hard_triggers():
    admin_batch = [GroupBatchItem(speaker_id="u1", text="继续推进", is_admin=True)]
    media_batch = [GroupBatchItem(speaker_id="u1", text="广告", has_nontext_media=True)]

    assert decide_project_group_dispatch(admin_batch) == (True, "admin_user")
    assert decide_project_group_dispatch(media_batch) == (True, "media")


def test_decide_project_group_dispatch_skips_low_signal_single_message():
    batch = [GroupBatchItem(speaker_id="u1", text="今天天气真好")]

    should_dispatch, reason = decide_project_group_dispatch(batch)

    assert should_dispatch is False
    assert reason == "score=0"


def test_decide_project_group_dispatch_dispatches_on_explicit_request_score():
    batch = [GroupBatchItem(speaker_id="u1", text="这个怎么安排？")]

    should_dispatch, reason = decide_project_group_dispatch(batch)

    assert should_dispatch is True
    assert reason == "explicit_request"


def test_decide_project_group_dispatch_uses_threshold_score_accumulation():
    batch = [
        GroupBatchItem(speaker_id="u1", text="A" * 90),
        GroupBatchItem(speaker_id="u2", text="B" * 90),
        GroupBatchItem(speaker_id="u3", text="C" * 5),
    ]

    should_dispatch, reason = decide_project_group_dispatch(
        batch,
        thresholds=GroupDispatchThresholds(
            min_messages=4,
            min_speakers=3,
            min_chars=160,
        ),
    )

    assert should_dispatch is True
    assert reason == "multi_speaker,text_volume"
