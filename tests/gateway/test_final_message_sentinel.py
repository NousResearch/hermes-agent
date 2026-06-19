from gateway.final_sentinel import (
    FINAL_MESSAGE_SENTINEL,
    FinalSentinelLifecycleSnapshot,
    should_send_final_sentinel,
    strip_trailing_final_sentinel,
)


def _idle_lifecycle() -> FinalSentinelLifecycleSnapshot:
    return FinalSentinelLifecycleSnapshot()


def test_strip_trailing_model_complete_sentinel() -> None:
    assert strip_trailing_final_sentinel("answer\nCOMPLETE") == "answer"
    assert strip_trailing_final_sentinel("answer\n\nCOMPLETE\n") == "answer"
    assert strip_trailing_final_sentinel("answer COMPLETE") == "answer COMPLETE"


def test_complete_requires_true_idle_lifecycle_snapshot() -> None:
    assert should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=_idle_lifecycle(),
    )

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=None,
    )


def test_no_complete_when_pending_messages_exist() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(pending_message=True)

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("pending_message",)


def test_no_complete_while_approval_is_pending() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(approval_pending=True)

    assert not should_send_final_sentinel(
        platform="discord",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("approval_pending",)


def test_no_complete_before_post_delivery_background_review_resolved() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(
        post_delivery_callback_pending=True,
        background_review_pending=True,
    )

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == (
        "post_delivery_callback_pending",
        "background_review_pending",
    )


def test_no_complete_when_drain_or_followup_was_spawned() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(drain_task_spawned=True)

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("drain_task_spawned",)


def test_complete_platform_and_delivery_gate_still_applies() -> None:
    assert FINAL_MESSAGE_SENTINEL == "COMPLETE"
    assert not should_send_final_sentinel(
        platform="slack",
        response_delivered=True,
        lifecycle=_idle_lifecycle(),
    )
    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=False,
        lifecycle=_idle_lifecycle(),
    )
    assert not should_send_final_sentinel(
        platform="telegram",
        message_type="command",
        response_delivered=True,
        lifecycle=_idle_lifecycle(),
    )
