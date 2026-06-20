from gateway.config import Platform, PlatformConfig
from gateway.final_sentinel import (
    FINAL_MESSAGE_SENTINEL,
    FinalSentinelLifecycleSnapshot,
    should_send_final_sentinel,
    strip_trailing_final_sentinel,
)
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _SentinelTestAdapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {}


def _adapter() -> _SentinelTestAdapter:
    return _SentinelTestAdapter(PlatformConfig(), Platform.TELEGRAM)


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


def test_no_complete_while_final_report_is_pending() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(final_report_pending=True)

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("final_report_pending",)


def test_no_complete_while_running_agent_is_active() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(running_agent_active=True)

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("running_agent_active",)


def test_no_complete_while_gateway_active_run_is_present() -> None:
    lifecycle = FinalSentinelLifecycleSnapshot(gateway_active_run=True)

    assert not should_send_final_sentinel(
        platform="telegram",
        response_delivered=True,
        lifecycle=lifecycle,
    )
    assert lifecycle.blocking_reasons() == ("gateway_active_run",)


def test_lifecycle_snapshot_uses_concrete_running_agent_checker() -> None:
    adapter = _adapter()
    adapter._final_sentinel_running_agent_checker = lambda session_key: session_key == "s1"

    lifecycle = adapter._final_sentinel_lifecycle_snapshot("s1")

    assert lifecycle.running_agent_active
    assert not lifecycle.gateway_active_run
    assert lifecycle.blocking_reasons() == ("running_agent_active",)


def test_lifecycle_snapshot_uses_concrete_gateway_active_run_checker() -> None:
    adapter = _adapter()
    adapter._final_sentinel_gateway_active_run_checker = lambda session_key: session_key == "s1"

    lifecycle = adapter._final_sentinel_lifecycle_snapshot("s1")

    assert not lifecycle.running_agent_active
    assert lifecycle.gateway_active_run
    assert lifecycle.blocking_reasons() == ("gateway_active_run",)


def test_lifecycle_snapshot_marks_final_report_pending() -> None:
    adapter = _adapter()

    lifecycle = adapter._final_sentinel_lifecycle_snapshot("s1", final_report_pending=True)

    assert lifecycle.final_report_pending
    assert lifecycle.blocking_reasons() == ("final_report_pending",)


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
