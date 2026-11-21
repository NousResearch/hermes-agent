"""The gave-up creator wake must not expose raw worker/provider errors."""
from types import SimpleNamespace

from gateway.kanban_watchers_notifier import _KanbanNotification


def test_gave_up_wake_redacts_first_line_and_keeps_recovery_hint():
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    event = SimpleNamespace(
        kind="gave_up",
        payload={
            "failures": 1,
            "error": (
                f"Credential rejected at /Users/alice/private/token.txt: {secret} "
                + "x" * 300
                + "\nsecond line must not enter the wake"
            ),
        },
    )
    task = SimpleNamespace(
        title="credential failure", assignee="worker", session_id="creator-session",
    )
    notification = _KanbanNotification(
        runner=object(),
        d={
            "sub": {
                "task_id": "t_123",
                "platform": "telegram",
                "chat_id": "creator-chat",
                "delivery_mode": "wake",
            },
            "task": task,
            "events": [event],
            "board": "default",
        },
        platform_cls=None,
        sub_fail_counts={},
    )

    notification.format_event(event)
    notification.build_wake_text()

    assert "gave up; blocked until fixed" in notification.synth
    assert "Credential rejected" in notification.synth
    assert "[local path]" in notification.synth
    assert secret not in notification.synth
    assert "/Users/alice/private/token.txt" not in notification.synth
    assert "second line must not enter the wake" not in notification.synth
    assert "hermes kanban unblock t_123" in notification.synth
    assert len(notification.wake_handoff.split(" Fix the cause", 1)[0]) <= 160
