from datetime import datetime, timezone

from agent.jarvis_notification_queue import (
    append_notification,
    latest_notifications,
    load_notifications,
    make_notification_id,
    mark_notification,
    pending_notifications,
)


def test_notification_queue_append_and_mark(tmp_path):
    path = tmp_path / "notifications.jsonl"
    appended = append_notification(
        path,
        {
            "notification_id": "jn_test",
            "handoff_id": "handoff-test",
            "source_profile": "caitsith",
            "target_profile": "default",
            "target_platform": "discord",
            "status": "pending",
            "message": "Jarvis 알림: 등록했습니다.",
            "created_at": "2026-07-08T00:00:00+00:00",
        },
    )
    assert appended["status"] == "pending"
    assert load_notifications(path)[0]["notification_id"] == "jn_test"

    marked = mark_notification(path, "jn_test", "sent", delivery_result="discord:ok")

    records = load_notifications(path)
    assert len(records) == 2
    assert marked["status"] == "sent"
    assert marked["delivery_result"] == "discord:ok"
    assert pending_notifications(path) == []


def test_latest_notifications_are_last_wins_by_handoff_id(tmp_path):
    path = tmp_path / "notifications.jsonl"
    append_notification(
        path,
        {
            "notification_id": "jn_old",
            "handoff_id": "handoff-same",
            "source_profile": "caitsith",
            "target_profile": "default",
            "target_platform": "discord",
            "status": "pending",
            "message": "old",
            "created_at": "2026-07-08T00:00:00+00:00",
        },
    )
    append_notification(
        path,
        {
            "notification_id": "jn_new",
            "handoff_id": "handoff-same",
            "source_profile": "caitsith",
            "target_profile": "default",
            "target_platform": "discord",
            "status": "failed",
            "message": "new",
            "created_at": "2026-07-08T00:01:00+00:00",
        },
    )

    latest = latest_notifications(load_notifications(path))

    assert len(latest) == 1
    assert latest[0]["notification_id"] == "jn_new"
    assert latest[0]["status"] == "failed"


def test_make_notification_id_is_stable_for_seed_and_time():
    notification_id = make_notification_id(
        datetime(2026, 7, 8, tzinfo=timezone.utc),
        seed="handoff-test",
    )

    assert notification_id.startswith("jn_20260708_000000_")
    assert notification_id == make_notification_id(
        datetime(2026, 7, 8, tzinfo=timezone.utc),
        seed="handoff-test",
    )
