"""Tests for QQ intel worker assignments."""

from gateway.qq_intel_assignments import (
    QqIntelAssignmentStore,
    get_group_monitoring_overlay,
)


def test_hire_worker_starts_waiting_when_group_not_joined():
    store = QqIntelAssignmentStore()

    worker = store.hire_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[],
    )

    assert worker["worker_name"] == "钢镚"
    assert worker["status"] == "awaiting_group_approval"
    assert worker["target_group_id"] is None
    assert worker["daily_report_enabled"] is True


def test_reconcile_joined_groups_activates_worker_and_exposes_collect_only_overlay():
    store = QqIntelAssignmentStore()
    store.hire_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[],
    )

    result = store.reconcile_joined_groups(
        [
            {
                "group_id": "987654321",
                "group_name": "目标群",
            }
        ],
        updated_by="scheduler",
    )
    worker = store.get_worker("钢镚")
    overlay = get_group_monitoring_overlay("987654321")

    assert result["changed"] == 1
    assert worker["status"] == "active_collecting"
    assert worker["target_group_id"] == "987654321"
    assert worker["target_group_name"] == "目标群"
    assert overlay["active"] is True
    assert overlay["mode"] == "collect_only"
    assert overlay["archive_enabled"] is True
    assert overlay["daily_report_enabled"] is True
    assert overlay["workers"][0]["worker_name"] == "钢镚"


def test_collect_only_overlay_lists_delivery_targets_and_worker_names():
    store = QqIntelAssignmentStore()
    store.hire_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:group:987654321",
        notify_target="qq_napcat:dm:200000001",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )

    overlay = get_group_monitoring_overlay("987654321")

    assert overlay["monitoring_intent"] == "intel_worker_collect_only"
    assert overlay["worker_names"] == ["钢镚"]
    assert overlay["daily_report_targets"] == ["qq_napcat:dm:179033731"]
    assert overlay["manual_report_targets"] == ["qq_napcat:group:987654321"]
    assert overlay["notify_targets"] == ["qq_napcat:dm:200000001"]


def test_collect_only_overlay_exposes_assignment_summaries_for_multiple_workers():
    store = QqIntelAssignmentStore()
    store.hire_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="盯业务消息",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:group:987654321",
        notify_target="qq_napcat:dm:200000001",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )
    store.hire_worker(
        worker_name="二狗",
        target_group_ref="group:987654321",
        objective="盯广告消息",
        daily_report_enabled=False,
        daily_report_target=None,
        manual_report_target="qq_napcat:dm:300000001",
        notify_target="qq_napcat:dm:400000001",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )

    overlay = get_group_monitoring_overlay("987654321")

    assert overlay["active_worker_count"] == 2
    assert sorted(overlay["worker_names"]) == ["二狗", "钢镚"]
    assert overlay["report_control"]["daily_report_enabled"] is True
    assert overlay["report_control"]["daily_report_targets"] == ["qq_napcat:dm:179033731"]
    assert sorted(overlay["report_control"]["manual_report_targets"]) == [
        "qq_napcat:dm:300000001",
        "qq_napcat:group:987654321",
    ]
    assert sorted(overlay["report_control"]["notify_targets"]) == [
        "qq_napcat:dm:200000001",
        "qq_napcat:dm:400000001",
    ]
    assignments = {item["worker_name"]: item for item in overlay["worker_assignments"]}
    assert assignments["钢镚"]["collecting"] is True
    assert assignments["二狗"]["manual_report_target"] == "qq_napcat:dm:300000001"


def test_reconcile_does_not_auto_resume_paused_worker():
    store = QqIntelAssignmentStore()
    store.hire_worker(
        worker_name="钢镚",
        target_group_ref="group:987654321",
        objective="去刺探情报",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        manual_report_target="qq_napcat:dm:179033731",
        notify_target="qq_napcat:dm:179033731",
        updated_by="test",
        joined_groups=[{"group_id": "987654321", "group_name": "目标群"}],
    )
    store.set_worker_status("钢镚", status="paused", updated_by="test")

    result = store.reconcile_joined_groups(
        [{"group_id": "987654321", "group_name": "目标群"}],
        updated_by="scheduler",
    )
    worker = store.get_worker("钢镚")

    assert result["changed"] == 0
    assert worker["status"] == "paused"
