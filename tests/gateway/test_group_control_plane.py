from gateway.group_control_plane import (
    NormalizedGroupControlRequest,
    build_group_runtime_snapshot,
)


def test_normalized_group_control_request_to_tool_args_for_report_delivery():
    request = NormalizedGroupControlRequest(
        action="deliver_report",
        target="group:726109087",
        delivery_target="current_chat",
    )

    assert request.to_tool_args() == {
        "action": "deliver_report",
        "target": "group:726109087",
        "delivery_target": "current_chat",
    }


def test_normalized_group_control_request_to_tool_args_preserves_report_targets():
    request = NormalizedGroupControlRequest(
        action="enable_collect_only",
        target="group:726109087",
        daily_report_enabled=True,
        daily_report_target="current_user_dm",
        manual_report_target="current_chat",
    )

    assert request.to_tool_args() == {
        "action": "enable_collect_only",
        "target": "group:726109087",
        "daily_report_enabled": True,
        "daily_report_target": "current_user_dm",
        "manual_report_target": "current_chat",
    }


def test_build_group_runtime_snapshot_normalizes_targets_and_workers():
    snapshot = build_group_runtime_snapshot(
        platform_label="QQ 群",
        target_label="726109087",
        effective_mode="collect_only",
        archive_enabled=True,
        daily_report_enabled=True,
        daily_targets=[" qq_napcat:dm:179033731 ", "", "qq_napcat:dm:179033731"],
        manual_targets=["qq_napcat:group:726109087", None, "qq_napcat:group:726109087"],
        worker_names=["钢镚", "", "钢镚", "二狗"],
    )

    assert snapshot.to_status_details() == {
        "platform_label": "QQ 群",
        "target_label": "726109087",
        "effective_mode": "collect_only",
        "can_reply_in_group": False,
        "archive_enabled": True,
        "daily_report_enabled": True,
        "daily_targets": ["qq_napcat:dm:179033731"],
        "manual_targets": ["qq_napcat:group:726109087"],
        "worker_names": ["钢镚", "二狗"],
    }
