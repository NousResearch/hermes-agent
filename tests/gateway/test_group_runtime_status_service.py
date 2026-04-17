from __future__ import annotations

from gateway.group_runtime_status_service import (
    build_qq_group_runtime_status_details,
    build_weixin_group_runtime_status_details,
    unique_report_targets,
    worker_report_targets,
)


def test_unique_report_targets_dedupes_and_skips_empty_values():
    assert unique_report_targets(
        [" qq_napcat:dm:179033731 ", "", None, "qq_napcat:dm:179033731", "qq_napcat:group:726109087"]
    ) == [
        "qq_napcat:dm:179033731",
        "qq_napcat:group:726109087",
    ]


def test_worker_report_targets_can_require_daily_enabled():
    workers = [
        {
            "worker_name": "钢镚",
            "daily_report_enabled": True,
            "daily_report_target": "qq_napcat:dm:179033731",
        },
        {
            "worker_name": "二狗",
            "daily_report_enabled": False,
            "daily_report_target": "qq_napcat:dm:000000",
        },
        {
            "worker_name": "翠花",
            "daily_report_enabled": True,
            "daily_report_target": "qq_napcat:dm:179033731",
        },
    ]

    assert worker_report_targets(workers, "daily_report_target", require_daily_enabled=True) == [
        "qq_napcat:dm:179033731"
    ]


def test_build_qq_group_runtime_status_details_merges_overlay_and_worker_targets():
    details = build_qq_group_runtime_status_details(
        "group:726109087",
        get_group_policy_fn=lambda group_id: {
            "group_id": group_id,
            "mode": "default",
            "archive_enabled": False,
            "daily_report_enabled": False,
            "daily_report_target": None,
            "manual_report_target": "qq_napcat:group:726109087",
        },
        get_group_monitoring_overlay_fn=lambda group_id: {
            "active": True,
            "mode": "collect_only",
            "archive_enabled": True,
            "daily_report_enabled": True,
            "workers": [
                {
                    "worker_name": "钢镚",
                    "daily_report_enabled": True,
                    "daily_report_target": "qq_napcat:dm:179033731",
                    "manual_report_target": "qq_napcat:group:726109087",
                },
                {
                    "worker_name": "二狗",
                    "daily_report_enabled": False,
                    "daily_report_target": "qq_napcat:dm:000000",
                    "manual_report_target": "qq_napcat:dm:179033731",
                },
            ],
        },
    )

    assert details == {
        "platform_label": "QQ 群",
        "target_label": "726109087",
        "effective_mode": "collect_only",
        "can_reply_in_group": False,
        "archive_enabled": True,
        "daily_report_enabled": True,
        "daily_targets": ["qq_napcat:dm:179033731"],
        "manual_targets": [
            "qq_napcat:group:726109087",
            "qq_napcat:dm:179033731",
        ],
        "worker_names": ["钢镚", "二狗"],
    }


def test_build_weixin_group_runtime_status_details_dedupes_reporting_targets():
    details = build_weixin_group_runtime_status_details(
        "project@chatroom",
        get_group_policy_fn=lambda chat_id: {
            "chat_id": chat_id,
            "mode": "collect_only",
            "archive_enabled": True,
            "daily_report_enabled": True,
        },
        describe_group_reporting_fn=lambda *, chat_id: {
            "effective_targets": {
                "daily_report_targets": ["weixin:wxid_admin", "weixin:wxid_admin"],
                "manual_report_targets": ["weixin:project@chatroom", ""],
            }
        },
    )

    assert details == {
        "platform_label": "微信群",
        "target_label": "project@chatroom",
        "effective_mode": "collect_only",
        "can_reply_in_group": False,
        "archive_enabled": True,
        "daily_report_enabled": True,
        "daily_targets": ["weixin:wxid_admin"],
        "manual_targets": ["weixin:project@chatroom"],
        "worker_names": [],
    }
