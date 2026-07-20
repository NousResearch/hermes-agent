"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest


from gateway.group_monitoring_runtime_service import build_group_monitoring_summary
from gateway.group_runtime_platform_specs import GroupMonitoringRuntimePlatformSpec


def test_build_group_monitoring_summary_merges_qq_and_weixin_collect_only_groups():
    summary = build_group_monitoring_summary(
        list_qq_group_policies_fn=lambda: [
            {
                "group_id": "726109087",
                "group_name": "项目群",
                "mode": "collect_only",
                "archive_enabled": True,
                "daily_report_enabled": True,
                "daily_report_target": "qq:179033731",
            }
        ],
        list_qq_intel_workers_fn=lambda status=None: [
            {
                "worker_name": "钢镚",
                "status": "active_collecting",
                "target_group_id": "726109087",
                "target_group_name": "项目群",
                "daily_report_enabled": True,
                "daily_report_target": "qq:179033731",
                "manual_report_target": "qq:179033731",
            }
        ],
        list_weixin_group_policies_fn=lambda: [
            {
                "chat_id": "wx-group-1",
                "group_name": "微信群项目组",
                "mode": "collect_only",
                "archive_enabled": True,
                "daily_report_enabled": True,
            }
        ],
        describe_weixin_group_reporting_fn=lambda *, chat_id: {
            "effective_targets": {
                "daily_report_targets": ["weixin:filehelper"],
                "manual_report_targets": ["weixin:filehelper"],
            }
        },
    )

    assert summary["active_collect_only_groups"] == 2
    assert summary["active_worker_count"] == 1
    assert summary["platform_counts"] == {"qq_napcat": 1, "weixin": 1}
    assert summary["platform_active_worker_counts"] == {"qq_napcat": 1, "weixin": 0}
    assert [group["platform"] for group in summary["groups"]] == ["qq_napcat", "weixin"]
    assert summary["groups"][0]["group_id"] == "726109087"
    assert summary["groups"][0]["worker_names"] == ["钢镚"]
    assert summary["groups"][0]["daily_targets"] == ["qq:179033731"]
    assert summary["groups"][1]["chat_id"] == "wx-group-1"
    assert summary["groups"][1]["platform_label"] == "微信群"
    assert summary["groups"][1]["can_reply_in_group"] is False


def test_build_group_monitoring_summary_accepts_platform_specs():
    summary = build_group_monitoring_summary(
        platform_specs=[
            GroupMonitoringRuntimePlatformSpec(
                platform="qq_napcat",
                load_summary=lambda: {
                    "active_worker_count": 2,
                    "groups": [
                        {
                            "platform": "qq_napcat",
                            "platform_label": "QQ 群",
                            "group_id": "726109087",
                            "group_name": "项目群",
                            "worker_names": ["钢镚"],
                        }
                    ],
                },
            ),
            GroupMonitoringRuntimePlatformSpec(
                platform="weixin",
                load_summary=lambda: {
                    "active_worker_count": 0,
                    "groups": [
                        {
                            "platform_label": "微信群",
                            "chat_id": "wx-group-1",
                            "group_name": "微信群项目组",
                            "worker_names": [],
                        }
                    ],
                },
            ),
        ]
    )

    assert summary["active_collect_only_groups"] == 2
    assert summary["active_worker_count"] == 2
    assert summary["platform_counts"] == {"qq_napcat": 1, "weixin": 1}
    assert summary["platform_active_worker_counts"] == {"qq_napcat": 2, "weixin": 0}
    assert summary["groups"][1]["platform"] == "weixin"
