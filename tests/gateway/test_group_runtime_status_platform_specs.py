from gateway.config import Platform
from gateway.group_runtime_status_platform_specs import (
    build_qq_group_runtime_status_platform_spec,
    build_weixin_group_runtime_status_platform_spec,
)


def test_build_qq_group_runtime_status_platform_spec_uses_injected_dependencies():
    spec = build_qq_group_runtime_status_platform_spec(
        get_group_policy_fn=lambda group_id: {
            "group_id": group_id,
            "mode": "collect_only",
            "archive_enabled": True,
            "daily_report_enabled": False,
        },
        get_group_monitoring_overlay_fn=lambda group_id: {
            "active": True,
            "mode": "collect_only",
            "archive_enabled": True,
            "daily_report_enabled": True,
            "workers": [{"worker_name": "钢镚"}],
        },
    )

    assert spec.platform is Platform.QQ_NAPCAT
    details = spec.load_status_details("group:726109087")
    assert details["platform_label"] == "QQ 群"
    assert details["target_label"] == "726109087"
    assert details["worker_names"] == ["钢镚"]


def test_build_weixin_group_runtime_status_platform_spec_uses_injected_dependencies():
    spec = build_weixin_group_runtime_status_platform_spec(
        get_group_policy_fn=lambda chat_id: {
            "chat_id": chat_id,
            "mode": "collect_only",
            "archive_enabled": True,
            "daily_report_enabled": True,
        },
        describe_group_reporting_fn=lambda *, chat_id: {
            "effective_targets": {
                "daily_report_targets": ["weixin:wxid_admin"],
                "manual_report_targets": ["weixin:project@chatroom"],
            }
        },
    )

    assert spec.platform is Platform.WEIXIN
    details = spec.load_status_details("project@chatroom")
    assert details["platform_label"] == "微信群"
    assert details["daily_targets"] == ["weixin:wxid_admin"]
    assert details["manual_targets"] == ["weixin:project@chatroom"]
