from gateway.group_reply_formatters import (
    format_admin_group_control_reply,
    format_admin_send_reply,
    format_group_runtime_status_reply,
)


def test_format_group_runtime_status_reply_includes_targets_and_worker_names():
    result = format_group_runtime_status_reply(
        platform_label="QQ 群",
        target_label="726109087",
        effective_mode="collect_only",
        can_reply_in_group=False,
        archive_enabled=True,
        daily_report_enabled=True,
        daily_targets=["qq_napcat:dm:179033731"],
        manual_targets=["qq_napcat:group:726109087"],
        worker_names=["钢镚"],
    )

    assert "QQ 群 726109087 当前模式：collect_only。" in result
    assert "群里主动说话：不能。" in result
    assert "日报目标：qq_napcat:dm:179033731" in result
    assert "立即汇报目标：qq_napcat:group:726109087" in result
    assert "当前监听情报员：钢镚" in result


def test_format_admin_group_control_reply_formats_collect_only_enable():
    result = format_admin_group_control_reply(
        {
            "action": "enable_collect_only",
            "target": "group:726109087",
            "daily_report_enabled": True,
            "daily_report_target": "current_user_dm",
            "manual_report_target": "current_user_dm",
        },
        {
            "policy": {
                "group_id": "726109087",
                "mode": "collect_only",
                "daily_report_enabled": True,
                "daily_report_target": "qq_napcat:dm:179033731",
                "manual_report_target": "qq_napcat:group:726109087",
            }
        },
        platform_label="QQ 群",
        target_key="group_id",
        collect_only_action="enable_collect_only",
        strip_group_prefix=True,
    )

    assert "已把 QQ 群 726109087 切到监听采集模式" in result
    assert "日报已开启，发送到 qq_napcat:dm:179033731" in result
    assert "立即汇报发到 qq_napcat:group:726109087" in result


def test_format_admin_group_control_reply_formats_manual_report_delivery():
    result = format_admin_group_control_reply(
        {
            "action": "deliver_report",
            "target": "project@chatroom",
            "delivery_target": "weixin:wxid_admin",
        },
        {
            "report": {"chat_id": "project@chatroom"},
            "delivery": {"target": "weixin:wxid_admin"},
        },
        platform_label="微信群",
        target_key="chat_id",
        collect_only_action="collect_only",
        strip_group_prefix=False,
    )

    assert result == "已把 微信群 project@chatroom 的汇报发到 weixin:wxid_admin。"


def test_format_admin_send_reply_formats_platform_label_and_target():
    result = format_admin_send_reply(
        {"target": "weixin:project@chatroom", "message": "开会了"},
        platform_label="微信群",
        target_normalizer=lambda value: str(value or "").replace("weixin:", "").strip(),
    )

    assert result == "已发到 微信群 project@chatroom：开会了"
