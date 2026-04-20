from __future__ import annotations

from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_admin_qq_intel_control_request_passes_expected_context():
    from gateway.intel_control_request_platform_specs import IntelControlRequestPlatformSpec
    from gateway.qq_intel_runtime_service import match_admin_platform_intel_control_request

    source = _make_source()
    matcher = MagicMock(return_value=({"action": "pause_worker", "worker_name": "钢镚"}, None))
    request_spec = IntelControlRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=matcher,
        worker_name_extractor=lambda body, known_worker_names: "钢镚",
        worker_context_checker=lambda body: True,
        target_extractor=lambda current_source, body: "group:726109087",
        hire_objective_extractor=lambda body, *, worker_name, target_group: "刺探情报",
    )

    tool_args, error = match_admin_platform_intel_control_request(
        source=source,
        body="让情报员钢镚暂停任务",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=lambda body: False,
        known_worker_names={"钢镚"},
        report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        request_spec=request_spec,
    )

    assert error is None
    assert tool_args == {"action": "pause_worker", "worker_name": "钢镚"}
    matcher.assert_called_once_with(
        source=source,
        body="让情报员钢镚暂停任务",
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_joined_group_list_query=matcher.call_args.kwargs["looks_like_joined_group_list_query"],
        extract_worker_name=request_spec.worker_name_extractor,
        looks_like_worker_context=request_spec.worker_context_checker,
        known_worker_names={"钢镚"},
        target_extractor=request_spec.target_extractor,
        report_target_resolver=matcher.call_args.kwargs["report_target_resolver"],
        hire_objective_extractor=request_spec.hire_objective_extractor,
    )


def test_match_admin_qq_intel_control_request_uses_qq_request_spec():
    from gateway.qq_intel_runtime_service import match_admin_qq_intel_control_request

    source = _make_source()

    with patch(
        "gateway.qq_intel_runtime_service.match_admin_platform_intel_control_request",
        return_value=({"action": "pause_worker", "worker_name": "钢镚"}, None),
    ) as matcher:
        tool_args, error = match_admin_qq_intel_control_request(
            source=source,
            body="让情报员钢镚暂停任务",
            admin_ids_configured=True,
            is_admin_user=True,
            looks_like_joined_group_list_query=lambda body: False,
            known_worker_names={"钢镚"},
            report_target_resolver=lambda current_source, body, prefer_dm: "current_user_dm",
        )

    assert error is None
    assert tool_args == {"action": "pause_worker", "worker_name": "钢镚"}
    matcher.assert_called_once()
    assert matcher.call_args.kwargs == {
        "source": source,
        "body": "让情报员钢镚暂停任务",
        "admin_ids_configured": True,
        "is_admin_user": True,
        "looks_like_joined_group_list_query": matcher.call_args.kwargs["looks_like_joined_group_list_query"],
        "known_worker_names": {"钢镚"},
        "report_target_resolver": matcher.call_args.kwargs["report_target_resolver"],
        "request_spec": matcher.call_args.kwargs["request_spec"],
    }
    assert matcher.call_args.kwargs["request_spec"].platform is Platform.QQ_NAPCAT


def test_format_admin_qq_intel_control_reply_for_joined_group_list():
    from gateway.qq_intel_runtime_service import format_admin_qq_intel_control_reply

    result = format_admin_qq_intel_control_reply(
        {"action": "list_joined_groups"},
        {"groups": [{"group_id": "726109087", "group_name": "项目群"}]},
        status_label_formatter=lambda status: status,
        unique_report_targets_fn=lambda values: values,
    )

    assert result == "当前已加入的 QQ 群：\n- 项目群 (726109087)"


def test_format_admin_qq_intel_control_reply_for_worker_status_details():
    from gateway.qq_intel_runtime_service import format_admin_qq_intel_control_reply

    result = format_admin_qq_intel_control_reply(
        {"action": "get_worker_status", "worker_name": "钢镚"},
        {
            "worker": {
                "worker_name": "钢镚",
                "status": "active_collecting",
                "target_group_id": "726109087",
                "target_group_name": "项目群",
                "objective": "刺探情报",
                "daily_report_enabled": True,
                "daily_report_target": "qq_napcat:dm:179033731",
                "manual_report_target": "qq_napcat:group:726109087",
            }
        },
        status_label_formatter=lambda status: "正在潜伏采集" if status == "active_collecting" else str(status),
        unique_report_targets_fn=lambda values: [str(value).strip() for value in values if str(value).strip()],
    )

    assert "情报员 钢镚 当前状态：正在潜伏采集。" in result
    assert "目标群：项目群 (726109087)" in result
    assert "任务：刺探情报" in result
    assert "日报目标：qq_napcat:dm:179033731" in result
    assert "立即汇报目标：qq_napcat:group:726109087" in result


def test_run_admin_qq_intel_control_shortcut_surfaces_tool_exception():
    from gateway.qq_intel_runtime_service import run_admin_qq_intel_control_shortcut

    logger = MagicMock()

    result = run_admin_qq_intel_control_shortcut(
        tool_args={"action": "pause_worker", "worker_name": "钢镚"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        reply_formatter=lambda tool_args, result: "unused",
        logger=logger,
    )

    assert result == "QQ 情报员控制执行失败：boom"
    logger.warning.assert_called_once()


def test_run_admin_qq_intel_control_shortcut_rejects_non_success_result():
    from gateway.qq_intel_runtime_service import run_admin_qq_intel_control_shortcut

    result = run_admin_qq_intel_control_shortcut(
        tool_args={"action": "pause_worker", "worker_name": "钢镚"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {},
        reply_formatter=lambda tool_args, result: "unused",
        logger=MagicMock(),
    )

    assert result == "QQ 情报员控制执行失败：工具未返回成功结果"
