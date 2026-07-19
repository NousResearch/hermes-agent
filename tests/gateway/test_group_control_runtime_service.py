"""See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
from __future__ import annotations
import pytest



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


def test_match_admin_platform_group_control_request_ignores_unresolved_worker_like_turn():
    from gateway.group_control_runtime_service import match_admin_platform_group_control_request

    source = _make_source()

    tool_args, error = match_admin_platform_group_control_request(
        source=source,
        body="让情报员继续盯这个群",
        target_extractor=lambda current_source, body: None,
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
        unresolved_target_guard=lambda body: "情报员" in body,
    )

    assert tool_args is None
    assert error is None


def test_match_admin_platform_group_control_request_passes_resolved_admin_context():
    from gateway.group_control_runtime_service import match_admin_platform_group_control_request

    source = _make_source(chat_type="group", chat_id="726109087")

    with patch(
        "gateway.group_control_runtime_service.match_group_control_request",
        return_value=({"action": "enable_collect_only", "target": "group:726109087"}, None),
    ) as matcher:
        tool_args, error = match_admin_platform_group_control_request(
            source=source,
            body="这个群只监听",
            target_extractor=lambda current_source, body: "group:726109087",
            admin_ids_configured=True,
            is_admin_user=True,
            missing_target_message="missing target",
            admin_only_message="admin only",
            collect_only_action="enable_collect_only",
            report_target_resolver=lambda current_source, message, prefer_dm: "current_user_dm",
        )

    assert error is None
    assert tool_args == {"action": "enable_collect_only", "target": "group:726109087"}
    matcher.assert_called_once_with(
        source=source,
        body="这个群只监听",
        target="group:726109087",
        admin_ids_configured=True,
        is_admin_user=True,
        missing_target_message="missing target",
        admin_only_message="admin only",
        collect_only_action="enable_collect_only",
        report_target_resolver=matcher.call_args.kwargs["report_target_resolver"],
    )


def test_run_admin_group_control_shortcut_formats_success_reply():
    from gateway.group_control_runtime_service import run_admin_group_control_shortcut

    logger = MagicMock()

    result = run_admin_group_control_shortcut(
        tool_args={"action": "resume_chat", "target": "group:192903718"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {"success": True, "policy": {"group_id": "192903718"}},
        error_prefix="QQ 群监听控制执行失败",
        reply_formatter=lambda tool_args, result: "已停止 QQ 群 192903718 的监听采集。",
        logger=logger,
    )

    assert result == "已停止 QQ 群 192903718 的监听采集。"
    logger.warning.assert_not_called()


def test_run_admin_group_control_shortcut_returns_tool_error_text():
    from gateway.group_control_runtime_service import run_admin_group_control_shortcut

    result = run_admin_group_control_shortcut(
        tool_args={"action": "resume_chat"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {"error": "群策略不存在"},
        error_prefix="QQ 群监听控制执行失败",
        reply_formatter=lambda tool_args, result: "unused",
        logger=MagicMock(),
    )

    assert result == "群策略不存在"


def test_run_admin_group_control_shortcut_surfaces_runner_exception():
    from gateway.group_control_runtime_service import run_admin_group_control_shortcut

    logger = MagicMock()

    result = run_admin_group_control_shortcut(
        tool_args={"action": "resume_chat"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        error_prefix="QQ 群监听控制执行失败",
        reply_formatter=lambda tool_args, result: "unused",
        logger=logger,
    )

    assert result == "QQ 群监听控制执行失败：boom"
    logger.warning.assert_called_once()


def test_run_admin_group_control_shortcut_rejects_non_success_result():
    from gateway.group_control_runtime_service import run_admin_group_control_shortcut

    result = run_admin_group_control_shortcut(
        tool_args={"action": "resume_chat"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {},
        error_prefix="QQ 群监听控制执行失败",
        reply_formatter=lambda tool_args, result: "unused",
        logger=MagicMock(),
    )

    assert result == "QQ 群监听控制执行失败：工具未返回成功结果"
