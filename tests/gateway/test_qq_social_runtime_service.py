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


def test_match_admin_qq_social_control_request_passes_expected_context():
    from gateway.config import Platform
    from gateway.qq_social_runtime_service import match_admin_platform_social_control_request
    from gateway.social_control_request_platform_specs import SocialControlRequestPlatformSpec

    source = _make_source()
    matcher = MagicMock(return_value=({"action": "list_requests", "request_type": "friend"}, None))
    request_spec = SocialControlRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=matcher,
        looks_like_request_list_query=lambda body: True,
        looks_like_policy_candidate=lambda body: False,
        looks_like_policy_query=lambda body: False,
        request_type_matcher=lambda body: "friend",
        notify_target_resolver=lambda source, body: "qq_napcat:dm:179033731",
    )

    tool_args, error = match_admin_platform_social_control_request(
        source=source,
        body="看看待处理的好友申请",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        request_spec=request_spec,
    )

    assert error is None
    assert tool_args == {"action": "list_requests", "request_type": "friend"}
    matcher.assert_called_once_with(
        source=source,
        body="看看待处理的好友申请",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        looks_like_request_list_query=request_spec.looks_like_request_list_query,
        looks_like_policy_candidate=request_spec.looks_like_policy_candidate,
        looks_like_policy_query=request_spec.looks_like_policy_query,
        request_type_matcher=request_spec.request_type_matcher,
        notify_target_resolver=request_spec.notify_target_resolver,
    )


def test_match_admin_qq_social_control_request_uses_qq_request_spec():
    from gateway.qq_social_runtime_service import match_admin_qq_social_control_request

    source = _make_source()

    with patch(
        "gateway.qq_social_runtime_service.match_admin_platform_social_control_request",
        return_value=({"action": "list_requests", "request_type": "friend"}, None),
    ) as matcher:
        tool_args, error = match_admin_qq_social_control_request(
            source=source,
            body="看看待处理的好友申请",
            admin_ids_configured=True,
            is_admin_user=True,
            admin_only_message="admin only",
        )

    assert error is None
    assert tool_args == {"action": "list_requests", "request_type": "friend"}
    assert matcher.call_args.kwargs["request_spec"].platform is Platform.QQ_NAPCAT


def test_format_admin_qq_social_control_reply_for_empty_friend_requests():
    from gateway.qq_social_runtime_service import format_admin_qq_social_control_reply

    result = format_admin_qq_social_control_reply(
        {"action": "list_requests", "request_type": "friend"},
        {"requests": []},
    )

    assert result == "当前没有待处理的 QQ 好友申请。"


def test_format_admin_qq_social_control_reply_for_policy_update():
    from gateway.qq_social_runtime_service import format_admin_qq_social_control_reply

    result = format_admin_qq_social_control_reply(
        {"action": "set_social_policy"},
        {
            "policy": {
                "auto_approve_friend_requests": True,
                "auto_approve_group_add_requests": False,
                "auto_approve_group_invites": True,
                "notify_target": "qq_napcat:dm:179033731",
            }
        },
    )

    assert "QQ 社交自动处理策略已更新：" in result
    assert "- 好友申请自动通过：已开启" in result
    assert "- 加群申请自动通过：已关闭" in result
    assert "- 群邀请自动通过：已开启" in result
    assert "- 通知目标：qq_napcat:dm:179033731" in result


def test_run_admin_qq_social_control_shortcut_surfaces_tool_exception():
    from gateway.qq_social_runtime_service import run_admin_qq_social_control_shortcut

    logger = MagicMock()

    result = run_admin_qq_social_control_shortcut(
        tool_args={"action": "list_requests"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        reply_formatter=lambda tool_args, result: "unused",
        logger=logger,
    )

    assert result == "QQ 社交控制执行失败：boom"
    logger.warning.assert_called_once()


def test_run_admin_qq_social_control_shortcut_rejects_non_success_result():
    from gateway.qq_social_runtime_service import run_admin_qq_social_control_shortcut

    result = run_admin_qq_social_control_shortcut(
        tool_args={"action": "list_requests"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {},
        reply_formatter=lambda tool_args, result: "unused",
        logger=MagicMock(),
    )

    assert result == "QQ 社交控制执行失败：工具未返回成功结果"
