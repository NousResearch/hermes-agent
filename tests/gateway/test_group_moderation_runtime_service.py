from __future__ import annotations

from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.direct_control_platform_specs import (
    QQ_ADMIN_GROUP_MODERATION_SPEC,
    WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
)
from gateway.group_moderation_request_platform_specs import GroupModerationRequestPlatformSpec
from gateway.session import SessionSource


def _make_source(
    *,
    platform: Platform = Platform.QQ_NAPCAT,
    chat_type: str = "group",
    user_id: str = "179033731",
    chat_id: str = "726109087",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_admin_platform_group_moderation_request_passes_expected_context():
    from gateway.group_moderation_runtime_service import match_admin_platform_group_moderation_request

    source = _make_source()
    matcher = MagicMock(return_value=({"action": "kick_user", "target": "group:726109087"}, None))
    request_spec = GroupModerationRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=matcher,
        action_matcher=lambda body: "kick_user",
        user_query_extractor=lambda body: "广告哥",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: 600,
    )

    tool_args, error = match_admin_platform_group_moderation_request(
        source=source,
        body="把广告哥踢了，原因广告。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
        request_spec=request_spec,
    )

    assert error is None
    assert tool_args == {"action": "kick_user", "target": "group:726109087"}
    matcher.assert_called_once_with(
        source=source,
        body="把广告哥踢了，原因广告。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        action_matcher=request_spec.action_matcher,
        target_extractor=QQ_ADMIN_GROUP_MODERATION_SPEC.target_extractor,
        user_query_extractor=request_spec.user_query_extractor,
        reason_extractor=request_spec.reason_extractor,
        duration_extractor=request_spec.duration_extractor,
        current_group_target_formatter=QQ_ADMIN_GROUP_MODERATION_SPEC.current_group_target_formatter,
        missing_target_message=QQ_ADMIN_GROUP_MODERATION_SPEC.missing_target_message,
    )


def test_format_admin_platform_group_moderation_reply_for_mute():
    from gateway.group_moderation_runtime_service import format_admin_platform_group_moderation_reply

    result = format_admin_platform_group_moderation_reply(
        {
            "action": "mute_user",
            "target": "group:726109087",
            "user_query": "广告哥",
            "reason": "广告",
            "duration_seconds": 600,
        },
        {"action": "mute_user", "group_id": "726109087"},
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "已把 QQ 群 726109087 的 广告哥 禁言 600 秒。 原因：广告。"


def test_format_admin_platform_group_moderation_reply_for_kick_uses_result_member():
    from gateway.group_moderation_runtime_service import format_admin_platform_group_moderation_reply

    result = format_admin_platform_group_moderation_reply(
        {"action": "kick_user", "target": "group:726109087", "user_query": "广告哥"},
        {"action": "kick_user", "group_id": "726109087", "member_name": "卖草的"},
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "已把 QQ 群 726109087 的 卖草的 踢出。"


def test_format_admin_platform_group_moderation_reply_returns_not_capable_detail():
    from gateway.group_moderation_runtime_service import format_admin_platform_group_moderation_reply

    result = format_admin_platform_group_moderation_reply(
        {"action": "kick_user", "target": "project@chatroom", "user_query": "广告哥"},
        {
            "success": False,
            "platform": "weixin",
            "action": "kick_user",
            "capability": "not_capable",
            "detail": "微信群暂不支持禁言/踢人。",
        },
        spec=WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "微信群暂不支持禁言/踢人。"


def test_run_admin_platform_group_moderation_shortcut_surfaces_tool_exception():
    from gateway.group_moderation_runtime_service import run_admin_platform_group_moderation_shortcut

    logger = MagicMock()

    result = run_admin_platform_group_moderation_shortcut(
        tool_args={"action": "kick_user"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        logger=logger,
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "QQ 群管理执行失败：boom"
    logger.warning.assert_called_once()


def test_run_admin_platform_group_moderation_shortcut_rejects_non_success_result():
    from gateway.group_moderation_runtime_service import run_admin_platform_group_moderation_shortcut

    result = run_admin_platform_group_moderation_shortcut(
        tool_args={"action": "kick_user"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {},
        logger=MagicMock(),
        spec=QQ_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "QQ 群管理执行失败：工具未返回成功结果"


def test_run_admin_platform_group_moderation_shortcut_returns_not_capable_reply():
    from gateway.group_moderation_runtime_service import run_admin_platform_group_moderation_shortcut

    result = run_admin_platform_group_moderation_shortcut(
        tool_args={"action": "kick_user", "target": "project@chatroom", "user_query": "广告哥"},
        shortcut_error=None,
        tool_runner=lambda tool_args: {
            "success": False,
            "platform": "weixin",
            "action": "kick_user",
            "capability": "not_capable",
            "detail": "微信群暂不支持禁言/踢人。",
        },
        logger=MagicMock(),
        spec=WEIXIN_ADMIN_GROUP_MODERATION_SPEC,
    )

    assert result == "微信群暂不支持禁言/踢人。"
