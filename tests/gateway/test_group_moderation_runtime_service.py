from __future__ import annotations

from unittest.mock import MagicMock, patch

from gateway.config import Platform
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


def test_match_admin_qq_group_moderation_request_passes_expected_context():
    from gateway.group_moderation_runtime_service import match_admin_qq_group_moderation_request

    source = _make_source()

    with patch(
        "gateway.group_moderation_runtime_service.match_qq_group_moderation_request",
        return_value=({"action": "kick_user", "target": "group:726109087"}, None),
    ) as matcher:
        tool_args, error = match_admin_qq_group_moderation_request(
            source=source,
            body="把广告哥踢了，原因广告。",
            admin_ids_configured=True,
            is_admin_user=True,
            admin_only_message="admin only",
        )

    assert error is None
    assert tool_args == {"action": "kick_user", "target": "group:726109087"}
    matcher.assert_called_once_with(
        source=source,
        body="把广告哥踢了，原因广告。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        action_matcher=matcher.call_args.kwargs["action_matcher"],
        target_extractor=matcher.call_args.kwargs["target_extractor"],
        user_query_extractor=matcher.call_args.kwargs["user_query_extractor"],
        reason_extractor=matcher.call_args.kwargs["reason_extractor"],
        duration_extractor=matcher.call_args.kwargs["duration_extractor"],
    )


def test_format_admin_qq_group_moderation_reply_for_mute():
    from gateway.group_moderation_runtime_service import format_admin_qq_group_moderation_reply

    result = format_admin_qq_group_moderation_reply(
        {"action": "mute_user", "target": "group:726109087", "user_query": "广告哥", "reason": "广告", "duration_seconds": 600},
        {"action": "mute_user", "group_id": "726109087"},
    )

    assert result == "已把 QQ 群 726109087 的 广告哥 禁言 600 秒。 原因：广告。"


def test_format_admin_qq_group_moderation_reply_for_kick_uses_result_member():
    from gateway.group_moderation_runtime_service import format_admin_qq_group_moderation_reply

    result = format_admin_qq_group_moderation_reply(
        {"action": "kick_user", "target": "group:726109087", "user_query": "广告哥"},
        {"action": "kick_user", "group_id": "726109087", "member_name": "卖草的"},
    )

    assert result == "已把 QQ 群 726109087 的 卖草的 踢出。"


def test_run_admin_qq_group_moderation_shortcut_surfaces_tool_exception():
    from gateway.group_moderation_runtime_service import run_admin_qq_group_moderation_shortcut

    logger = MagicMock()

    result = run_admin_qq_group_moderation_shortcut(
        tool_args={"action": "kick_user"},
        shortcut_error=None,
        tool_runner=lambda tool_args: (_ for _ in ()).throw(RuntimeError("boom")),
        logger=logger,
    )

    assert result == "QQ 群管理执行失败：boom"
    logger.warning.assert_called_once()
