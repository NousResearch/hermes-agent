from gateway.config import Platform
from gateway.session import SessionSource

from gateway.qq_group_moderation_requests import match_qq_group_moderation_request


def _make_source(
    *,
    chat_type: str = "group",
    user_id: str = "179033731",
    chat_id: str = "726109087",
) -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_qq_group_moderation_request_uses_current_group_as_default_target():
    source = _make_source()

    tool_args, error = match_qq_group_moderation_request(
        source=source,
        body="把广告哥禁言10分钟，原因广告。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        action_matcher=lambda body: "mute_user",
        target_extractor=lambda current_source, body: None,
        user_query_extractor=lambda body: "广告哥",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: 600,
        current_group_target_formatter=lambda chat_id: f"group:{chat_id}",
    )

    assert error is None
    assert tool_args == {
        "action": "mute_user",
        "target": "group:726109087",
        "user_query": "广告哥",
        "reason": "广告",
        "duration_seconds": 600,
    }


def test_match_qq_group_moderation_request_rejects_non_admin():
    source = _make_source(user_id="555")

    tool_args, error = match_qq_group_moderation_request(
        source=source,
        body="把广告哥踢了，原因广告。",
        admin_ids_configured=True,
        is_admin_user=False,
        admin_only_message="admin only",
        action_matcher=lambda body: "kick_user",
        target_extractor=lambda current_source, body: "group:726109087",
        user_query_extractor=lambda body: "广告哥",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: None,
        current_group_target_formatter=lambda chat_id: f"group:{chat_id}",
    )

    assert tool_args is None
    assert error == "admin only"


def test_match_qq_group_moderation_request_ignores_group_cleanup_phrase():
    source = _make_source(chat_type="dm")

    tool_args, error = match_qq_group_moderation_request(
        source=source,
        body="726109087群你已经被踢出了 去掉",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        action_matcher=lambda body: "kick_user",
        target_extractor=lambda current_source, body: "group:726109087",
        user_query_extractor=lambda body: "你",
        reason_extractor=lambda body: "",
        duration_extractor=lambda body: None,
        current_group_target_formatter=lambda chat_id: f"group:{chat_id}",
    )

    assert tool_args is None
    assert error is None


def test_match_qq_group_moderation_request_supports_platform_specific_current_group_formatter():
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="179033731",
        user_name="發發發",
        chat_id="project@chatroom",
        chat_type="group",
    )

    tool_args, error = match_qq_group_moderation_request(
        source=source,
        body="把广告哥踢了，原因广告。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        action_matcher=lambda body: "kick_user",
        target_extractor=lambda current_source, body: None,
        user_query_extractor=lambda body: "广告哥",
        reason_extractor=lambda body: "广告",
        duration_extractor=lambda body: None,
        current_group_target_formatter=lambda chat_id: str(chat_id),
        missing_target_message="要禁言/踢人，请直接说清 chatroom。",
    )

    assert error is None
    assert tool_args == {
        "action": "kick_user",
        "target": "project@chatroom",
        "user_query": "广告哥",
        "reason": "广告",
    }
