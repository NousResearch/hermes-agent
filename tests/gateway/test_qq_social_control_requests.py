from gateway.config import Platform
from gateway.session import SessionSource

from gateway.qq_social_control_requests import match_qq_social_control_request


def _make_source(
    *,
    chat_type: str = "dm",
    user_id: str = "179033731",
    chat_id: str = "179033731",
) -> SessionSource:
    return SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id=user_id,
        user_name="發發發",
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_match_qq_social_control_request_lists_pending_friend_requests():
    source = _make_source()

    tool_args, error = match_qq_social_control_request(
        source=source,
        body="看看待处理的好友申请",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        looks_like_request_list_query=lambda body: True,
        looks_like_policy_candidate=lambda body: False,
        looks_like_policy_query=lambda body: False,
        request_type_matcher=lambda body: "friend",
        notify_target_resolver=lambda current_source, body: None,
    )

    assert error is None
    assert tool_args == {
        "action": "list_requests",
        "status": "pending",
        "request_type": "friend",
        "limit": 20,
    }


def test_match_qq_social_control_request_updates_policy_and_notify_target():
    source = _make_source()

    tool_args, error = match_qq_social_control_request(
        source=source,
        body="把自动通过好友申请打开，通知发我私聊。",
        admin_ids_configured=True,
        is_admin_user=True,
        admin_only_message="admin only",
        looks_like_request_list_query=lambda body: False,
        looks_like_policy_candidate=lambda body: True,
        looks_like_policy_query=lambda body: False,
        request_type_matcher=lambda body: None,
        notify_target_resolver=lambda current_source, body: "current_user_dm",
    )

    assert error is None
    assert tool_args == {
        "action": "set_social_policy",
        "auto_approve_friend_requests": True,
        "notify_target": "current_user_dm",
    }


def test_match_qq_social_control_request_rejects_non_admin():
    source = _make_source(user_id="555")

    tool_args, error = match_qq_social_control_request(
        source=source,
        body="看看待处理的好友申请",
        admin_ids_configured=True,
        is_admin_user=False,
        admin_only_message="admin only",
        looks_like_request_list_query=lambda body: True,
        looks_like_policy_candidate=lambda body: False,
        looks_like_policy_query=lambda body: False,
        request_type_matcher=lambda body: "friend",
        notify_target_resolver=lambda current_source, body: None,
    )

    assert tool_args is None
    assert error == "admin only"
