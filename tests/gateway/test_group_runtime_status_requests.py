from gateway.config import Platform
from gateway.session import SessionSource

from gateway.group_runtime_status_requests import match_group_runtime_status_request


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


def test_match_group_runtime_status_request_prefers_direct_target():
    source = _make_source()

    target = match_group_runtime_status_request(
        source=source,
        body="这个群现在谁在监听，日报开了吗？",
        conversation_history=[],
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_group_runtime_status_query=lambda body: True,
        target_extractor=lambda source, body: "group:726109087",
        history_target_extractor=lambda source, history: "group:999999999",
    )

    assert target == "group:726109087"


def test_match_group_runtime_status_request_can_fall_back_to_recent_history_target():
    source = _make_source(chat_type="dm", chat_id="179033731")

    target = match_group_runtime_status_request(
        source=source,
        body="你现在在群里能说话吗 不是监听模式了吗",
        conversation_history=[
            {"role": "user", "content": "往 QQ 群 192903718 发：绿帽哥！"},
        ],
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_group_runtime_status_query=lambda body: True,
        target_extractor=lambda source, body: None,
        history_target_extractor=lambda source, history: "group:192903718",
    )

    assert target == "group:192903718"


def test_match_group_runtime_status_request_does_not_steal_recent_history_without_group_reference():
    source = _make_source(chat_type="dm", chat_id="179033731")

    target = match_group_runtime_status_request(
        source=source,
        body="能说话吗 不是监听模式了吗",
        conversation_history=[
            {"role": "user", "content": "往 QQ 群 192903718 发：绿帽哥！"},
        ],
        admin_ids_configured=True,
        is_admin_user=True,
        looks_like_group_runtime_status_query=lambda body: True,
        target_extractor=lambda source, body: None,
        history_target_extractor=lambda source, history: "group:192903718",
    )

    assert target is None


def test_match_group_runtime_status_request_returns_none_for_non_admin_or_non_query():
    source = _make_source()

    assert (
        match_group_runtime_status_request(
            source=source,
            body="这个群现在谁在监听，日报开了吗？",
            conversation_history=[],
            admin_ids_configured=True,
            is_admin_user=False,
            looks_like_group_runtime_status_query=lambda body: True,
            target_extractor=lambda source, body: "group:726109087",
            history_target_extractor=lambda source, history: "",
        )
        is None
    )
    assert (
        match_group_runtime_status_request(
            source=source,
            body="这个群只监听，不要走大模型。",
            conversation_history=[],
            admin_ids_configured=True,
            is_admin_user=True,
            looks_like_group_runtime_status_query=lambda body: False,
            target_extractor=lambda source, body: "group:726109087",
            history_target_extractor=lambda source, history: "",
        )
        is None
    )
