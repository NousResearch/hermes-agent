from gateway.config import Platform
from gateway.session import SessionSource

from gateway.send_requests import match_send_request


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


def test_match_send_request_returns_inline_target_and_message():
    source = _make_source()

    tool_args, error = match_send_request(
        source=source,
        body="往 QQ 群 192903718 发：绿帽哥！",
        conversation_history=[],
        inline_extractor=lambda body: ("group:192903718", "绿帽哥！"),
        history_target_extractor=lambda source, history: "",
        direct_target_extractor=lambda source, body: None,
        looks_like_send_query=lambda body: False,
        looks_like_send_confirmation=lambda body: False,
        extract_send_confirmation_message=lambda body: "",
        query_prompt_formatter=lambda target: f"prompt:{target}",
    )

    assert error is None
    assert tool_args == {
        "target": "group:192903718",
        "message": "绿帽哥！",
    }


def test_match_send_request_uses_followup_confirmation_target_from_history():
    source = _make_source(platform=Platform.WEIXIN)

    tool_args, error = match_send_request(
        source=source,
        body="收到\n发这句",
        conversation_history=[
            {"role": "user", "content": "你现在能在 微信群 project@chatroom 发送一句话吗"}
        ],
        inline_extractor=lambda body: ("", ""),
        history_target_extractor=lambda source, history: "project@chatroom",
        direct_target_extractor=lambda source, body: None,
        looks_like_send_query=lambda body: False,
        looks_like_send_confirmation=lambda body: True,
        extract_send_confirmation_message=lambda body: "收到",
        query_prompt_formatter=lambda target: f"prompt:{target}",
    )

    assert error is None
    assert tool_args == {
        "target": "project@chatroom",
        "message": "收到",
    }


def test_match_send_request_returns_query_prompt_when_target_can_be_resolved():
    source = _make_source(platform=Platform.WEIXIN)

    tool_args, error = match_send_request(
        source=source,
        body="你现在能在 微信群 project@chatroom 发送一句话吗",
        conversation_history=[],
        inline_extractor=lambda body: ("", ""),
        history_target_extractor=lambda source, history: "",
        direct_target_extractor=lambda source, body: "project@chatroom",
        looks_like_send_query=lambda body: True,
        looks_like_send_confirmation=lambda body: False,
        extract_send_confirmation_message=lambda body: "",
        query_prompt_formatter=lambda target: f"可以，把要发到 {target} 的内容直接发我。",
    )

    assert tool_args is None
    assert error == "可以，把要发到 project@chatroom 的内容直接发我。"
