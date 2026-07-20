from gateway.config import Platform
from gateway.group_target_intents import (
    extract_qq_group_target,
    extract_recent_target_from_history,
    extract_weixin_group_target,
    resolve_current_group_target_reference,
)
from gateway.session import SessionSource


def test_resolve_current_group_target_reference_returns_formatted_current_group():
    source = SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="726109087",
        chat_type="group",
    )

    result = resolve_current_group_target_reference(
        source,
        "这个群只监听，不要聊天",
        expected_platform=Platform.QQ_NAPCAT,
        validator=lambda chat_id: bool(chat_id),
        formatter=lambda chat_id: f"group:{chat_id}",
    )

    assert result == "group:726109087"


def test_resolve_current_group_target_reference_rejects_non_matching_source():
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="179033731",
        user_name="發發發",
        chat_id="project@chatroom",
        chat_type="group",
    )

    result = resolve_current_group_target_reference(
        source,
        "这个群只监听，不要聊天",
        expected_platform=Platform.QQ_NAPCAT,
        validator=lambda chat_id: bool(chat_id),
        formatter=lambda chat_id: f"group:{chat_id}",
    )

    assert result is None


def test_extract_recent_target_from_history_applies_predicate_before_extractor():
    source = SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="179033731",
        chat_type="dm",
    )
    history = [
        {"role": "assistant", "content": "可以，把要发的内容直接发我。"},
        {"role": "user", "content": "你现在能在 群 192903718 发送一句话吗"},
    ]

    result = extract_recent_target_from_history(
        source,
        history,
        extractor=lambda _source, content: "group:192903718" if "192903718" in content else None,
        predicate=lambda item, content: item.get("role") == "user" and "发送一句话" in content,
    )

    assert result == "group:192903718"


def test_extract_recent_target_from_history_returns_empty_when_no_match():
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="179033731",
        user_name="發發發",
        chat_id="wxid_admin",
        chat_type="dm",
    )

    result = extract_recent_target_from_history(
        source,
        [{"role": "assistant", "content": "没有目标群。"}],
        extractor=lambda _source, _content: None,
    )

    assert result == ""


def test_extract_qq_group_target_prefers_explicit_numeric_group_id():
    source = SessionSource(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        user_name="發發發",
        chat_id="179033731",
        chat_type="dm",
    )

    result = extract_qq_group_target(source, "停止QQ 群 192903718 的监听采集")

    assert result == "group:192903718"


def test_extract_weixin_group_target_accepts_chatroom_reference():
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="179033731",
        user_name="發發發",
        chat_id="wxid_admin",
        chat_type="dm",
    )

    result = extract_weixin_group_target(source, "停止微信群 project@chatroom 的监听采集")

    assert result == "project@chatroom"


def test_extract_weixin_group_target_can_fall_back_to_current_group():
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="179033731",
        user_name="發發發",
        chat_id="project@chatroom",
        chat_type="group",
    )

    result = extract_weixin_group_target(source, "这个群只监听，不要聊天")

    assert result == "project@chatroom"
