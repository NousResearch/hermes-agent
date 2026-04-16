from gateway.send_intents import (
    extract_qq_inline_send_target_and_message,
    extract_send_confirmation_message,
    extract_weixin_inline_send_target_and_message,
    looks_like_send_confirmation,
    looks_like_send_query,
)


def test_looks_like_send_query_matches_existing_terms():
    assert looks_like_send_query("你现在能在 群 192903718 发送一句话吗")
    assert not looks_like_send_query("这个群只监听，不要走大模型")


def test_looks_like_send_confirmation_matches_existing_terms():
    assert looks_like_send_confirmation("绿帽哥!\n\n发这句")
    assert not looks_like_send_confirmation("往 QQ 群 192903718 发：绿帽哥！")


def test_extract_send_confirmation_message_removes_confirmation_suffix():
    assert extract_send_confirmation_message("绿帽哥!\n\n发这句") == "绿帽哥!"


def test_extract_qq_inline_send_target_and_message_parses_group_send():
    target, message = extract_qq_inline_send_target_and_message("往 QQ 群 192903718 发：绿帽哥！")

    assert target == "group:192903718"
    assert message == "绿帽哥！"


def test_extract_weixin_inline_send_target_and_message_parses_group_send():
    target, message = extract_weixin_inline_send_target_and_message("往 微信群 project@chatroom 发：开会了")

    assert target == "project@chatroom"
    assert message == "开会了"
