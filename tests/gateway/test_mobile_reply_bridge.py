from pathlib import Path

from gateway.mobile_reply_bridge import (
    create_mobile_ask,
    format_mobile_ask_card,
    handle_mobile_reply_text,
    load_latest_reply,
    looks_like_mobile_reply_missing_id,
    parse_mobile_reply,
)


def test_parse_mobile_reply_with_chinese_prefix():
    parsed = parse_mobile_reply("回复 HERMES-ASK-20260517-120000-ABC123 同意继续")

    assert parsed is not None
    assert parsed.request_id == "HERMES-ASK-20260517-120000-ABC123"
    assert parsed.content == "同意继续"


def test_parse_mobile_reply_without_prefix():
    parsed = parse_mobile_reply("HERMES-ASK-20260517-120000-ABC123 降到 2 workers")

    assert parsed is not None
    assert parsed.request_id == "HERMES-ASK-20260517-120000-ABC123"
    assert parsed.content == "降到 2 workers"


def test_missing_id_prompt_only_for_reply_intent():
    assert looks_like_mobile_reply_missing_id("回复 同意") is True
    assert looks_like_mobile_reply_missing_id("回复我一下") is False
    assert looks_like_mobile_reply_missing_id("普通聊天") is False


def test_handle_mobile_reply_records_latest(tmp_path: Path):
    record = create_mobile_ask(
        title="云端卡点",
        question="是否继续？",
        hermes_home=tmp_path,
        request_id="HERMES-ASK-20260517-120000-ABC123",
    )
    card = format_mobile_ask_card(record)

    assert "回复 HERMES-ASK-20260517-120000-ABC123 你的指令" in card

    result = handle_mobile_reply_text(
        "回复 HERMES-ASK-20260517-120000-ABC123 继续，但只开 2 workers",
        platform="weixin",
        chat_id="filehelper",
        user_id="tester",
        hermes_home=tmp_path,
    )

    assert result is not None
    assert result.action == "recorded"
    assert result.reply_path is not None
    assert result.reply_path.exists()
    latest = load_latest_reply("HERMES-ASK-20260517-120000-ABC123", hermes_home=tmp_path)
    assert latest is not None
    assert latest["content"] == "继续，但只开 2 workers"
    assert latest["source"]["platform"] == "weixin"


def test_handle_missing_id_returns_guidance(tmp_path: Path):
    result = handle_mobile_reply_text("回复 同意", hermes_home=tmp_path)

    assert result is not None
    assert result.action == "missing_id"
    assert "请带上请求编号" in result.message
