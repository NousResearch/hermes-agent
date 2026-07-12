"""Tests for Feishu Card v2 long-content splitting."""

import json

from gateway.platforms.feishu_card_renderer import (
    _DEFAULT_MAX_CARD_CHARS,
    _DEFAULT_MAX_ELEMENTS_PER_CARD,
    _DEFAULT_MAX_MARKDOWN_CHARS,
    build_feishu_card_v2_payloads,
)


def test_default_split_thresholds_prioritize_readability_below_api_limits():
    assert _DEFAULT_MAX_MARKDOWN_CHARS == 3000
    assert _DEFAULT_MAX_ELEMENTS_PER_CARD == 120
    assert _DEFAULT_MAX_CARD_CHARS == 6000


def test_long_final_response_splits_into_multiple_cards_without_losing_text():
    text = "\n\n".join(
        f"第 {i:02d} 段：" + ("这是一段用于验证飞书卡片长内容拆分的正文。" * 35)
        for i in range(1, 9)
    )

    payloads = build_feishu_card_v2_payloads(
        text,
        max_markdown_chars=900,
        max_elements_per_card=4,
    )

    assert len(payloads) > 1
    cards = [json.loads(payload) for payload in payloads]
    combined = "\n".join(
        element["content"]
        for card in cards
        for element in card["body"]["elements"]
        if element.get("tag") == "markdown"
    )
    assert "第 01 段" in combined
    assert "第 08 段" in combined
    for card in cards:
        assert len(card["body"]["elements"]) <= 4
        for element in card["body"]["elements"]:
            if element.get("tag") == "markdown":
                assert len(element["content"]) <= 900
    assert cards[0]["header"]["title"]["content"].startswith("Hermes 1/")
    assert cards[-1]["header"]["title"]["content"].startswith(f"Hermes {len(cards)}/")


def test_default_readability_threshold_splits_long_wall_of_text():
    marker = "END-MARKER-默认阈值保留完整内容"
    text = ("这是一段用于验证默认 UX 阈值的长文本。" * 900) + marker

    payloads = build_feishu_card_v2_payloads(text)

    assert len(payloads) > 1
    cards = [json.loads(payload) for payload in payloads]
    combined = "".join(
        element["content"]
        for card in cards
        for element in card["body"]["elements"]
        if element.get("tag") == "markdown"
    )
    assert marker in combined
    for card in cards:
        assert len(card["body"]["elements"]) <= _DEFAULT_MAX_ELEMENTS_PER_CARD
        assert len(json.dumps(card, ensure_ascii=False)) <= _DEFAULT_MAX_CARD_CHARS + 1000
        for element in card["body"]["elements"]:
            if element.get("tag") == "markdown":
                assert len(element["content"]) <= _DEFAULT_MAX_MARKDOWN_CHARS
