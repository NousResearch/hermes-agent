import json


def test_cardkit_v2_element_kinds_are_extracted():
    from plugins.platforms.feishu.adapter import _normalize_interactive_message

    result = _normalize_interactive_message("interactive", {
        "body": {"elements": [
            {"tag_type": "markdown", "content": "the actual answer"},
            {"element_type": "text_run", "text_run": {"content": "more text"}},
            {"tag_type": "button", "type": "primary", "text": {"content": "Open"}},
        ]},
    })

    assert result.text_content == "the actual answer\nmore text\nOpen"
    assert result.metadata["actions"] == ["Open"]


def test_double_encoded_cardkit_v2_card_is_decoded():
    from plugins.platforms.feishu.adapter import _normalize_interactive_message

    result = _normalize_interactive_message("interactive", {
        "card": json.dumps({"body": {"elements": [{"tag_type": "markdown", "content": "decoded"}]}}),
    })

    assert result.text_content == "decoded"