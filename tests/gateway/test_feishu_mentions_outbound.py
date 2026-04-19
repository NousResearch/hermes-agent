import json

from gateway.platforms.feishu import FeishuAdapter, _build_rich_post_payload


def test_build_rich_post_payload_serializes_structured_mentions() -> None:
    payload = json.loads(_build_rich_post_payload("你好 @[龙虾](ou_test)"))
    row = payload["zh_cn"]["content"][0]
    assert row == [
        {"tag": "text", "text": "你好 "},
        {"tag": "at", "user_id": "ou_test", "user_name": "龙虾"},
    ]


def test_build_outbound_payload_prefers_post_when_mentions_present() -> None:
    adapter = object.__new__(FeishuAdapter)
    msg_type, payload = FeishuAdapter._build_outbound_payload(adapter, "@ [broken]\n@[龙虾](ou_test)")
    assert msg_type == "post"
    parsed = json.loads(payload)
    assert parsed["zh_cn"]["content"][1] == [{"tag": "at", "user_id": "ou_test", "user_name": "龙虾"}]


def test_build_rich_post_payload_accepts_feishu_at_tag_markup() -> None:
    payload = json.loads(_build_rich_post_payload('hello <at user_id="ou_real">龙虾</at>'))
    row = payload["zh_cn"]["content"][0]
    assert row == [
        {"tag": "text", "text": "hello "},
        {"tag": "at", "user_id": "ou_real", "user_name": "龙虾"},
    ]
