import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import _reply_anchor_for_event
from gateway.platforms.feishu import FeishuAdapter


def _make_feishu_adapter() -> FeishuAdapter:
    adapter = object.__new__(FeishuAdapter)
    adapter._client = MagicMock()
    adapter._sender_name_cache = {}
    adapter._known_at_mapping = {}
    adapter._bot_open_id = ""
    adapter._bot_name = ""
    return adapter


def test_feishu_thread_reply_anchor_prefers_current_message_id():
    event = SimpleNamespace(
        message_id="topic-msg-7",
        reply_to_message_id="seed-msg-1",
        source=SimpleNamespace(
            platform=Platform.FEISHU,
            thread_id="topic-1",
            chat_type="forum",
        ),
    )

    assert _reply_anchor_for_event(event) == "topic-msg-7"


def test_feishu_outbound_payload_prioritizes_real_at_mentions_over_markdown():
    adapter = _make_feishu_adapter()
    adapter._remember_at_mapping("ou_alice", "alice")

    msg_type, payload = adapter._build_outbound_payload("@alice please check:\n```py\nprint(1)\n```")

    assert msg_type == "post"
    parsed = json.loads(payload)
    assert parsed["zh_cn"]["content"][0][0] == {
        "tag": "at",
        "user_id": "ou_alice",
        "user_name": "alice",
    }
    assert any(
        part.get("tag") == "text"
        for row in parsed["zh_cn"]["content"]
        for part in row
    )


@pytest.mark.asyncio
async def test_feishu_thread_send_falls_back_to_chat_create_when_no_reply_anchor():
    adapter = _make_feishu_adapter()
    create_mock = MagicMock(
        return_value=SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id="om_1"),
        )
    )
    adapter._client.im = SimpleNamespace(
        v1=SimpleNamespace(
            message=SimpleNamespace(
                create=create_mock,
                reply=MagicMock(),
            )
        )
    )
    adapter._build_create_message_body = MagicMock(return_value={"receive_id": "oc_group"})
    adapter._build_create_message_request = MagicMock(return_value="create-request")

    response = await adapter._send_raw_message(
        chat_id="oc_group",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to=None,
        metadata={"thread_id": "topic-9"},
    )

    assert response.success()
    adapter._build_create_message_body.assert_called_once()
    assert adapter._build_create_message_body.call_args.kwargs["receive_id"] == "oc_group"
    adapter._build_create_message_request.assert_called_once_with(
        "chat_id",
        {"receive_id": "oc_group"},
    )
