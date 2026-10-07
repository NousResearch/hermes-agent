"""Unit tests for QQBot expired reply anchor handling in group and C2C chats.

Verifies that when a long-running background task finishes and QQ rejects the
inbound msg_id with '回复消息msg_id已过期' (e.g. group chat > 5 minutes), the adapter
gracefully falls back to sending a standalone message without the expired reply_to.
"""

from types import SimpleNamespace
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.qqbot.adapter import QQAdapter


@pytest.mark.asyncio
async def test_group_expired_reply_anchor_falls_back_to_standalone_message():
    """An expired QQ reply anchor in a group must fall back to standalone message."""
    adapter = QQAdapter(PlatformConfig(enabled=True, token="dummy_token"))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    adapter._chat_type_map["group_123"] = "group"
    calls = []

    async def fake_api_request(method, path, body=None, *, timeout=None):
        calls.append(body)
        if body.get("msg_id"):
            raise RuntimeError(
                "QQ Bot API error [400] /v2/groups/group_123/messages: 回复消息msg_id已过期"
            )
        return {"id": "standalone_resp_ok"}

    adapter._api_request = fake_api_request

    result = await adapter.send(
        "group_123", "这是在群里跑完的超长任务结果", reply_to="EXPIRED_GROUP_MSG_OID"
    )

    assert result.success is True
    assert result.message_id == "standalone_resp_ok"
    assert len(calls) == 2
    # First attempt tried using the reply anchor
    assert calls[0].get("msg_id") == "EXPIRED_GROUP_MSG_OID"
    # Second attempt gracefully retried without msg_id
    assert "msg_id" not in calls[1]
    assert "这是在群里跑完的超长任务结果" in calls[1]["markdown"]["content"]


@pytest.mark.asyncio
async def test_keyboard_expired_reply_anchor_falls_back():
    """Keyboard messages also gracefully retry as standalone when anchor expires."""
    adapter = QQAdapter(PlatformConfig(enabled=True, token="dummy_token"))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    adapter._chat_type_map["group_123"] = "group"
    calls = []

    async def fake_api_request(method, path, body=None, *, timeout=None):
        calls.append(body)
        if body.get("msg_id"):
            raise RuntimeError("QQ Bot API error [400]: msg_id expired")
        return {"id": "keyboard_resp_ok"}

    adapter._api_request = fake_api_request
    fake_keyboard = SimpleNamespace(to_dict=lambda: {"content": {"rows": []}})

    result = await adapter.send_with_keyboard(
        "group_123", "需要审批", fake_keyboard, reply_to="EXPIRED_KEYBOARD_MSG_OID"
    )

    assert result.success is True
    assert result.message_id == "keyboard_resp_ok"
    assert len(calls) == 2
    assert calls[0].get("msg_id") == "EXPIRED_KEYBOARD_MSG_OID"
    assert "msg_id" not in calls[1]
    assert calls[1].get("keyboard") == {"content": {"rows": []}}


def test_is_expired_reply_error_detection():
    assert QQAdapter._is_expired_reply_error(RuntimeError("回复消息msg_id已过期"))
    assert QQAdapter._is_expired_reply_error(RuntimeError("QQ Bot API error [400]: msg_id expired"))
    assert QQAdapter._is_expired_reply_error(RuntimeError("QQ Bot API error [400]: msg_id expire"))
    assert not QQAdapter._is_expired_reply_error(RuntimeError("msg_id missing"))
    assert not QQAdapter._is_expired_reply_error(RuntimeError("invalid request"))
