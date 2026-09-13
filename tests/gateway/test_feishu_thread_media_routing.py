"""Feishu topic-thread routing: media/anchor metadata and reply-API fallback.

Real-API facts under test (verified against open.feishu.cn):
- create-message with receive_id_type='thread_id' returns 99992402 for ALL
  msg_types — 'thread_id' is not a documented receive_id_type enum value.
- reply API with an in-thread anchor succeeds and lands in the topic.

So the gateway must carry a reply anchor for Feishu thread sends
(_thread_metadata_for_source), and the adapter must fall back to the reply
API when a thread_id create is still attempted.
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import _thread_metadata_for_source


def _feishu_source(thread_id="omt_abc123", message_id="om_root", chat_type="group"):
    return SimpleNamespace(
        platform=SimpleNamespace(value="feishu") if not isinstance(thread_id, str) else SimpleNamespace(value="feishu"),
        chat_id="oc_chat1",
        chat_name="topic group",
        chat_type=chat_type,
        user_id="ou_user",
        user_name="user",
        thread_id=thread_id,
        message_id=message_id,
        reply_to_message_id=None,
        scope_id=None,
        is_bot=False,
    )


class TestThreadMetadataForSource:
    def test_feishu_thread_carries_reply_anchor(self):
        """Root fix: Feishu thread metadata must carry reply_to_message_id
        so media sends route through the reply API, not the dead thread_id
        create path."""
        source = _feishu_source()
        meta = _thread_metadata_for_source(source)
        assert meta["thread_id"] == "omt_abc123"
        assert meta["reply_to_message_id"] == "om_root"

    def test_feishu_thread_explicit_reply_anchor_wins(self):
        source = _feishu_source()
        meta = _thread_metadata_for_source(source, reply_to_message_id="om_parent")
        assert meta["reply_to_message_id"] == "om_parent"

    def test_feishu_no_thread_no_anchor(self):
        source = _feishu_source(thread_id=None)
        meta = _thread_metadata_for_source(source)
        assert meta in (None, {})

    def test_feishu_thread_without_message_id_has_no_anchor(self):
        source = _feishu_source(message_id=None)
        meta = _thread_metadata_for_source(source)
        assert meta["thread_id"] == "omt_abc123"
        assert "reply_to_message_id" not in meta


class TestSendRawMessageThreadFallback:
    """_send_raw_message must fall back to the reply API when the thread_id
    create is rejected with 99992402 — for ALL message types."""

    def _make_adapter(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter, _FEISHU_THREAD_ROUTE_INVALID_CODE

        adapter = object.__new__(FeishuAdapter)
        self._rejected = SimpleNamespace(code=_FEISHU_THREAD_ROUTE_INVALID_CODE, success=False, msg="field validation failed")
        self._reply_ok = SimpleNamespace(code=0, success=True, msg="success", data=SimpleNamespace(message_id="om_new"))
        self._create_fn = SimpleNamespace()  # marker for self._client.im.v1.message.create
        self._reply_fn = SimpleNamespace()   # marker for self._client.im.v1.message.reply

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
                create=self._create_fn, reply=self._reply_fn))),
        )

        async def fake_run_blocking(fn, req):
            if fn is self._create_fn:
                return self._rejected
            return self._reply_ok

        adapter._run_blocking = fake_run_blocking
        adapter._build_create_message_body = lambda **kw: kw
        adapter._build_create_message_request = lambda rtype, body: SimpleNamespace(receive_id_type=rtype)
        adapter._build_reply_message_body = lambda **kw: kw
        adapter._build_reply_message_request = lambda anchor, body: SimpleNamespace(anchor=anchor)
        adapter._response_succeeded = lambda r: getattr(r, "code", None) == 0
        adapter._fetch_last_message_in_thread = AsyncMock(return_value="om_anchor")
        return adapter

    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type,payload", [
        ("image", '{"image_key": "img_v3_x"}'),
        ("text", '{"text": "hi"}'),
        ("post", '{"post": {}}'),
        ("interactive", '{}'),
    ])
    async def test_rejected_thread_create_falls_back_to_reply(self, msg_type, payload):
        adapter = self._make_adapter()
        response = await adapter._send_raw_message(
            chat_id="oc_chat1",
            msg_type=msg_type,
            payload=payload,
            reply_to=None,
            metadata={"thread_id": "omt_abc123"},
        )
        assert getattr(response, "code", None) == 0

    @pytest.mark.asyncio
    async def test_no_anchor_returns_original_failure(self):
        adapter = self._make_adapter()
        adapter._fetch_last_message_in_thread = AsyncMock(return_value=None)
        from plugins.platforms.feishu.adapter import _FEISHU_THREAD_ROUTE_INVALID_CODE

        response = await adapter._send_raw_message(
            chat_id="oc_chat1",
            msg_type="image",
            payload='{"image_key": "img_v3_x"}',
            reply_to=None,
            metadata={"thread_id": "omt_abc123"},
        )
        assert getattr(response, "code", None) == _FEISHU_THREAD_ROUTE_INVALID_CODE


class TestApprovalExpiredNoticeThreadRouting:
    """The expired-approval notice must reuse the approval card's thread
    metadata so it lands in the topic, not the main chat."""

    @pytest.mark.asyncio
    async def test_notice_carries_thread_metadata(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = object.__new__(FeishuAdapter)
        adapter._approval_state = {
            1: {
                "session_key": "feishu:oc_chat1:omt_abc123",
                "message_id": "om_card",
                "chat_id": "oc_chat1",
                "thread_metadata": {"thread_id": "omt_abc123", "reply_to_message_id": "om_root"},
            },
        }
        adapter._is_interactive_operator_authorized = lambda open_id: True
        sent = {}

        def fake_resolve(session_key, choice):
            return 0

        async def fake_send(chat_id, content, reply_to=None, metadata=None):
            sent["metadata"] = metadata
            return SimpleNamespace(success=True)

        import tools.approval as approval_mod
        original = approval_mod.resolve_gateway_approval
        approval_mod.resolve_gateway_approval = fake_resolve
        adapter.send = fake_send
        try:
            await adapter._resolve_approval(
                approval_id=1, choice="approve_once",
                user_name="user", open_id="ou_user", chat_id="oc_chat1",
            )
        finally:
            approval_mod.resolve_gateway_approval = original

        assert sent["metadata"] == {"thread_id": "omt_abc123", "reply_to_message_id": "om_root"}
