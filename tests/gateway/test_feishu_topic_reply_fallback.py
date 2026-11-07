"""Topic-targeted sends must reach a Feishu topic via the reply API.

``POST /im/v1/messages`` rejects ``receive_id_type="thread_id"`` with
``99992402 field validation failed``, so a delivery target of
``feishu:<chat_id>:<thread_id>`` cannot be created directly.  The adapter has to
reply into the topic through its last message instead (``reply_in_thread``),
which is also how the audio path has always handled topics.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import types

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")

_PAYLOAD = '{"zh_cn":{"content":[[{"tag":"md","text":"x"}]]}}'


class _Response:
    def __init__(self, items=None):
        self.data = types.SimpleNamespace(message_id="om_sent", items=items or [])
        self.code = 0
        self.msg = "success"

    def success(self):
        return True


class _MessageAPI:
    """Records the SDK calls the adapter makes and answers them like Feishu does."""

    def __init__(self, *, thread_has_messages: bool):
        self.thread_has_messages = thread_has_messages
        self.calls: list[tuple[str, bool]] = []
        self.message = self

    def list(self, request):
        items = [types.SimpleNamespace(message_id="om_anchor")] if self.thread_has_messages else []
        return _Response(items=items)

    def reply(self, request):
        self.calls.append(("reply", bool(request.request_body.reply_in_thread)))
        return _Response()

    def create(self, request):
        self.calls.append(("create", False))
        return _Response()


def _bare_adapter(message_api: _MessageAPI):
    adapter = object.__new__(_adapter.FeishuAdapter)
    adapter._client = types.SimpleNamespace(im=types.SimpleNamespace(v1=message_api))
    adapter._sdk_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return adapter


def _send(adapter, metadata):
    return asyncio.run(
        adapter._send_raw_message(
            chat_id="oc_chat", msg_type="post", payload=_PAYLOAD,
            reply_to=None, metadata=metadata,
        )
    )


def test_topic_target_replies_into_thread():
    message_api = _MessageAPI(thread_has_messages=True)
    _send(_bare_adapter(message_api), {"thread_id": "omt_topic"})
    assert message_api.calls == [("reply", True)], message_api.calls


def test_topic_without_messages_falls_back_to_flat_chat():
    message_api = _MessageAPI(thread_has_messages=False)
    _send(_bare_adapter(message_api), {"thread_id": "omt_empty"})
    assert message_api.calls == [("create", False)], message_api.calls


def test_plain_chat_target_still_creates():
    message_api = _MessageAPI(thread_has_messages=False)
    _send(_bare_adapter(message_api), None)
    assert message_api.calls == [("create", False)], message_api.calls
