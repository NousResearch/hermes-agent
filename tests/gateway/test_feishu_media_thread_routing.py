"""Feishu media sent into a topic must leave as a *reply*, never as a create with
``receive_id_type=thread_id``.

Feishu answers ``99992402 field validation failed`` for the create-with-thread_id shape —
measured on both text and file against a live DM topic — so media whose delivery carries a
``thread_id`` but no reply anchor (post-turn uploads) used to fail 100% and was reported to
nobody. Routing it as a reply to the thread's last message is the fix; when the tail cannot be
resolved the message must still go out (chat-level), never into the invalid shape.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FileAPI:
    def create(self, request):  # pragma: no cover - request body not under test
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(file_key="file_123"))


class _MessageAPI:
    """Records which Feishu API each send used."""

    def __init__(self, tail: str = "om_tail"):
        self.tail = tail
        self.calls: list[tuple[str, object]] = []

    def list(self, request):
        items = [SimpleNamespace(message_id=self.tail)] if self.tail else []
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=items))

    def reply(self, request):
        self.calls.append(("reply", request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))

    def create(self, request):
        self.calls.append(("create", request))
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_created"))


def _adapter(tail: str = "om_tail") -> tuple[FeishuAdapter, _MessageAPI]:
    adapter = FeishuAdapter(PlatformConfig())
    messages = _MessageAPI(tail=tail)
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(file=_FileAPI(), message=messages))
    )
    return adapter, messages


def _send_into_topic(adapter: FeishuAdapter):
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 test")
        path = tmp.name
    try:
        return asyncio.run(
            adapter.send_document(chat_id="oc_chat", file_path=path, metadata={"thread_id": "omt_thread"})
        )
    finally:
        os.unlink(path)


def test_topic_media_replies_to_thread_tail():
    adapter, messages = _adapter(tail="om_tail")
    result = _send_into_topic(adapter)

    assert result.success, result.error
    kinds = [kind for kind, _ in messages.calls]
    assert kinds == ["reply"], f"expected a single reply to the thread tail, got {kinds}"
    body = messages.calls[0][1].request_body
    assert body.reply_in_thread is True


def test_topic_media_without_resolvable_tail_still_sends():
    adapter, messages = _adapter(tail="")
    result = _send_into_topic(adapter)

    assert result.success, result.error
    kinds = [kind for kind, _ in messages.calls]
    assert kinds == ["create"], f"expected a chat-level create, got {kinds}"


def test_no_create_ever_uses_thread_id_as_receive_id():
    """The invalid shape must be gone from both paths."""
    adapter, messages = _adapter(tail="")
    _send_into_topic(adapter)

    for kind, request in messages.calls:
        assert kind != "create" or getattr(request, "receive_id_type", "") != "thread_id", (
            f"create used receive_id_type=thread_id: {request}"
        )
