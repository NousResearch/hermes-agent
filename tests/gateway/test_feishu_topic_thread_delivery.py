"""Regression: topic-thread sends were silently swallowed with [99992402].

Anchorless sends inside a topic (synthetic wakes, watch-pattern injections, cron)
used to take ``im.v1.message.create`` with ``receive_id_type="thread_id"`` — an
unsupported parameter that the API rejects with ``[99992402] field validation
failed`` for every message type, so the message (and its plain-text fallback)
vanished silently. Now such sends re-anchor on a message inside the topic and
reply in-thread; withdrawn/invalid reply anchors re-anchor instead of dropping;
the plain-text fallback strips markdown and chunks instead of rebuilding the
failing post payload.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def _response(*, ok: bool, code: int = 0, msg: str = "ok", message_id: str = "om_new") -> Any:
    data = SimpleNamespace(message_id=message_id) if ok else None
    return SimpleNamespace(success=lambda: ok, code=code, msg=msg, data=data)


class _RecordingMessageAPI:
    """Fake ``im.v1.message`` recording every create/reply call."""

    def __init__(self, reply_results: Optional[list] = None):
        self.sent: list[tuple[str, Any]] = []
        self._reply_results = list(reply_results or [])

    def reply(self, request: Any) -> Any:
        self.sent.append(("reply", request))
        if self._reply_results:
            return self._reply_results.pop(0)
        return _response(ok=True)

    def create(self, request: Any) -> Any:
        self.sent.append(("create", request))
        return _response(ok=True)


def _adapter_with(message_api: _RecordingMessageAPI) -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=message_api)))

    async def _run_direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def _no_thread_anchor(thread_id: str) -> Optional[str]:
        return "om_anchor"

    adapter._run_blocking = _run_direct
    adapter._fetch_last_message_in_thread = _no_thread_anchor
    return adapter


def _kind(message_api: _RecordingMessageAPI) -> list[str]:
    return [kind for kind, _request in message_api.sent]


def _create_receive_id_types(message_api: _RecordingMessageAPI) -> list[str]:
    return [str(getattr(request, "receive_id_type", "")) for kind, request in message_api.sent if kind == "create"]


def test_anchorless_topic_send_replies_inside_thread_instead_of_invalid_create():
    message_api = _RecordingMessageAPI()
    adapter = _adapter_with(message_api)

    result = asyncio.run(adapter.send("oc_chat", "topic reply", metadata={"thread_id": "omt_topic"}))

    assert result.success
    assert _kind(message_api) == ["reply"]
    kind, request = message_api.sent[0]
    assert request.message_id == "om_anchor"
    assert request.request_body.reply_in_thread is True
    # The regression: create must never see receive_id_type="thread_id" again.
    assert "thread_id" not in _create_receive_id_types(message_api)


def test_anchorless_topic_send_without_anchor_delivers_top_level_instead_of_dropping():
    message_api = _RecordingMessageAPI()
    adapter = _adapter_with(message_api)

    async def _no_anchor(thread_id: str) -> Optional[str]:
        return None

    adapter._fetch_last_message_in_thread = _no_anchor

    result = asyncio.run(adapter.send("oc_chat", "topic reply", metadata={"thread_id": "omt_topic"}))

    assert result.success  # delivered, never silently dropped
    assert _kind(message_api) == ["create"]
    assert _create_receive_id_types(message_api) == ["chat_id"]


@pytest.mark.parametrize("withdrawn_code", [230011, 99992354])
def test_withdrawn_reply_anchor_reanchors_on_fresh_thread_message(withdrawn_code: int):
    message_api = _RecordingMessageAPI(
        reply_results=[_response(ok=False, code=withdrawn_code, msg="target gone"), _response(ok=True)],
    )
    adapter = _adapter_with(message_api)

    async def _fresh_anchor(thread_id: str) -> Optional[str]:
        return "om_fresh"

    adapter._fetch_last_message_in_thread = _fresh_anchor

    result = asyncio.run(
        adapter.send("oc_chat", "topic reply", reply_to="om_withdrawn", metadata={"thread_id": "omt_topic"}),
    )

    assert result.success
    anchors = [request.message_id for kind, request in message_api.sent if kind == "reply"]
    assert anchors == ["om_withdrawn", "om_fresh"]


def test_plain_text_fallback_strips_markdown_and_chunks_instead_of_rebuilding_post():
    message_api = _RecordingMessageAPI()
    adapter = _adapter_with(message_api)
    markdown = "# Title\n\n" + "\n\n".join(
        f"## Section {n}\n\n**bold {n}** with `code` and a [link](https://example.com)\n\n- item\n"
        for n in range(1, 400)
    )
    assert len(markdown) > adapter.MAX_MESSAGE_LENGTH

    result = asyncio.run(
        adapter._send_plain_fallback("oc_chat", markdown, reply_to=None, metadata=None),
    )

    assert result.success
    assert len(message_api.sent) > 1  # long content was chunked, not truncated away
    for _kind_name, request in message_api.sent:
        assert request.request_body.msg_type == "text"
        text = json.loads(request.request_body.content)["text"]
        assert "**" not in text and "](" not in text  # markdown stripped
