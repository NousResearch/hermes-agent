"""Feishu topic delivery must never use ``receive_id_type='thread_id'`` on message create.

The Feishu API rejects create-with-``thread_id`` outright with code 99992402 ("field validation
failed"), so it is not a usable fallback: turns woken by an internal notification carry no parent
message id, took that route, and had their final answer dropped with only a warning in the log.

The reply API (with ``reply_in_thread``) is the only working route into a topic, so these assert
the routing CONTRACT through the real ``_send_raw_message`` selection logic: a topic send must
resolve a live anchor and reply to it, and must fall back to the chat rather than lose the message.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


def _resp(ok: bool, code: int = 0) -> Any:
    return SimpleNamespace(success=lambda: ok, code=code, msg="field validation failed")


class _Recorder:
    """Captures how each send attempt was routed by the real _send_raw_message."""

    def __init__(self, codes: list[int], thread_anchors: Optional[list[Optional[str]]] = None) -> None:
        self._codes = codes
        # Anchor ids _fetch_last_message_in_thread yields, in order.
        self._thread_anchors = list(thread_anchors or [])
        self.reply_targets: list[str] = []
        self.create_targets: list[tuple[str, str]] = []
        self.anchor_lookups = 0

    def _next(self) -> Any:
        code = self._codes.pop(0)
        return _resp(code == 0, code)

    def next_anchor(self) -> Optional[str]:
        self.anchor_lookups += 1
        return self._thread_anchors.pop(0) if self._thread_anchors else None

    # The two terminal SDK calls _send_raw_message picks between.
    def reply(self, request: Any) -> Any:
        self.reply_targets.append(getattr(request, "message_id", "?"))
        return self._next()

    def create(self, request: Any) -> Any:
        self.create_targets.append((request.receive_id_type, request.receive_id))
        return self._next()


def _adapter(rec: _Recorder) -> FeishuAdapter:
    a = object.__new__(FeishuAdapter)

    async def _run_blocking(func, *args):
        return func(*args)

    async def _fetch_last_message_in_thread(thread_id):
        return rec.next_anchor()

    a._run_blocking = _run_blocking
    a._fetch_last_message_in_thread = _fetch_last_message_in_thread
    a._build_reply_message_body = lambda **kw: kw
    a._build_reply_message_request = lambda message_id, request_body: SimpleNamespace(
        message_id=message_id, body=request_body,
    )
    a._build_create_message_body = lambda **kw: kw
    a._build_create_message_request = lambda receive_id_type, request_body: SimpleNamespace(
        receive_id_type=receive_id_type, receive_id=request_body["receive_id"], body=request_body,
    )
    a._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=SimpleNamespace(reply=rec.reply, create=rec.create),
    )))
    return a


async def _send(adapter: FeishuAdapter, **kwargs: Any) -> Any:
    defaults: dict[str, Any] = dict(chat_id="oc_chat", msg_type="text", payload='{"text":"hi"}')
    defaults.update(kwargs)
    return await adapter._feishu_send_with_retry(**defaults)


@pytest.mark.asyncio
async def test_anchorless_thread_send_borrows_an_anchor_instead_of_create_with_thread_id():
    """The regression: a notification-woken turn has no parent id and used to be dropped."""
    rec = _Recorder([0], thread_anchors=["om_live"])
    adapter = _adapter(rec)

    result = await _send(adapter, reply_to=None, metadata={"thread_id": "omt_topic"})

    assert result.success() is True, "an anchorless topic send must still be delivered"
    assert rec.reply_targets == ["om_live"], "delivery must route through the reply API"
    assert rec.create_targets == [], "create with receive_id_type='thread_id' is rejected by Feishu"


@pytest.mark.asyncio
async def test_anchorless_thread_send_falls_back_to_chat_when_thread_is_empty():
    """No anchor is recoverable — deliver to the chat rather than lose the message."""
    rec = _Recorder([0], thread_anchors=[None])
    adapter = _adapter(rec)

    result = await _send(adapter, reply_to=None, metadata={"thread_id": "omt_topic"})

    assert result.success() is True
    assert rec.reply_targets == []
    assert rec.create_targets == [("chat_id", "oc_chat")], "must never create against a thread_id"


@pytest.mark.asyncio
async def test_invalid_reply_target_in_thread_retries_with_a_different_anchor():
    """99992402 on the parent id must be recovered inside the topic, not given up on."""
    rec = _Recorder([99992402, 0], thread_anchors=["om_live"])
    adapter = _adapter(rec)

    result = await _send(
        adapter, reply_to="om_dead", metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_dead"},
    )

    assert result.success() is True, "a dead parent id must not end delivery"
    assert rec.reply_targets == ["om_dead", "om_live"], "the retry must use a DIFFERENT anchor"
    assert rec.create_targets == []


@pytest.mark.asyncio
async def test_retry_never_reuses_the_dead_anchor():
    """If the topic's newest message IS the dead target, retrying it would fail identically."""
    rec = _Recorder([99992402, 0], thread_anchors=["om_dead"])
    adapter = _adapter(rec)

    result = await _send(
        adapter, reply_to="om_dead", metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_dead"},
    )

    assert result.success() is True
    assert rec.reply_targets == ["om_dead"], "the known-dead anchor must not be replayed"
    assert rec.create_targets == [("chat_id", "oc_chat")], "fall back to the chat instead"


@pytest.mark.asyncio
async def test_metadata_only_reply_target_is_also_recovered():
    """reply_to=None still reply-routes via metadata, so that failure must recover too."""
    rec = _Recorder([99992402, 0], thread_anchors=["om_live"])
    adapter = _adapter(rec)

    result = await _send(
        adapter, reply_to=None, metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_dead"},
    )

    assert result.success() is True
    assert rec.reply_targets == ["om_dead", "om_live"]
    assert rec.create_targets == []


@pytest.mark.asyncio
async def test_invalid_reply_target_outside_thread_falls_back_to_chat():
    """Without a topic there is nothing to preserve — fall back to a normal chat message."""
    rec = _Recorder([99992402, 0])
    adapter = _adapter(rec)

    result = await _send(adapter, reply_to="om_dead", metadata=None)

    assert result.success() is True
    assert rec.create_targets == [("chat_id", "oc_chat")]
    assert rec.anchor_lookups == 0, "no topic means no anchor lookup"


@pytest.mark.asyncio
async def test_successful_reply_is_not_retried():
    """The recovery path must stay dormant when the reply target is healthy."""
    rec = _Recorder([0])
    adapter = _adapter(rec)

    result = await _send(
        adapter, reply_to="om_live", metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_live"},
    )

    assert result.success() is True
    assert rec.reply_targets == ["om_live"]
    assert rec.create_targets == [], "a healthy reply must not trigger a second send"
    assert rec.anchor_lookups == 0, "a supplied anchor must not cost an API lookup"
