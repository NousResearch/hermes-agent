"""Yuanbao send ACK: business rejection must not look like successful delivery.

Covers issue #107227 — Yuanbao can return head.status=0 with a nonzero
business code (observed production: code=999 / inner 9992501) while
``MessageSender._dispatch_encoded`` still reported success. Also pins the
conservative 1200-char outbound chunk target and refuses leftover
unsplittable markdown blocks larger than ``MAX_TEXT_CHUNK``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.yuanbao import MessageSender, YuanbaoAdapter
from gateway.platforms.yuanbao_proto import _s, _v


def _run(coro):
    return asyncio.run(coro)


def _adapter_for_dispatch(response):
    async def send_biz_request(encoded, req_id, timeout=None):  # noqa: ANN001, ARG001
        return response

    return SimpleNamespace(
        name="yuanbao",
        _connection=SimpleNamespace(send_biz_request=send_biz_request),
    )


def _dispatch(response, encoded=b"encoded", req_id="c2c_1"):
    return _run(MessageSender._dispatch_encoded(_adapter_for_dispatch(response), encoded, req_id))


def test_business_rejection_status0_code999_is_not_success():
    """THE bug: head.status=0 + business code=999 must fail closed."""
    biz = _v(1, 999) + _s(2, '{"code":9992501,"message":"请求云 IM 服务返回失败"}')
    resp = {"head": {"status": 0, "msg_id": "c2c_1"}, "data": biz}
    result = _dispatch(resp)
    assert result["success"] is False
    assert "999" in str(result.get("error", ""))


def test_protocol_header_rejection_nonzero_status():
    resp = {"head": {"status": 1, "msg_id": "c2c_2"}, "data": _v(1, 0)}
    result = _dispatch(resp, req_id="c2c_2")
    assert result["success"] is False


def test_valid_success_ack_uses_head_msg_id():
    resp = {"head": {"status": 0, "msg_id": "c2c_ok"}, "data": _v(1, 0) + _s(2, "ok")}
    result = _dispatch(resp, req_id="c2c_ok")
    assert result["success"] is True
    assert result["msg_key"] == "c2c_ok"


def test_fail_open_status0_no_data_is_success():
    resp = {"head": {"status": 0, "msg_id": "c2c_empty"}}
    result = _dispatch(resp, req_id="c2c_empty")
    assert result["success"] is True


def test_fail_open_empty_dict_is_success():
    result = _dispatch({})
    assert result == {"success": True, "msg_key": ""}


@pytest.mark.parametrize("response", [None, [], "unexpected", 42])
def test_non_dict_send_response_is_malformed(response):
    result = _dispatch(response)
    assert result == {"success": False, "error": "malformed send response"}


def test_malformed_nonempty_truncated_varint_is_not_success():
    resp = {"head": {"status": 0, "msg_id": "c2c_bad"}, "data": b"\x08\x80"}
    result = _dispatch(resp, req_id="c2c_bad")
    assert result["success"] is False


def test_chunk_target_is_1200_and_splits_2491_chars():
    assert YuanbaoAdapter.MAX_TEXT_CHUNK == 1200
    chunks = MessageSender.truncate_message("汉" * 2491, YuanbaoAdapter.MAX_TEXT_CHUNK)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)


def test_oversized_unsplittable_fence_is_not_dispatched():
    """A single fence larger than MAX_TEXT_CHUNK must fail closed, not send."""
    limit = YuanbaoAdapter.MAX_TEXT_CHUNK
    fence = "```\n" + ("x" * (limit + 80)) + "\n```"
    leftover = MessageSender.truncate_message(fence, limit)
    assert leftover
    assert any(len(c) > limit for c in leftover)

    dispatched = []

    async def _should_not_send(*args, **kwargs):  # noqa: ANN001, ARG001
        dispatched.append(True)
        raise AssertionError("oversized leftover chunk must not be dispatched")

    adapter = SimpleNamespace(
        name="yuanbao",
        MAX_TEXT_CHUNK=limit,
        _connection=SimpleNamespace(ws=object()),
        _outbound=SimpleNamespace(
            slow_notifier=SimpleNamespace(cancel=lambda chat_id: None),
            heartbeat=SimpleNamespace(send_heartbeat_once=lambda *a, **k: None),
        ),
    )
    sender = MessageSender(adapter)
    sender.send_text_chunk = _should_not_send  # type: ignore[method-assign]
    result = _run(sender.send_text("direct:peer", fence))
    assert result.success is False
    assert not dispatched
    err = str(result.error or "").lower()
    assert "direct:peer" not in err
    assert "```" not in err
    assert "xxxx" not in err
