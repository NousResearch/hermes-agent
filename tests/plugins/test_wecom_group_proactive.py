"""Known group classification must not prevent native delivery (#140883)."""

import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import APP_CMD_SEND, WeComAdapter


@pytest_asyncio.fixture
async def adapter():
    instance = WeComAdapter(PlatformConfig(enabled=True))
    try:
        yield instance
    finally:
        workers = list(instance._chat_workers.values()) + list(instance._control_workers.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        assert all(worker.done() for worker in workers)


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["unknown", "known", "stale", "fresh", "explicit"])
async def test_send_uses_native_route_without_rejecting_known_groups(context, adapter):
    chat_id = "fixture-group"
    if context != "unknown":
        adapter._group_chat_ids.add(chat_id)
    if context in {"stale", "fresh", "explicit"}:
        adapter._remember_chat_req_id(chat_id, "cached-request")
    if context == "explicit":
        adapter._reply_req_ids["original-message"] = "explicit-request"
    ack = {"errcode": 0, "headers": {"req_id": "delivery-receipt"}}
    adapter._send_request = AsyncMock(return_value=ack)
    adapter._send_reply_request = AsyncMock(
        return_value=ack,
        side_effect=RuntimeError("invalid req_id") if context == "stale" else None,
    )

    result = await adapter.send(
        chat_id, "fixture notification",
        reply_to="original-message" if context == "explicit" else None,
    )

    assert result.success, result.error
    body = {"msgtype": "markdown", "markdown": {"content": "fixture notification"}}
    if context in {"fresh", "explicit"}:
        adapter._send_request.assert_not_awaited()
        adapter._send_reply_request.assert_awaited_once_with(
            "explicit-request" if context == "explicit" else "cached-request", body,
        )
    else:
        adapter._send_request.assert_awaited_once_with(APP_CMD_SEND, {"chatid": chat_id, **body})
        if context == "stale":
            adapter._send_reply_request.assert_awaited_once_with("cached-request", body)
        else:
            adapter._send_reply_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rejected", "exception", "timeout"])
async def test_known_group_reports_native_failure_without_retry(failure, adapter):
    adapter._group_chat_ids.add("fixture-group")
    errors = {
        "rejected": "WeCom errcode 600039: device type not support",
        "exception": "transport refused",
        "timeout": "Timeout sending message to WeCom",
    }
    adapter._send_request = AsyncMock(
        return_value={"errcode": 600039, "errmsg": "device type not support"},
        side_effect={"exception": RuntimeError("transport refused"), "timeout": TimeoutError()}.get(failure),
    )

    result = await adapter.send("fixture-group", "fixture notification")

    assert result.success is False
    assert result.error == errors[failure]
    adapter._send_request.assert_awaited_once()
