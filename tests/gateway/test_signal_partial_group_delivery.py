"""Regression for #133215: a partially delivered group send must not be broadcast again."""
import json
import logging

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.helpers import redact_phone
from gateway.platforms.signal import SignalAdapter
from gateway.platforms.signal_rate_limit import _reset_scheduler


@pytest.fixture(autouse=True)
def signal_scheduler():
    _reset_scheduler()
    yield
    _reset_scheduler()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_kind", ["text", "attachment", "album"])
@pytest.mark.parametrize("failure_type", ["IDENTITY_FAILURE", "RATE_LIMIT_FAILURE", "UNREGISTERED_FAILURE"])
@pytest.mark.parametrize("success_first", [False, True])
async def test_partial_group_delivery_is_not_resent(send_kind, failure_type, success_first, tmp_path, caplog):
    failed_number = "+15550000002"
    recipients = [
        {"recipientAddress": {"number": failed_number}, "type": failure_type},
        {"recipientAddress": {"number": "+15550000001"}, "type": "SUCCESS"},
    ]
    if success_first:
        recipients.reverse()
    timestamp = 1712345678000
    sends = []

    def respond(request):
        payload = json.loads(request.content)
        if payload["method"] == "send":
            sends.append(payload["params"])
        return httpx.Response(200, json={"result": {"timestamp": timestamp, "results": recipients}})

    adapter = SignalAdapter(PlatformConfig(extra={"account": "+15550000000"}))
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.signal"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter.client = client
            if send_kind == "text":
                result = await adapter._send_with_retry("group:test-group", "**hello**")
            elif send_kind == "attachment":
                result = await adapter.send_image_file("group:test-group", str(image), caption="hello")
            else:
                result = await adapter.send_multiple_images("group:test-group", [(image.as_uri(), "hello")])

    assert result.success
    assert len(sends) == 1, "reachable group members must receive only one copy"
    assert sends[0]["groupId"] == "test-group"
    assert timestamp in adapter._recent_sent_timestamps
    assert failure_type in caplog.text
    assert redact_phone(failed_number) in caplog.text
    assert failed_number not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["group:test-group", "00000000-0000-0000-0000-000000000001"])
@pytest.mark.parametrize("failure_type", ["IDENTITY_FAILURE", "RATE_LIMIT_FAILURE", "UNREGISTERED_FAILURE"])
@pytest.mark.parametrize("error_envelope", [False, True])
async def test_total_recipient_failure_stays_failed_without_plain_text_fallback(chat_id, failure_type, error_envelope):
    sends = []

    def respond(request):
        payload = json.loads(request.content)
        if payload["method"] == "send":
            sends.append(payload["params"])
        entry = {"type": failure_type}
        if failure_type == "RATE_LIMIT_FAILURE":
            entry["retryAfterSeconds"] = 120
        response = {"timestamp": 1712345678000, "results": [entry]}
        envelope = {"result": response}
        if error_envelope:
            # signal-cli v0.14.7 throws after writing recipient outcomes when none succeeded.
            code = {"IDENTITY_FAILURE": -4, "RATE_LIMIT_FAILURE": -5, "UNREGISTERED_FAILURE": -1}[failure_type]
            envelope = {"error": {"code": code, "message": "Failed to send message",
                                  "data": {"response": response}}}
        return httpx.Response(200, json=envelope)

    adapter = SignalAdapter(PlatformConfig(extra={"account": "+15550000000"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter.client = client
        result = await adapter._send_with_retry(chat_id, "**hello**", max_retries=1)

    assert not result.success
    assert result.error == failure_type
    if failure_type == "RATE_LIMIT_FAILURE":
        assert result.retry_after == 120
        assert result.error_kind == "rate_limited"
    assert len(sends) == 1, "recipient refusals cannot be repaired by stripping formatting"
    assert not adapter._recent_sent_timestamps


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    None, {}, {"timestamp": 1712345678000}, {"results": []}, {"results": [None, {}]},
    {"results": [{"type": "SUCCESS"}]},
    {"results": [{"type": "SUCCESS"}, {"type": "IDENTITY_FAILURE"}]},
])
async def test_rpc_error_cannot_be_promoted_to_success(response):
    def respond(request):
        return httpx.Response(200, json={"error": {
            "code": -1, "message": "Failed to send message", "data": {"response": response},
        }})

    adapter = SignalAdapter(PlatformConfig(extra={"account": "+15550000000"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter.client = client
        result = await adapter.send("group:test-group", "hello")

    assert not result.success
    assert not adapter._recent_sent_timestamps
