"""Regression tests for Feishu UUID scope across logical send payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu import adapter as feishu_adapter_module
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeResponse:
    code = 0
    msg = "ok"
    data = SimpleNamespace(message_id="om_1")

    def success(self):
        return True


class _FakeErrorResponse:
    def __init__(self, code):
        self.code = code
        self.msg = "reply target unavailable"

    def success(self):
        return False


def _make_adapter():
    cfg = PlatformConfig(
        enabled=True,
        token="fake",
        typing_indicator=False,
        extra={"final_response_format": "card"},
    )
    adapter = FeishuAdapter(cfg)
    adapter._client = object()
    return adapter


@pytest.mark.asyncio
async def test_retries_reuse_same_uuid_without_metadata_uuid(monkeypatch):
    """Two exception retries and the successful attempt share one bounded UUID."""
    adapter = _make_adapter()
    captured_uuid_values = []

    async def fake_send_raw(*, chat_id, msg_type, payload, reply_to, metadata, uuid_value):
        captured_uuid_values.append(uuid_value)
        if len(captured_uuid_values) < 3:
            raise RuntimeError("transient send failure")
        return _FakeResponse()

    adapter._send_raw_message = fake_send_raw
    sleep = AsyncMock()
    monkeypatch.setattr(feishu_adapter_module.asyncio, "sleep", sleep)

    response = await adapter._feishu_send_with_retry(
        chat_id="oc_1",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to="om_test",
        metadata={"message_id": "om_shared_inbound"},
    )

    assert response.success()
    assert len(captured_uuid_values) == 3
    assert len(set(captured_uuid_values)) == 1
    assert len(captured_uuid_values[0]) <= 50
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_reply_to_create_fallback_reuses_same_uuid():
    """Reply-not-found fallback to create preserves the same dedup UUID."""
    adapter = _make_adapter()
    calls = []
    fallback_code = next(iter(feishu_adapter_module._FEISHU_REPLY_FALLBACK_CODES))

    async def fake_send_raw(*, chat_id, msg_type, payload, reply_to, metadata, uuid_value):
        calls.append((reply_to, uuid_value))
        if reply_to:
            return _FakeErrorResponse(fallback_code)
        return _FakeResponse()

    adapter._send_raw_message = fake_send_raw

    response = await adapter._feishu_send_with_retry(
        chat_id="oc_1",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to="om_missing",
        metadata={"message_id": "om_shared_inbound"},
    )

    assert response.success()
    assert [reply_to for reply_to, _ in calls] == ["om_missing", None]
    assert calls[0][1] == calls[1][1]
    assert len(calls[0][1]) <= 50


@pytest.mark.asyncio
async def test_independent_retry_entries_ignore_shared_metadata_message_id():
    """Independent payloads must never collide because inbound metadata is shared."""
    adapter = _make_adapter()
    captured_uuid_values = []

    async def fake_send_raw(*, chat_id, msg_type, payload, reply_to, metadata, uuid_value):
        captured_uuid_values.append(uuid_value)
        return _FakeResponse()

    adapter._send_raw_message = fake_send_raw
    metadata = {"message_id": "om_shared_inbound"}

    for payload in ('{"text":"first"}', '{"text":"second"}'):
        response = await adapter._feishu_send_with_retry(
            chat_id="oc_1",
            msg_type="text",
            payload=payload,
            reply_to=None,
            metadata=metadata,
        )
        assert response.success()

    assert len(captured_uuid_values) == 2
    assert captured_uuid_values[0] != captured_uuid_values[1]
    assert all(len(value) <= 50 for value in captured_uuid_values)


@pytest.mark.asyncio
async def test_different_card_parts_get_different_uuids():
    """Each adapter-generated card part gets a fresh bounded retry-entry UUID."""
    adapter = _make_adapter()
    captured_calls = []

    async def fake_send_raw(*, chat_id, msg_type, payload, reply_to, metadata, uuid_value):
        captured_calls.append((payload, metadata, uuid_value))
        return _FakeResponse()

    adapter._send_raw_message = fake_send_raw
    long_text = "\n\n".join(f"段 {i}: " + "x" * 500 for i in range(1, 101))

    result = await adapter.send(
        "oc_1",
        long_text,
        metadata={"hermes_final_response": True, "message_id": "om_multicard"},
    )

    assert result.success
    assert len(captured_calls) >= 2
    uuid_values = [uuid_value for _, _, uuid_value in captured_calls]
    assert len(set(uuid_values)) == len(uuid_values)
    assert all(len(value) <= 50 for value in uuid_values)
    assert all("_card_index" not in metadata for _, metadata, _ in captured_calls)
