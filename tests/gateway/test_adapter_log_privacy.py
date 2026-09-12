"""Shared adapter diagnostics preserve raw authorization, delivery, and retry inputs."""

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(params=[Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
def adapter(request):
    value = WhatsAppAdapter.__new__(WhatsAppAdapter)
    value.platform = request.param
    return value


def assert_scoped_log(caplog, adapter, private):
    if adapter.platform == Platform.TELEGRAM:
        assert private in caplog.text
    else:
        assert private not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)


def test_authorization_failure_preserves_callback_inputs(caplog, adapter):
    identity = "15551234567"
    private = "private authorization response"
    adapter._authorization_check = Mock(side_effect=RuntimeError(private))
    with caplog.at_level(logging.DEBUG, logger="gateway.platforms.base"):
        assert adapter._is_sender_authorized(identity, "dm", identity) is None
    adapter._authorization_check.assert_called_once_with(identity, "dm", identity)
    assert_scoped_log(caplog, adapter, private)
    if adapter.platform != Platform.TELEGRAM:
        assert identity not in caplog.text
        assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_format_fallback_keeps_raw_error_and_send_inputs(caplog, adapter):
    private = "private provider response"
    error = SendResult(success=False, error=private)
    adapter.send = AsyncMock(return_value=error)
    with caplog.at_level(logging.DEBUG, logger="gateway.platforms.base"):
        result = await adapter._send_with_retry("15551234567", "private reply", metadata={"key": "raw"})
    assert result is error
    assert result.error == private
    assert adapter.send.await_count == 2
    first, second = adapter.send.await_args_list
    assert first.kwargs["chat_id"] == second.kwargs["chat_id"] == "15551234567"
    assert first.kwargs["content"] == "private reply"
    assert second.kwargs["content"].endswith("private reply")
    assert first.kwargs["metadata"] == second.kwargs["metadata"] == {"key": "raw"}
    assert_scoped_log(caplog, adapter, private)


@pytest.mark.asyncio
async def test_image_diagnostics_keep_raw_caption_and_url(caplog, adapter):
    private = "private-caption-marker"
    url = "https://example.com/private-filename.png"
    adapter.send_image = AsyncMock(return_value=SendResult(success=True))
    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter.send_multiple_images("15551234567", [(url, private)])
    adapter.send_image.assert_awaited_once_with(chat_id="15551234567", image_url=url, caption=private, metadata=None)
    assert_scoped_log(caplog, adapter, private)
    if adapter.platform != Platform.TELEGRAM:
        assert "private-filename" not in caplog.text


def test_runtime_status_failure_keeps_status_payload(monkeypatch, caplog, adapter):
    private = "private status detail"
    writer = Mock(side_effect=OSError(private))
    monkeypatch.setattr("gateway.status.write_runtime_status", writer)
    with caplog.at_level(logging.DEBUG, logger="gateway.platforms.base"):
        adapter._write_runtime_status_safe("fatal", error_message=private)
    writer.assert_called_once_with(platform=adapter.platform.value, error_message=private)
    assert_scoped_log(caplog, adapter, private)


@pytest.mark.asyncio
async def test_chained_delivery_callbacks_keep_execution_and_scoped_errors(caplog, adapter):
    private = "private callback detail"
    adapter._post_delivery_callbacks = {}
    first = Mock(side_effect=RuntimeError(private))
    second = AsyncMock()
    key = "agent:main:whatsapp:dm:15551234567"
    adapter.register_post_delivery_callback(key, first)
    adapter.register_post_delivery_callback(key, second)
    callback = adapter.pop_post_delivery_callback(key)
    with caplog.at_level(logging.DEBUG, logger="gateway.platforms.base"):
        await callback()
    first.assert_called_once_with()
    second.assert_awaited_once_with()
    assert adapter._post_delivery_callbacks == {}
    assert_scoped_log(caplog, adapter, private)
