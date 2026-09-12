"""WhatsApp adapter diagnostics never replace live payloads or error results."""
import logging
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


@pytest.mark.asyncio
async def test_whatsapp_cloud_media_exception_log_omits_raw_url_and_traceback(
    caplog,
):
    media_id = "wamid.SECRET-987654321012345"
    private_name = "Private Contact"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._http_client = MagicMock()
    adapter._http_client.get = AsyncMock(
        side_effect=RuntimeError(
            f"GET https://graph.facebook.com/v20.0/{media_id} failed "
            f"for {private_name}"
        )
    )
    adapter._access_token = "test-token"
    adapter._api_version = "v20.0"

    with caplog.at_level(
        logging.WARNING, logger="gateway.platforms.whatsapp_cloud"
    ):
        result = await adapter._download_media_to_cache(media_id)

    assert result == (None, None)
    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "media metadata fetch raised" in record.message
    ]
    assert len(records) == 1
    record = records[0]
    assert media_id not in record.message
    assert private_name not in record.message
    assert "https://graph.facebook.com" not in record.message
    assert "RuntimeError" in record.message
    assert record.exc_info is None

@pytest.mark.asyncio
async def test_whatsapp_cloud_graph_error_log_omits_raw_body(caplog):
    """Graph error bodies can echo WhatsApp text and must stay out of logs."""
    private_marker = "15551234567-private-graph-body-marker"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        return_value=MagicMock(
            status_code=400,
            json=MagicMock(
                return_value={
                    "error": {
                        "code": 131026,
                        "message": f"invalid recipient {private_marker}",
                    }
                }
            ),
        )
    )
    adapter._phone_number_id = "1234567890"
    adapter._api_version = "v20.0"
    adapter._access_token = "test-token"
    adapter._reply_prefix = None

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.whatsapp_cloud"):
        result = await adapter.send("15551234567", "hello")

    assert not result.success
    assert private_marker in result.error
    assert adapter._http_client.post.call_args.kwargs["json"]["to"] == "15551234567"
    assert adapter._http_client.post.call_args.kwargs["json"]["text"]["body"] == "hello"
    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "send rejected" in record.message
    ]
    assert len(records) == 1
    assert private_marker not in records[0].message
    assert "GraphAPIError" in records[0].message

@pytest.mark.asyncio
async def test_whatsapp_cloud_wamid_dispatch_exception_log_is_type_only(caplog):
    wamid = "wamid.SECRET-987654321012345"
    private_name = "Private Contact"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._seen_wamids = OrderedDict()
    adapter._duplicate_count = 0
    adapter._accepted_count = 0
    adapter._build_message_event_from_cloud = AsyncMock(
        side_effect=RuntimeError(f"build failed for {wamid} {private_name}")
    )
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "field": "messages",
                "value": {
                    "messages": [{"id": wamid}],
                    "contacts": [],
                    "metadata": {},
                },
            }],
        }],
    }

    with caplog.at_level(
        logging.WARNING, logger="gateway.platforms.whatsapp_cloud"
    ):
        await adapter._dispatch_payload(payload)

    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "failed to build event for wamid" in record.message
    ]
    assert len(records) == 1
    record = records[0]
    assert wamid not in record.message
    assert private_name not in record.message
    assert "RuntimeError" in record.message
    assert record.exc_info is None
