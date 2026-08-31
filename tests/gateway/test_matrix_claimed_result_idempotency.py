"""Matrix provider-enforced idempotency for durable claimed-result parts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )
    adapter._text_batch_delay_seconds = 0
    return adapter


@pytest.mark.asyncio
async def test_claimed_text_retries_reuse_matrix_transaction_ids_per_chunk():
    adapter = _adapter()
    adapter.max_message_length = 5
    send_event = AsyncMock(return_value="$event")
    adapter._client = SimpleNamespace(send_message_event=send_event, crypto=None)
    metadata = {"_hermes_delivery_part_id": "0123456789abcdef01234567"}

    first = await adapter.send("!room:example.org", "abcdefgh", metadata=metadata)
    first_call_count = send_event.await_count
    second = await adapter.send("!room:example.org", "abcdefgh", metadata=metadata)

    assert first.success and second.success
    first_ids = [
        call.kwargs["txn_id"]
        for call in send_event.await_args_list[:first_call_count]
    ]
    retry_ids = [
        call.kwargs["txn_id"]
        for call in send_event.await_args_list[first_call_count:]
    ]
    assert first_ids == retry_ids
    assert len(first_ids) > 1
    assert first_ids[0] != first_ids[1]


@pytest.mark.asyncio
async def test_claimed_attachment_retries_reuse_matrix_transaction_id():
    adapter = _adapter()
    send_event = AsyncMock(side_effect=["$media", "$media"])
    adapter._client = SimpleNamespace(
        send_message_event=send_event,
        upload_media=AsyncMock(return_value="mxc://example.org/media"),
        crypto=None,
    )
    metadata = {"_hermes_delivery_part_id": "fedcba9876543210fedcba98"}

    first = await adapter._upload_and_send(
        "!room:example.org",
        b"payload",
        "photo.png",
        "image/png",
        "m.image",
        None,
        None,
        metadata,
    )
    second = await adapter._upload_and_send(
        "!room:example.org",
        b"payload",
        "photo.png",
        "image/png",
        "m.image",
        None,
        None,
        metadata,
    )

    assert first.success and second.success
    first_id = send_event.await_args_list[0].kwargs["txn_id"]
    retry_id = send_event.await_args_list[1].kwargs["txn_id"]
    assert first_id == retry_id
