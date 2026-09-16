"""Matrix provider-enforced idempotency for durable claimed-result parts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.claimed_result_publication import ClaimedResultPartDeliveryError
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


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


@pytest.mark.asyncio
async def test_claimed_missing_attachment_warning_does_not_complete_part(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "isolated-hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    missing = tmp_path / "missing.pdf"
    room_id = "!room:example.org"
    session_key = "agent:main:matrix:room:example"
    oid = dl.record_claimed_result(
        session_key=session_key,
        claim_id="claim-missing-matrix",
        claim_event_id="event-missing-matrix",
        platform="matrix",
        chat_id=room_id,
        thread_id=None,
        content="",
        raw_content=f"MEDIA:{missing}",
        adapter_profile=None,
    )
    dl.prepare_claimed_result_delivery(
        oid,
        session_key=session_key,
        platform="matrix",
        chat_id=room_id,
        thread_id=None,
        content="",
        adapter_profile=None,
    )
    adapter = _adapter()
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="$warning")
    )

    with pytest.raises(
        ClaimedResultPartDeliveryError,
        match="claimed continuation attachment delivery failed",
    ):
        await adapter._deliver_claimed_response_parts(
            obligation_id=oid,
            chat_id=room_id,
            text_content="",
            images=[],
            media_files=[],
            local_files=[str(missing)],
            force_document_attachments=True,
            metadata={},
            reply_to=None,
        )

    adapter.send.assert_awaited_once()
    assert str(missing) not in repr(adapter.send.await_args)
    assert dl.get_claimed_result_parts(oid)[0]["state"] == "failed"
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id=?",
            (oid,),
        ).fetchone()[0] == "failed"
