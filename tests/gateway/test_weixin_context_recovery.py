"""Weixin context generations and partial text delivery contracts."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.weixin import adapter as weixin


@pytest_asyncio.fixture
async def adapter():
    instance = weixin.WeixinAdapter(PlatformConfig(
        enabled=True, token="test-token", extra={
            "account_id": "test-account", "dm_policy": "allowlist", "allow_from": ["peer"],
            "send_chunk_delay_seconds": 0, "send_chunk_retries": 0,
            "send_chunk_retry_delay_seconds": 0, "text_batch_delay_seconds": 0,
        }))
    instance._send_session = instance._poll_session = object()
    instance._typing_cache.set("peer", "test-ticket")
    await instance._token_store.set(instance._account_id, "peer", "old-context")
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("refreshed", ["old-context", "new-context"])
async def test_invalidation_preserves_newer_generation_even_with_same_token(tmp_path, refreshed):
    store = weixin.ContextTokenStore(str(tmp_path))
    await store.set("account", "peer", "old-context")
    snapshot = store.snapshot("account", "peer")
    await store.set("account", "peer", refreshed)
    await store.invalidate("account", "peer", snapshot)
    restored = weixin.ContextTokenStore(str(tmp_path))
    restored.restore("account")
    assert restored.get("account", "peer") == refreshed

    await store.set("other-account", "peer", "other-context")
    await store.invalidate("account", "peer", store.snapshot("account", "peer"))
    restored = weixin.ContextTokenStore(str(tmp_path))
    restored.restore("account")
    restored.restore("other-account")
    assert restored.get("account", "peer") is None
    assert restored.get("other-account", "peer") == "other-context"


@pytest.mark.asyncio
async def test_invalidation_flush_cannot_overwrite_concurrent_inbound(tmp_path, monkeypatch):
    store = weixin.ContextTokenStore(str(tmp_path))
    await store.set("account", "peer", "old-context")
    snapshot = store.snapshot("account", "peer")
    entered, release = threading.Event(), threading.Event()
    real_write = weixin.atomic_json_write

    def blocked_write(path, payload):
        if not payload:
            entered.set()
            assert release.wait(timeout=10)
        real_write(path, payload)

    monkeypatch.setattr(weixin, "atomic_json_write", blocked_write)
    invalidation = asyncio.create_task(store.invalidate("account", "peer", snapshot))
    refresh = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        refresh = asyncio.create_task(store.set("account", "peer", "new-context"))
        # Waiting for the persist lock proves that set() has updated the in-memory value.
        await asyncio.sleep(0)
        assert store.get("account", "peer") == "new-context"
    finally:
        release.set()
        await asyncio.gather(invalidation, *([refresh] if refresh else []))
    restored = weixin.ContextTokenStore(str(tmp_path))
    restored.restore("account")
    assert restored.get("account", "peer") == "new-context"


def inbound(message_id, context="new-context", **fields):
    return {"from_user_id": "peer", "message_id": message_id, "message_type": 1,
            "context_token": context, "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
            **fields}


@pytest.mark.asyncio
async def test_new_message_id_refreshes_context_before_content_dedup(adapter):
    received = []
    adapter._enqueue_text_event = received.append
    await adapter._process_message(inbound("first", "first-context"))
    await adapter._process_message(inbound("second"))
    assert adapter._token_store.get(adapter._account_id, "peer") == "new-context"
    assert len(received) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [
    {"message_type": 2}, {"message_type": 0}, {"from_user_id": "unlisted"},
    {"from_user_id": "test-account"}, {"room_id": "disabled-room"},
])
async def test_rejected_updates_do_not_refresh_context_or_consume_content(adapter, rejected):
    received = []
    adapter._enqueue_text_event = received.append
    await adapter._process_message(inbound("rejected", **rejected))
    assert adapter._token_store.get(adapter._account_id, "peer") == "old-context"
    assert adapter._token_store.get(adapter._account_id, "unlisted") is None
    await adapter._process_message(inbound("accepted"))
    assert [event.text for event in received] == ["hello"]


@pytest.mark.asyncio
async def test_duplicate_message_id_cannot_refresh_same_token_generation(adapter):
    adapter._enqueue_text_event = lambda event: None
    await adapter._process_message(inbound("message"))
    snapshot = adapter._token_store.snapshot(adapter._account_id, "peer")
    await adapter._process_message(inbound("message"))
    assert adapter._token_store.snapshot(adapter._account_id, "peer") == snapshot


@pytest.mark.asyncio
async def test_referenced_media_wrapper_is_not_a_control_update(adapter, monkeypatch, tmp_path):
    received = []
    adapter.handle_message = AsyncMock(side_effect=received.append)
    monkeypatch.setattr(weixin, "_download_and_decrypt_media", AsyncMock(return_value=b"test-image"))
    monkeypatch.setattr(weixin, "cache_image_from_bytes_async", AsyncMock(return_value=str(tmp_path / "image.jpg")))
    reference = {"ref_msg": {"message_item": {"type": weixin.ITEM_IMAGE, "image_item": {"media": {}}}}}
    await adapter._process_message(inbound("reference", item_list=[reference]))
    assert len(received) == 1
    assert received[0].media_types == ["image/jpeg"]
    assert adapter._token_store.get(adapter._account_id, "peer") == "new-context"


@pytest.mark.asyncio
@pytest.mark.parametrize("refreshed", ["old-context", "new-context"])
async def test_stale_text_response_preserves_inflight_inbound_generation(adapter, monkeypatch, refreshed):
    calls = []
    adapter._enqueue_text_event = lambda event: None

    async def send_message(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await adapter._process_message(inbound("fresh", refreshed))
            return {"ret": -14, "errmsg": "session expired"}
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_message", send_message)
    result = await adapter.send("peer", "reply")
    assert result.success
    assert [call["context_token"] for call in calls] == ["old-context", refreshed]
    assert calls[0]["client_id"] == calls[1]["client_id"]
    restored = weixin.ContextTokenStore(adapter._hermes_home)
    restored.restore(adapter._account_id)
    assert restored.get(adapter._account_id, "peer") == refreshed


@pytest.mark.asyncio
async def test_queued_send_reads_context_after_gate_opens(adapter, monkeypatch):
    send_message = AsyncMock(return_value={"ret": 0})
    monkeypatch.setattr(weixin, "_send_message", send_message)
    await adapter._send_text_gate.acquire()
    task = asyncio.create_task(adapter.send("peer", "reply"))
    try:
        await asyncio.sleep(0)
        await adapter._token_store.set(adapter._account_id, "peer", "new-context")
    finally:
        adapter._send_text_gate.release()
    assert (await task).success
    assert send_message.await_args.kwargs["context_token"] == "new-context"


@pytest.mark.asyncio
async def test_chunks_and_transport_retries_read_latest_context(adapter, monkeypatch):
    adapter.MAX_MESSAGE_LENGTH = 12
    adapter._send_chunk_retries = 1
    calls = []

    async def send_message(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await adapter._token_store.set(adapter._account_id, "peer", "retry-context")
            raise ConnectionError("connection reset")
        if len(calls) == 2:
            await adapter._token_store.set(adapter._account_id, "peer", "next-context")
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_message", send_message)
    assert (await adapter.send("peer", "first\n\nsecond\n\nthird")).success
    assert [call["context_token"] for call in calls] == [
        "old-context", "retry-context", "next-context", "next-context"]
    assert calls[0]["client_id"] == calls[1]["client_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [{"ret": -14}, {"ret": -2, "errmsg": "prepare failed"}])
async def test_stale_recovery_is_bounded_and_invalidated_token_stays_off_disk(adapter, monkeypatch, error):
    adapter._send_chunk_retries = 3
    send_message = AsyncMock(return_value=error)
    monkeypatch.setattr(weixin, "_send_message", send_message)
    result = await adapter.send("peer", "reply")
    assert not result.success
    assert "session not ready" in result.error
    assert [call.kwargs["context_token"] for call in send_message.await_args_list] == ["old-context", None]
    assert adapter._rate_limit_cooldown_remaining() == 0
    restored = weixin.ContextTokenStore(adapter._hermes_home)
    restored.restore(adapter._account_id)
    assert restored.get(adapter._account_id, "peer") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["send", "base_retry", "router"])
async def test_partial_text_receipts_block_whole_payload_fallback(adapter, monkeypatch, entrypoint):
    from gateway.config import GatewayConfig, Platform
    from gateway.delivery import DeliveryRouter, DeliveryTarget, PartialDeliveryError

    adapter.MAX_MESSAGE_LENGTH = 12
    send_message = AsyncMock(side_effect=[{"ret": 0}, {"ret": 0}, RuntimeError("refused")])
    monkeypatch.setattr(weixin, "_send_message", send_message)
    content = "first\n\nsecond\n\nthird"
    if entrypoint == "send":
        result = await adapter.send("peer", content)
    elif entrypoint == "base_retry":
        result = await adapter._send_with_retry("peer", content)
    else:
        router = DeliveryRouter(GatewayConfig(), adapters={Platform.WEIXIN: adapter})
        with pytest.raises(PartialDeliveryError) as caught:
            await router._deliver_to_platform(DeliveryTarget(Platform.WEIXIN, "peer"), content, None)
        result = caught.value.result
    assert not result.success
    assert send_message.await_count == 3
    ids = tuple(call.kwargs["client_id"] for call in send_message.await_args_list[:2])
    assert result.message_id == ids[-1]
    assert result.continuation_message_ids == ids[:-1]
    assert result.raw_response == {
        "partial_overflow": True, "delivered_chunks": 2, "total_chunks": 3,
        "delivered_message_ids": ids, "last_message_id": ids[-1],
    }
    assert await adapter._resume_partial_send("peer", result, reply_to=None, metadata=None) is None


@pytest.mark.asyncio
async def test_successful_split_returns_confirmed_ids_in_send_order(adapter, monkeypatch):
    adapter.MAX_MESSAGE_LENGTH = 12
    send_message = AsyncMock(return_value={"ret": 0})
    monkeypatch.setattr(weixin, "_send_message", send_message)
    result = await adapter.send("peer", "first\n\nsecond\n\nthird")
    ids = tuple(call.kwargs["client_id"] for call in send_message.await_args_list)
    assert result.success
    assert result.message_id == ids[-1]
    assert result.continuation_message_ids == ids[:-1]
    assert result.raw_response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_chunks", [3, 0, 1, 2], ids=[
    "all-success", "first-fails", "only-first-success", "multiple-success-then-failure",
])
async def test_split_receipts_cover_confirmed_ids_in_real_stream_consumer(adapter, monkeypatch, accepted_chunks):
    from gateway.stream_consumer import GatewayStreamConsumer

    adapter.MAX_MESSAGE_LENGTH = 12
    responses = [{"ret": 0}] * accepted_chunks
    if accepted_chunks < 3:
        responses.append(RuntimeError("refused"))
    send_message = AsyncMock(side_effect=responses)
    monkeypatch.setattr(weixin, "_send_message", send_message)
    result = await adapter.send("peer", "first\n\nsecond\n\nthird")
    attempted_ids = tuple(call.kwargs["client_id"] for call in send_message.await_args_list)
    confirmed_ids = attempted_ids[:accepted_chunks]
    unconfirmed_ids = attempted_ids[accepted_chunks:]

    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="peer")
    consumer._track_preview_ids_from_result(result)
    assert consumer._preview_message_ids == set(confirmed_ids)
    assert consumer._segment_preview_message_ids == set(confirmed_ids)
    assert not consumer._preview_message_ids.intersection(unconfirmed_ids)
    assert result.success == (accepted_chunks == 3)
    assert result.message_id == (confirmed_ids[-1] if confirmed_ids else None)
    assert result.message_id not in result.continuation_message_ids
    assert result.continuation_message_ids == confirmed_ids[:-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("refreshed", [None, "old-context", "new-context"])
async def test_media_stale_response_invalidates_only_its_snapshot(adapter, monkeypatch, tmp_path, refreshed):
    document = tmp_path / "document.txt"
    document.write_bytes(b"test-document")
    monkeypatch.setattr(weixin, "_get_upload_url", AsyncMock(return_value={"upload_full_url": "https://example.invalid/upload"}))
    monkeypatch.setattr(weixin, "_upload_ciphertext", AsyncMock(return_value="test-media-reference"))
    calls = []

    async def send_items(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if refreshed:
                await adapter._token_store.set(adapter._account_id, "peer", refreshed)
            return {"ret": -14}
        if len(calls) == 2:
            await adapter._token_store.set(adapter._account_id, "peer", "caption-context")
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_items", send_items)
    result = await adapter.send_document("peer", str(document), caption="caption")
    assert result.success
    assert [call["context_token"] for call in calls] == ["old-context", refreshed, "caption-context"]
    assert calls[0]["client_id"] == calls[1]["client_id"]


@pytest.mark.asyncio
async def test_media_recovery_stops_after_one_new_generation_retry(adapter, monkeypatch, tmp_path):
    document = tmp_path / "document.txt"
    document.write_bytes(b"test-document")
    monkeypatch.setattr(weixin, "_get_upload_url", AsyncMock(return_value={"upload_full_url": "https://example.invalid/upload"}))
    monkeypatch.setattr(weixin, "_upload_ciphertext", AsyncMock(return_value="test-media-reference"))
    calls = []

    async def send_items(*args, **kwargs):
        calls.append(kwargs)
        await adapter._token_store.set(adapter._account_id, "peer", "new-context")
        return {"ret": -14}

    monkeypatch.setattr(weixin, "_send_items", send_items)
    result = await adapter.send_document("peer", str(document))
    assert not result.success
    assert "session not ready" in result.error
    assert len(calls) == 2
    assert adapter._token_store.get(adapter._account_id, "peer") == "new-context"
