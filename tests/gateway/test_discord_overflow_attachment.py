"""Full-response overflow delivery with real discord.File and fake network I/O."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

# gateway/conftest installs a lightweight SDK double before test collection.
# This file exercises real File ownership/closing, never a live Discord client.
for name in list(sys.modules):
    if (name == "discord" or name.startswith("discord.")) and isinstance(sys.modules[name], Mock):
        del sys.modules[name]
import discord
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


CONTENT = "# Héllo 🌍\n\n> quote\n- item\n```python\nprint('é')\n```\n" + "long line " * 6000


@pytest.fixture
def delivery(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    calls, uploads = [], []

    async def transport(**kwargs):
        calls.append(kwargs)
        files = kwargs.get("files", kwargs.get("attachments", []))
        for file in files:
            assert isinstance(file, discord.File)
            uploads.append((Path(file.fp.name), file.filename, file.fp.read()))
        return SimpleNamespace(id=901, attachments=[SimpleNamespace(id=1)] if files else [])

    msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=transport))
    channel = SimpleNamespace(id=555, guild=None, send=AsyncMock(side_effect=transport),
                              get_partial_message=lambda _: msg)
    adapter._client = SimpleNamespace(get_channel=lambda _: channel)
    adapter._resolve_channel = AsyncMock(return_value=channel)
    async def record(reply, result, content, final=False, metadata=None):
        return result
    adapter._record_response_async = AsyncMock(side_effect=record)
    return adapter, channel, msg, calls, uploads


@pytest.mark.asyncio
async def test_send_full_original_and_cleanup(delivery):
    adapter, channel, _, calls, uploads = delivery
    result = await adapter.send("555", CONTENT, reply_to="123", metadata={"notify": True})
    assert result.success and result.message_id == "901"
    assert len(calls) == len(uploads) == 1
    path, filename, data = uploads[0]
    assert data == CONTENT.encode("utf-8")
    assert filename.startswith("hermes-response-") and filename.endswith(".md")
    assert "555" not in filename and "123" not in filename
    assert not path.exists()
    assert calls[0]["reference"].message_id == 123
    assert "attached" in calls[0]["content"] and len(calls[0]["content"]) < 300
    assert adapter._last_self_message_id["555"] == "901"
    adapter._record_response_async.assert_awaited_once_with("123", result, CONTENT, True, {"notify": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [3, 8])
async def test_under_and_exact_cap_unchanged(delivery, monkeypatch, count):
    adapter, _, _, calls, uploads = delivery
    chunks = [f"chunk {i}" for i in range(count)]
    monkeypatch.setattr(adapter, "truncate_message", lambda *a: chunks)
    result = await adapter.send("555", "original")
    assert result.success
    assert [c["content"] for c in calls] == chunks
    assert not uploads


@pytest.mark.asyncio
async def test_thread_metadata_and_nonconversational_tracking(delivery):
    adapter, _, _, _, uploads = delivery
    adapter._nonconversational_messages.mark_many = AsyncMock()
    metadata = {"thread_id": "777", "non_conversational": True}
    result = await adapter.send("555", CONTENT, metadata=metadata)
    assert result.success and len(uploads) == 1
    assert all(call.args[0] == "777" for call in adapter._resolve_channel.await_args_list)
    adapter._nonconversational_messages.mark_many.assert_awaited_once_with(["901"])
    assert not adapter._last_self_message_id


@pytest.mark.asyncio
async def test_final_edit_attaches_in_place_and_records(delivery):
    adapter, channel, msg, calls, uploads = delivery
    metadata = {"thread_id": "777", "reply_to_message_id": "123"}
    result = await adapter.edit_message("555", "42", CONTENT, finalize=True, metadata=metadata)
    assert result.success and result.message_id == "42"
    assert not result.continuation_message_ids
    channel.send.assert_not_awaited()
    assert len(uploads) == 1 and uploads[0][2] == CONTENT.encode("utf-8")
    assert not uploads[0][0].exists()
    adapter._record_response_async.assert_awaited_once_with("123", result, CONTENT, True, metadata)
    adapter._resolve_channel.assert_awaited_with("777")


@pytest.mark.asyncio
async def test_forum_single_starter_attachment(delivery):
    adapter, channel, _, calls, uploads = delivery
    async def create(**kwargs):
        starter = await channel.send(**kwargs)
        return SimpleNamespace(thread=SimpleNamespace(id=777), message=starter)
    forum = SimpleNamespace(id=666, type=SimpleNamespace(value=15),
                            create_thread=AsyncMock(side_effect=create))
    adapter._resolve_channel.return_value = forum
    result = await adapter.send("666", CONTENT)
    assert result.success and result.raw_response["thread_id"] == "777"
    assert len(calls) == len(uploads) == 1
    assert uploads[0][2] == CONTENT.encode("utf-8") and not uploads[0][0].exists()
    forum.create_thread.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["first", "all", "off"])
async def test_reply_modes_and_rejected_anchor_reopens_file(delivery, mode):
    adapter, channel, _, calls, uploads = delivery
    adapter._reply_to_mode = mode
    original = channel.send.side_effect
    async def send(**kwargs):
        if kwargs.get("reference"):
            file = kwargs["files"][0]
            file.fp.read()
            file.close()  # discord.py closes files even on rejected HTTP sends
            calls.append(kwargs)
            raise RuntimeError("error code: 10008 Unknown Message")
        return await original(**kwargs)
    channel.send.side_effect = send
    result = await adapter.send("555", CONTENT, reply_to="123")
    assert result.success
    assert len(calls) == (1 if mode == "off" else 2)
    assert uploads[0][2] == CONTENT.encode("utf-8")
    assert not uploads[0][0].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "rejected", "transport", "cancel"])
async def test_upload_failure_is_bounded_honest_and_cleans_up(delivery, failure):
    adapter, channel, _, calls, uploads = delivery
    original = channel.send.side_effect
    async def send(**kwargs):
        result = await original(**kwargs)
        if kwargs.get("files"):
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "transport":
                raise ConnectionError("connection closed")
            if failure == "rejected":
                raise RuntimeError("upload denied")
            result.attachments = []
        return result
    channel.send.side_effect = send
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await adapter.send("555", CONTENT)
    else:
        result = await adapter._send_with_retry("555", CONTENT, base_delay=0)
        assert not result.success
        assert result.retryable is (failure == "transport")
        assert len(calls) <= 2  # upload plus one bounded failure notice; no retries/flood
        if failure == "transport":
            assert result.error == "send_path_degraded"
        else:
            assert "could not" in calls[-1]["content"].lower()
    assert len(uploads) == 1
    assert not uploads[0][0].exists()
    assert uploads[0][2] == CONTENT.encode("utf-8")


@pytest.mark.asyncio
async def test_oversized_upload_no_false_success(delivery, monkeypatch):
    adapter, _, _, calls, uploads = delivery
    monkeypatch.setattr(adapter, "_discord_upload_limit_bytes", lambda _: 1)
    result = await adapter._send_with_retry("555", CONTENT)
    assert not result.success
    assert not uploads and len(calls) == 1
    assert "session" in calls[0]["content"].lower()


@pytest.mark.asyncio
async def test_concurrent_upload_names_unique(delivery):
    adapter, _, _, _, uploads = delivery
    results = await asyncio.gather(adapter.send("555", CONTENT), adapter.send("555", CONTENT))
    assert all(r.success for r in results)
    assert len({u[1] for u in uploads}) == 2
    assert all(not p.exists() for p, _, _ in uploads)


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_real_stream_consumer_preserves_overflow_and_failure(delivery, preview, failure):
    from gateway.stream_consumer import GatewayStreamConsumer
    adapter, channel, msg, calls, uploads = delivery
    if failure:
        original = msg.edit.side_effect if preview else channel.send.side_effect
        async def rejected(**kwargs):
            result = await original(**kwargs)
            if kwargs.get("files") or kwargs.get("attachments"):
                raise ConnectionError("Connection reset by peer")
            return result
        if preview:
            msg.edit.side_effect = rejected
        else:
            channel.send.side_effect = rejected
    consumer = GatewayStreamConsumer(adapter, "555", initial_reply_to_id="123")
    if preview:
        consumer._message_id = "42"
        consumer._last_sent_text = "preview"
        consumer._already_sent = True
    consumer.on_delta(CONTENT)
    consumer.finish(final_text=CONTENT)
    await consumer.run()
    assert len(uploads) == 1
    assert uploads[0][2] == CONTENT.encode("utf-8")
    assert len(calls) == 1
    assert not uploads[0][0].exists()
    assert consumer.final_response_sent is not failure
    assert consumer.final_content_delivered is not failure
    if failure:
        result = consumer.deferred_final_delivery(CONTENT)
        assert result is not None and not result.success and result.retryable
        assert consumer.deferred_final_delivery("different final") is None
    else:
        assert consumer.delivered_final_matches(CONTENT) is True
        if preview:
            assert adapter._record_response_async.call_args.args[0] == "123"


@pytest.mark.asyncio
async def test_failure_notice_is_nonconversational(delivery, monkeypatch):
    adapter, _, _, calls, _ = delivery
    monkeypatch.setattr(adapter, "_discord_upload_limit_bytes", lambda _: 1)
    adapter._nonconversational_messages.mark_many = AsyncMock()
    result = await adapter.send("555", CONTENT)
    assert not result.success and len(calls) == 1
    adapter._nonconversational_messages.mark_many.assert_awaited_once_with(["901"])


@pytest.mark.asyncio
async def test_interim_send_is_preview_not_attachment(delivery):
    adapter, channel, _, calls, uploads = delivery
    result = await adapter.send("555", CONTENT, metadata={"expect_edits": True})
    assert result.success and len(calls) == 1 and not uploads
    assert len(calls[0]["content"]) <= adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_unchanged_large_preview_still_finalizes_to_original_file(delivery):
    from gateway.stream_consumer import GatewayStreamConsumer
    adapter, _, _, calls, uploads = delivery
    consumer = GatewayStreamConsumer(adapter, "555")
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(CONTENT)
    while not calls:
        await asyncio.sleep(0)
    consumer.finish(final_text=CONTENT)
    await task
    assert len(uploads) == 1 and uploads[0][2] == CONTENT.encode("utf-8")
    assert consumer.final_content_delivered


@pytest.mark.asyncio
async def test_streaming_open_fence_is_not_added_to_attachment(delivery):
    from gateway.stream_consumer import GatewayStreamConsumer
    adapter, _, _, _, uploads = delivery
    content = "```python\n" + "print('é')\n" * 6000
    consumer = GatewayStreamConsumer(adapter, "555")
    consumer.on_delta(content)
    consumer.finish(final_text=content)
    await consumer.run()
    assert uploads[0][2] == content.encode("utf-8")


@pytest.mark.asyncio
async def test_failed_interim_edit_cannot_bypass_final_cap(delivery):
    from gateway.stream_consumer import GatewayStreamConsumer
    adapter, _, msg, calls, uploads = delivery
    consumer = GatewayStreamConsumer(adapter, "555")
    consumer._message_id = "42"
    consumer._last_sent_text = "preview"
    consumer._already_sent = True
    consumer._fallback_final_send = True
    consumer._fallback_prefix = "preview"
    consumer._edit_supported = False
    consumer.on_delta(CONTENT)
    consumer.finish(final_text=CONTENT)
    await consumer.run()
    assert len(uploads) == 1 and uploads[0][2] == CONTENT.encode("utf-8")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_forum_transport_error_retryable_no_notice_thread(delivery):
    adapter, _, _, _, _ = delivery
    forum = SimpleNamespace(id=666, type=SimpleNamespace(value=15),
                            create_thread=AsyncMock(side_effect=ConnectionError("Connection reset by peer")))
    adapter._resolve_channel.return_value = forum
    result = await adapter.send("666", CONTENT)
    assert not result.success and result.retryable
    forum.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_forbidden_error_retains_permanent_classification(delivery):
    from gateway.platforms.base import classify_send_error
    adapter, channel, _, _, _ = delivery
    channel.send.side_effect = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions")
    result = await adapter.send("555", CONTENT)
    assert not result.success and not result.retryable
    assert classify_send_error(None, result.error) == "forbidden"


@pytest.mark.asyncio
async def test_short_correction_removes_owned_overflow_attachment(delivery):
    adapter, _, msg, calls, uploads = delivery
    assert (await adapter.edit_message("555", "42", CONTENT, finalize=True)).success
    calls.clear()
    assert (await adapter.edit_message("555", "42", "corrected", finalize=True)).success
    assert calls[0]["attachments"] == []


@pytest.mark.asyncio
async def test_partial_small_final_is_not_recorded_as_complete(delivery):
    from gateway.stream_consumer import GatewayStreamConsumer
    adapter, channel, _, _, _ = delivery
    channel.send.side_effect = RuntimeError("upload rejected")
    consumer = GatewayStreamConsumer(adapter, "555")
    consumer._message_id = "42"
    consumer._last_sent_text = "preview"
    consumer._already_sent = True
    content = "small final " * 400
    consumer.on_delta(content)
    consumer.finish(final_text=content)
    await consumer.run()
    assert not consumer.final_content_delivered
    assert not consumer.final_response_sent
    assert not any(call.args[1].success for call in adapter._record_response_async.await_args_list)
