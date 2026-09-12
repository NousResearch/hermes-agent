"""Actual streaming writes must never expose a secret split by a delivery boundary."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import redact
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.fixture(autouse=True)
def dynamic_registry(monkeypatch):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    redact._reset_plugin_redaction_patterns()
    yield
    redact._reset_plugin_redaction_patterns()


def consumer():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 600
    adapter.supports_streaming = adapter.supports_native_streaming = False
    writes = []
    async def send(*args, **kwargs):
        writes.append(kwargs["content"])
        return SimpleNamespace(success=True, message_id=str(len(writes)))
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(side_effect=send)
    config = StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor="", transport="edit")
    return GatewayStreamConsumer(adapter, "test", config=config), writes


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["none", "segment", "flush"])
@pytest.mark.parametrize("parts,secret", [
    (("sk-abc", "123def456ghi789jkl012mno345pqr678stu "), "123def456ghi789jkl012mno345pqr678stu"),
    (("client_secret=hunter", "2\n"), "hunter2"),
    (("spring.datasource.password=hunter", "2\n"), "hunter2"),
    (("password: hunter", "2\n"), "hunter2"),
    (("acme_abcdefg", "hijklmnopqrstuvwxyz "), "hijklmnopqrstuvwxyz"),
    (("data=" + "x" * 480 + "&client_secret=", "supersecretvalue123456789"), "supersecretvalue123456789"),
])
async def test_secret_survives_delivery_boundaries(boundary, parts, secret):
    stream, writes = consumer()
    redact.register_redaction_patterns([r"acme_[A-Za-z0-9]{20,}"], source="test")
    task = asyncio.create_task(stream.run())
    stream.on_delta(parts[0])
    await asyncio.sleep(.08)
    if boundary == "segment":
        stream.on_segment_break()
        await asyncio.sleep(.08)
    elif boundary == "flush":
        assert await asyncio.to_thread(stream.flush_pending_sync, 1)
    assert not any("hunter" in value or "acme_abcdefg" in value or "sk-abc" in value for value in writes)
    stream.on_delta(parts[1])
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert writes
    assert all(secret not in value for value in writes)
    assert "***" in "".join(writes) or "..." in "".join(writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["segment", "flush"])
async def test_benign_held_prefix_survives_boundary(boundary):
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("sk-abc")
    await asyncio.sleep(.08)
    if boundary == "segment":
        stream.on_segment_break()
        await asyncio.sleep(.08)
    else:
        assert await asyncio.to_thread(stream.flush_pending_sync, 1)
    stream.on_delta(" is documentation.")
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert "sk-abc is documentation." in "".join(writes)


@pytest.mark.asyncio
async def test_held_plain_word_does_not_turn_inline_think_tag_into_block():
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("example")
    await asyncio.sleep(.08)
    stream.on_delta("<think>visible inline mention")
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert "example<think>visible inline mention" in "".join(writes)


@pytest.mark.asyncio
async def test_redacted_delivery_reconciles_with_raw_final_response():
    stream, writes = consumer()
    payload = "password: hunter2\n"
    task = asyncio.create_task(stream.run())
    stream.on_delta(payload)
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert "hunter2" not in "".join(writes)
    assert stream.delivered_final_matches(payload) is True


@pytest.mark.asyncio
async def test_new_segment_starts_a_think_block_without_resetting_secret_state():
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("Done. ")
    await asyncio.sleep(.08)
    stream.on_segment_break()
    await asyncio.sleep(.08)
    stream.on_delta("<think>hidden reasoning</think>Visible answer.")
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert "hidden reasoning" not in "".join(writes)
    assert "Visible answer." in "".join(writes)


@pytest.mark.asyncio
async def test_one_byte_consumer_preserves_raw_ledger_with_linear_candidate_scans(monkeypatch):
    from agent import streaming_redact
    original = streaming_redact._partial_context_opener_start
    scanned = []
    def counted(text, **kwargs):
        scanned.append(len(text))
        return original(text, **kwargs)
    monkeypatch.setattr(streaming_redact, "_partial_context_opener_start", counted)
    stream, writes = consumer()
    payload = "sk-" + "x" * 10000 + "!"
    for char in payload:
        stream.on_delta(char)
    stream.finish(payload)
    await asyncio.wait_for(stream.run(), 3)
    assert stream._stream_ledger == payload
    assert writes and all("x" * 20 not in value for value in writes)
    assert stream.delivered_final_matches(payload)
    assert sum(scanned) < 10 * len(payload)


@pytest.mark.asyncio
async def test_overflow_keeps_secret_context_and_authoritative_final_suffix():
    stream, writes = consumer()
    prefix = "A complete harmless sentence! " * 50 + "\npassword=hunter"
    task = asyncio.create_task(stream.run())
    stream.on_delta(prefix)
    await asyncio.sleep(.15)
    assert stream._turn_split_delivery
    assert len(writes) > 1
    stream.on_delta("2\n")
    final = prefix + "2\nCompletion footer!"
    stream.finish(final)
    await asyncio.wait_for(task, 3)
    assert all("hunter2" not in text for text in writes)
    assert any("Completion footer!" in text for text in writes)
    assert stream.delivered_final_matches(final)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["segment", "flush"])
async def test_authoritative_current_segment_final_keeps_prior_secret_prefix(boundary):
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("sk-abc")
    await asyncio.sleep(.08)
    if boundary == "segment":
        stream.on_segment_break()
        await asyncio.sleep(.08)
    else:
        assert await asyncio.to_thread(stream.flush_pending_sync, 1)
    suffix = "123def456ghi789jkl012mno345pqr678stu "
    stream.on_delta(suffix)
    stream.finish(suffix)
    await asyncio.wait_for(task, 2)
    assert writes and all(suffix.strip() not in text for text in writes)
    assert stream.delivered_final_matches(suffix)


@pytest.mark.asyncio
@pytest.mark.parametrize("channels", [("commentary", "commentary"), ("stream", "commentary"), ("commentary", "stream")])
async def test_commentary_and_answer_share_secret_recognition_without_final_ownership(channels):
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    parts = ["sk-abc", "123def456ghi789jkl012mno345pqr678stu "]
    for channel, part in zip(channels, parts):
        (stream.on_commentary if channel == "commentary" else stream.on_delta)(part)
        await asyncio.sleep(.08)
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert writes and all(parts[1].strip() not in text for text in writes)
    if channels[-1] == "commentary":
        assert not stream.final_response_sent


@pytest.mark.asyncio
async def test_benign_commentary_divergence_preserves_answer_candidate():
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta("sk-abc")
    await asyncio.sleep(.08)
    stream.on_commentary(" Using a tool!")
    await asyncio.sleep(.08)
    stream.on_delta(" is documentation.")
    stream.finish()
    await asyncio.wait_for(task, 2)
    assert any("Using a tool!" in text and "sk-abc" not in text for text in writes)
    assert any("sk-abc is documentation." in text for text in writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["  password: hunter2\n", "data=abc&signature=opaqueCredential123456"])
async def test_line_and_form_context_survive_actual_one_byte_deliveries(payload):
    stream, writes = consumer()
    for char in payload:
        stream.on_delta(char)
    stream.finish()
    await asyncio.wait_for(stream.run(), 2)
    assert writes and all("hunter2" not in text and "opaqueCredential123456" not in text for text in writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["ordinaryword", "sk-abc", "password=hunter2", '{"password":"unfinishedOpaqueCredential'])
async def test_cancelled_edit_resolves_held_tail_before_recording_delivery(tail):
    from agent.streaming_redact import sanitize_terminal_secret_text
    stream, writes = consumer()
    payload = "Visible partial " + tail
    task = asyncio.create_task(stream.run())
    stream.on_delta(payload)
    await asyncio.sleep(.08)
    assert stream._message_id is not None
    task.cancel()
    await task
    assert writes[-1] == sanitize_terminal_secret_text(payload)
    assert stream._secret_sanitizer.pending_length == 0
    assert stream.final_response_sent is True


@pytest.mark.asyncio
@pytest.mark.parametrize("first,second", [
    ('Data: {"tok', 'en": "opaqueJsonCredentialValue123"}'),
    ('URL: postgr', 'es://user:opaqueDbCredentialValue123@db.example'),
    ('Key:\n-----BEG', 'IN PRIVATE KEY-----\nSYNTHETICINERTPRIVATEKEYBODY1234567890\n-----END PRIVATE KEY-----'),
    ('OPENAI_API_', 'KEY=opaqueEnvCredentialValue123 done'),
    ('Authoriza', 'tion: Bearer opaqueAuthCredentialValue123 done'),
    ('x-api-', 'key: opaqueHeaderCredentialValue123 done'),
    ('mode=x&tok', 'en=opaqueFormCredentialValue123'),
])
async def test_shadow_commentary_uses_complete_event_mode(first, second):
    stream, writes = consumer()
    task = asyncio.create_task(stream.run())
    stream.on_delta(first)
    stream.on_commentary('Still working.')
    await asyncio.sleep(.08)
    stream.on_delta(second)
    stream.finish()
    await task
    assert writes.count('Still working.') == 1
    assert all('CredentialValue123' not in text for text in writes)
