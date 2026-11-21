"""Public Slack commentary gets one transport-owned prefix, never the final."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.slack.adapter_message_style import style_public_message


@pytest.fixture
def adapter():
    slack = SlackAdapter(PlatformConfig(
        enabled=True, extra={"interim_speech_balloon": True}))
    client = SimpleNamespace(
        chat_postMessage=AsyncMock(
            side_effect=[{"ok": True, "ts": "1.1"}, {"ok": True, "ts": "1.2"}]),
        chat_startStream=AsyncMock(return_value={"ok": True, "ts": "2.1"}),
        chat_appendStream=AsyncMock(return_value={"ok": True}),
        chat_stopStream=AsyncMock(return_value={"ok": True}),
        chat_update=AsyncMock(return_value={"ok": True, "ts": "2.1"}),
    )
    slack._app = SimpleNamespace(client=client)
    slack._get_client = Mock(return_value=client)
    slack.stop_typing = AsyncMock()
    return slack, client


@pytest.mark.asyncio
async def test_public_commentary_has_automatic_prefix_but_final_does_not(adapter):
    slack, client = adapter
    consumer = GatewayStreamConsumer(
        slack, "CTEST", StreamConsumerConfig(buffer_threshold=1000),
        metadata={"thread_id": "1.0"})
    consumer.on_commentary("I will check the service.")
    consumer.on_delta("The service is healthy.")
    consumer.finish()

    await consumer.run()

    assert [c.kwargs["text"] for c in client.chat_postMessage.await_args_list] == [
        "💬 I will check the service.", "The service is healthy."]
    assert consumer.has_delivered_text("I will check the service.")
    assert consumer.delivered_final_matches("The service is healthy.")


@pytest.mark.asyncio
@pytest.mark.parametrize("initial, expected", [
    ("Checking.", "💬 Checking."),
    ("💬 Checking.", "💬 Checking."),
    (":speech_balloon: Checking.", ":speech_balloon: Checking."),
    ("💬💬 Checking.", "💬💬 Checking."),
])
async def test_interim_commentary_receives_exactly_one_marker(adapter, initial, expected):
    slack, client = adapter
    result = await slack.send(
        "CTEST", initial,
        metadata={"_interim_send": True, "_assistant_commentary": True},
    )
    assert result.success
    assert client.chat_postMessage.await_args.kwargs["text"] == expected


@pytest.mark.parametrize("initial, expected", [
    ("💬 Answer", "💬 Answer"),
    (":speech_balloon: Answer", ":speech_balloon: Answer"),
    ("💬💬 Answer", "💬💬 Answer"),
    ("   ", "   "),
    ("    indented text", "    indented text"),
    ("An embedded 💬 stays.", "An embedded 💬 stays."),
    ("`💬` names the emoji.", "`💬` names the emoji."),
])
def test_final_answers_remain_byte_for_byte_canonical(initial, expected):
    metadata = None
    assert style_public_message(initial, metadata, {"interim_speech_balloon": True}) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [
    {"_interim_send": True},
    {"_interim_send": True, "notify": True},
    {"_interim_send": True, "is_approval_prompt": True},
    {"expect_edits": True},
])
async def test_non_commentary_interim_messages_are_not_ballooned(adapter, metadata):
    slack, client = adapter
    result = await slack.send("CTEST", "Checking.", metadata=metadata)
    assert result.success
    assert client.chat_postMessage.await_args.kwargs["text"] == "Checking."


@pytest.mark.parametrize("value", [False, None, "false", 0])
def test_style_is_explicit_opt_in(value):
    text = "💬 Existing content."
    metadata = {"_interim_send": True, "_assistant_commentary": True}
    assert style_public_message(text, metadata, {"interim_speech_balloon": value}) == text


@pytest.mark.parametrize("value", [True, "true", "1", "yes", "on", 1])
def test_style_accepts_standard_truthy_config_forms(value):
    metadata = {"_interim_send": True, "_assistant_commentary": True}
    assert style_public_message("Checking.", metadata, {"interim_speech_balloon": value}) == "💬 Checking."


@pytest.mark.asyncio
async def test_failed_interim_send_is_not_recorded_as_delivered(adapter):
    slack, client = adapter
    client.chat_postMessage.side_effect = RuntimeError("test transport failure")
    consumer = GatewayStreamConsumer(slack, "CTEST")
    assert await consumer._send_commentary("Checking.") is False
    assert consumer.has_delivered_text("Checking.") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical", ["💬 Answer", ":speech_balloon: Answer", "💬💬 Answer"])
async def test_native_final_preserves_canonical_message(adapter, canonical):
    slack, client = adapter
    metadata = {"thread_id": "1.0"}
    draft = await slack.send_draft("CTEST", 7, canonical, metadata=metadata)
    assert draft.success
    key = slack._stream_key("CTEST", metadata)
    stream = slack._active_streams[key]

    result = await slack.send("CTEST", canonical, metadata=metadata)

    client.chat_postMessage.assert_not_awaited()
    assert result.success
    assert result.message_id == draft.message_id
    client.chat_startStream.assert_awaited_once_with(
        channel="CTEST", thread_ts="1.0", markdown_text=canonical)
    client.chat_stopStream.assert_awaited_once_with(
        channel="CTEST", ts=draft.message_id)
    # Canonical final does not get stripped or rewritten via chat_update
    client.chat_update.assert_not_awaited()
    assert stream["sent"] == canonical
    assert key not in slack._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("text", ["💬 Answer", ":speech_balloon: Answer", "💬💬 Answer"])
async def test_final_edit_preserves_canonical_text(adapter, enabled, text):
    slack, client = adapter
    slack.config.extra["interim_speech_balloon"] = enabled
    metadata = {"thread_id": "1.0"}
    preview = await slack.send(
        "CTEST", "Starting", metadata={**metadata, "expect_edits": True})
    assert preview.success

    final = await slack.edit_message(
        "CTEST", preview.message_id, text, finalize=True, metadata=metadata)
    assert final.success
    assert final.message_id == preview.message_id
    assert client.chat_update.await_args.kwargs == {
        "channel": "CTEST", "ts": preview.message_id,
        "text": text}


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["_interim_send", "expect_edits"])
@pytest.mark.parametrize("interim_text", ["Answer", "💬 Answer"])
async def test_public_interim_send_cannot_seal_or_rewrite_native_stream(adapter, flag, interim_text):
    slack, client = adapter
    metadata = {"thread_id": "1.0"}
    canonical = "💬 Answer"
    draft = await slack.send_draft("CTEST", 7, canonical, metadata=metadata)
    assert draft.success
    key = slack._stream_key("CTEST", metadata)
    stream = slack._active_streams[key]
    interim_metadata = {**metadata, flag: True}
    if flag == "_interim_send":
        interim_metadata["_assistant_commentary"] = True
    original_interim_metadata = dict(interim_metadata)

    interim = await slack.send("CTEST", interim_text, metadata=interim_metadata)

    assert interim.success
    assert interim.message_id != draft.message_id
    client.chat_stopStream.assert_not_awaited()
    client.chat_update.assert_not_awaited()
    client.chat_postMessage.assert_awaited_once()
    assert client.chat_postMessage.await_args.kwargs["text"] == (
        "💬 Answer" if flag == "_interim_send" else interim_text)
    assert slack._active_streams[key] is stream
    assert stream["sent"] == canonical
    assert interim_metadata == original_interim_metadata

    continued = canonical + " continued"
    assert (await slack.send_draft("CTEST", 7, continued, metadata=metadata)).success
    client.chat_appendStream.assert_awaited_once_with(
        channel="CTEST", ts=draft.message_id, markdown_text=" continued")
    assert stream["sent"] == continued
    final = await slack.send("CTEST", continued, metadata=metadata)
    assert final.success
    assert final.message_id == draft.message_id
    client.chat_postMessage.assert_awaited_once()  # Only the public interim, never a second final.
    client.chat_stopStream.assert_awaited_once_with(
        channel="CTEST", ts=draft.message_id)
    client.chat_update.assert_not_awaited()
    assert key not in slack._active_streams
