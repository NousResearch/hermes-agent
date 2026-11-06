"""Collapsed voice echoes through real inbound preparation and Telegram delivery."""

import asyncio
from dataclasses import asdict
import json

import pytest
import pytest_asyncio
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.stt_echo import (
    chunk_telegram_stt_echo_html,
    format_stt_transcript_echo,
    stt_echo_metadata,
)
from tests.gateway._telegram_stt_test_support import (
    RecordingTelegramBot,
    assert_stt_echo,
    telegram_adapter,
)


@pytest_asyncio.fixture(autouse=True)
async def virtual_clock(monkeypatch):
    """Advance the event-loop clock at the sleep boundary without real retry waits."""
    loop = asyncio.get_running_loop()
    clock = {"now": loop.time(), "sleeps": []}
    original_sleep = asyncio.sleep

    async def sleep(delay, result=None):
        clock["sleeps"].append(delay)
        clock["now"] += delay
        return await original_sleep(0, result)

    monkeypatch.setattr(loop, "time", lambda: clock["now"])
    monkeypatch.setattr(asyncio, "sleep", sleep)
    yield clock


def _runner(adapter, *, echo=True):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict({"stt": {"enabled": True, "echo_transcripts": echo}})
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("echo,is_bot", [(True, False), (False, False), (True, True)])
async def test_inbound_voice_echo_preserves_agent_input_and_notice_policy(monkeypatch, tmp_path, echo, is_bot):
    """Catch missing HTML wiring, caption corruption and disabled/bot-peer echo leaks."""
    transcript = " \n**speech** & <b>literal</b> 😀\n  "
    bot = RecordingTelegramBot()
    adapter = telegram_adapter(bot)
    runner = _runner(adapter, echo=echo)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm",
        thread_id="41", is_bot=is_bot,
    )
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"voice fixture")
    event = MessageEvent(
        text="caption", message_id="91", source=source, message_type=MessageType.VOICE,
        media_urls=[str(audio)], media_types=["audio/ogg"],
    )
    transcribed = []

    def transcribe(path, *_args):
        transcribed.append(path)
        return {"success": True, "transcript": transcript, "provider": "fixture"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    text = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    assert text == f'"{transcript}"\n\ncaption'
    assert transcribed == [str(audio)]
    assert event.text == "caption"
    assert event.media_urls == [str(audio)]
    assert event.media_types == ["audio/ogg"]
    if echo and not is_bot:
        assert len(bot.delivered) == 1
        assert_stt_echo(bot.delivered, transcript)
        assert bot.delivered[0]["message_thread_id"] == 41
        assert bot.delivered[0]["reply_to_message_id"] == 91
        assert bot.delivered[0]["disable_notification"] is True
    else:
        assert bot.attempts == []
    assert bot.rich_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript,minimum_chunks",
    [
        ('\n <details>\n<summary>literal</summary>\n$$x$$ &amp; &#x1f600; </blockquote>\n\t ', 1),
        ("x" * 4056, 1),
        ("x" * 4057, 2),
        ("😀" * 2028, 1),
        ("😀" * 2029, 2),
        (" \n" + "&<😀> \n\t" * 1800 + "\n  ", 2),
    ],
    ids=["injection-and-rich-markup", "wire-limit", "wire-overflow", "utf16-limit", "utf16-overflow", "escaped-continuations"],
)
async def test_send_keeps_each_voice_chunk_expandable_and_lossless(transcript, minimum_chunks):
    """Catch Markdown/rich diversion, trimming, entity cuts and UTF-16 overflow."""
    adapter = telegram_adapter()
    route = {"thread_id": "41", "hermes_profile": "satellite", "notify": True}
    metadata = stt_echo_metadata(Platform.TELEGRAM, route)
    result = await adapter.send("-100123", format_stt_transcript_echo(transcript, Platform.TELEGRAM), metadata=metadata)

    assert result.success
    assert len(adapter._bot.delivered) >= minimum_chunks
    if minimum_chunks == 1:
        assert len(adapter._bot.delivered) == 1
    assert_stt_echo(adapter._bot.delivered, transcript)
    assert all(message["message_thread_id"] == 41 for message in adapter._bot.delivered)
    assert all(not message.get("disable_notification", False) for message in adapter._bot.delivered)
    assert adapter._bot.rich_requests == []
    assert route == {"thread_id": "41", "hermes_profile": "satellite", "notify": True}


@pytest.mark.parametrize("budget,size", [(64, 300), (64, 3000)])
def test_chunk_pagination_growth_keeps_the_full_wire_budget(budget, size):
    """Catch a 9-to-10/99-to-100 page-count transition overflowing a complete quote."""
    transcript = "x" * size
    chunks = chunk_telegram_stt_echo_html(
        format_stt_transcript_echo(transcript, Platform.TELEGRAM), budget,
        len_fn=lambda text: len(text.encode("utf-16-le")) // 2,
    )
    assert len(chunks) >= (10 if size == 300 else 100)
    assert_stt_echo([{"text": text, "parse_mode": "HTML"} for text in chunks], transcript, wire_limit=budget)


@pytest.mark.asyncio
async def test_confirmed_html_parse_rejection_falls_back_without_losing_speech():
    """Catch escaped entities leaking into plain fallback or routing being dropped."""
    transcript = " \n<&> literal &amp; 😀\n "
    bot = RecordingTelegramBot([BadRequest("Can't parse entities: unsupported start tag blockquote"), None])
    adapter = telegram_adapter(bot)
    result = await adapter.send(
        "12345", format_stt_transcript_echo(transcript, Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, {
            "thread_id": "41", "telegram_dm_topic_reply_fallback": True, "telegram_reply_to_message_id": "91",
        }),
    )

    assert result.success
    assert len(bot.attempts) == 2 and len(bot.delivered) == 1
    assert bot.attempts[0]["parse_mode"] == "HTML"
    plain = bot.delivered[0]
    assert plain["parse_mode"] is None
    assert plain["text"].endswith(transcript)
    assert plain["text"].count(transcript) == 1
    assert plain["message_thread_id"] == 41 and plain["reply_to_message_id"] == 91
    assert plain["disable_notification"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,attempts",
    [(BadRequest("Chat not found"), 1), (NetworkError("Connection reset by peer"), 3), (TimedOut("Timed out"), 1)],
    ids=["target-rejection", "network-failure", "ambiguous-timeout"],
)
async def test_non_parse_failures_do_not_send_plain_transcripts(failure, attempts):
    """Catch plain fallback after a transport or route failure that cannot reject HTML."""
    bot = RecordingTelegramBot([failure] * attempts)
    adapter = telegram_adapter(bot)
    result = await adapter.send(
        "12345", format_stt_transcript_echo("speech", Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM),
    )

    assert not result.success
    assert len(bot.attempts) == attempts
    assert all(message["parse_mode"] == "HTML" for message in bot.attempts)
    assert bot.delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,failure,expected_calls",
    [
        ({"thread_id": "41", "telegram_dm_topic_reply_fallback": True}, None, 0),
        ({"thread_id": "41", "telegram_dm_topic_reply_fallback": True, "telegram_reply_to_message_id": "91"},
         "Message thread not found", 1),
        ({"thread_id": "41", "telegram_dm_topic_reply_fallback": True, "telegram_reply_to_message_id": "91"},
         "Message to be replied not found", 1),
        ({"thread_id": "41", "telegram_dm_topic_created_for_send": True}, "Message thread not found", 1),
        ({"direct_messages_topic_id": "41"}, "Direct messages topic not found", 1),
    ],
    ids=["missing-anchor", "dead-hermes-topic", "deleted-anchor", "dead-created-topic", "dead-native-topic"],
)
async def test_private_topic_echo_failures_never_escape_to_general_chat(route, failure, expected_calls):
    """Catch a failed collapsed echo being retried outside its private topic."""
    bot = RecordingTelegramBot([BadRequest(failure)] if failure else [])
    adapter = telegram_adapter(bot)
    result = await adapter.send(
        "12345", format_stt_transcript_echo("private speech", Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, route),
    )

    assert not result.success
    assert len(bot.attempts) == expected_calls
    assert bot.delivered == []
    assert all(message["parse_mode"] == "HTML" for message in bot.attempts)
    assert all(message.get("message_thread_id") == 41 or message.get("direct_messages_topic_id") == 41 for message in bot.attempts)


@pytest.mark.asyncio
async def test_forum_echo_preserves_the_existing_deleted_thread_fallback():
    """Catch HTML parsing fallback swallowing the ordinary forum route-recovery ladder."""
    bot = RecordingTelegramBot([BadRequest("Message thread not found"), BadRequest("Message thread not found"), None])
    adapter = telegram_adapter(bot)
    result = await adapter.send(
        "-100123", format_stt_transcript_echo("forum speech", Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, {"thread_id": "41"}),
    )

    assert result.success
    assert [message["message_thread_id"] for message in bot.attempts] == [41, 41, None]
    assert len(bot.delivered) == 1
    assert_stt_echo(bot.delivered, "forum speech")


@pytest.mark.asyncio
async def test_split_echo_retry_sends_only_the_undelivered_html_tail():
    """Catch retrying an already-visible head or resuming escaped HTML as Markdown."""
    bot = RecordingTelegramBot([None, RetryAfter(7), None, None])
    adapter = telegram_adapter(bot)
    transcript = "&😀\n" * 1600
    result = await adapter._send_with_retry(
        "-100123", format_stt_transcript_echo(transcript, Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, {"thread_id": "41"}),
    )

    assert result.success
    assert_stt_echo(bot.delivered, transcript)
    assert len(bot.attempts) == len(bot.delivered) + 1
    assert all(message["message_thread_id"] == 41 for message in bot.delivered)


@pytest.mark.asyncio
async def test_partial_echo_receipt_round_trip_retains_html_and_cooldown(virtual_clock):
    """Catch receipt restoration losing HTML mode or a cooldown leaking across chats."""
    bot = RecordingTelegramBot([None, RetryAfter(120)])
    adapter = telegram_adapter(bot)
    transcript = "&😀\n" * 1600
    route = {"thread_id": "41"}
    result = await adapter.send(
        "-100123", format_stt_transcript_echo(transcript, Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, route),
    )
    assert not result.success
    assert result.raw_response["partial_overflow"] is True
    assert result.retry_after == 120
    assert len(bot.attempts) == 2
    assert not virtual_clock["sleeps"]

    held = await adapter.send(
        "-100123", format_stt_transcript_echo("later speech", Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM, route),
    )
    assert not held.success and len(bot.attempts) == 2
    other = await adapter.send("555", "other chat")
    assert other.success and len(bot.attempts) == 3
    virtual_clock["now"] += 121
    restored = SendResult(**json.loads(json.dumps(asdict(result))))
    resumed = await adapter._resume_partial_send("-100123", restored, reply_to=None, metadata=route)

    assert resumed.success
    echoes = [message for message in bot.delivered if message["chat_id"] == -100123]
    assert_stt_echo(echoes, transcript)
    assert route == {"thread_id": "41"}


@pytest.mark.asyncio
async def test_ambiguous_timeout_after_a_landed_echo_chunk_never_resends_it():
    """Catch duplicated speech when a timed-out continuation may already be visible."""
    bot = RecordingTelegramBot([None, TimedOut("Timed out")])
    adapter = telegram_adapter(bot)
    result = await adapter._send_with_retry(
        "12345", format_stt_transcript_echo("x" * 9000, Platform.TELEGRAM),
        metadata=stt_echo_metadata(Platform.TELEGRAM),
    )

    assert not result.success
    assert result.retryable is False
    assert result.raw_response["partial_overflow"] is True
    assert not result.raw_response.get("undelivered_chunks")
    assert len(bot.attempts) == 2 and len(bot.delivered) == 1
    assert all(message["parse_mode"] == "HTML" for message in bot.attempts)
