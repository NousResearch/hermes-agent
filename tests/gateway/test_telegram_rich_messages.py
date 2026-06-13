"""Tests for Bot API 10.1 Rich Messages (sendRichMessage) on Telegram.

Final / new-message replies opportunistically use ``sendRichMessage`` with the
RAW agent markdown so tables, task lists, etc. render natively. The legacy
MarkdownV2 ``send_message`` path stays as the fallback for unsupported /
oversized content and for transports that lack the endpoint.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``
(:func:`_ensure_telegram_mock`), so these tests construct a real
``TelegramAdapter`` and wire a mock bot.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.telegram import TelegramAdapter
from telegram.error import BadRequest, NetworkError, TimedOut


# Content exercising rich-only constructs: a heading, a real Markdown table,
# and a task list. Pipes / brackets must survive untouched into the payload.
RICH_CONTENT = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ✅ |\n\n- [x] table renders"

# PTB 22.6's real unknown-endpoint errors (verified against the wheel):
# do_api_request raises EndPointNotFound on HTTP 404; the request layer can
# wrap the same 404 as InvalidToken. Class NAMES and messages must match.
EndPointNotFound = type("EndPointNotFound", (Exception,), {})
InvalidToken = type("InvalidToken", (Exception,), {})
PTB_ENDPOINT_NOT_FOUND = EndPointNotFound(
    "Endpoint 'sendRichMessage' not found in Bot API"
)
PTB_INVALID_TOKEN_404 = InvalidToken(
    "Either the bot token was rejected by Telegram or the endpoint "
    "'sendRichMessage' does not exist."
)


class FakeRetryAfter(Exception):
    """Mimics telegram.error.RetryAfter (flood control, HTTP 429)."""

    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for the rich path."""
    config = PlatformConfig(enabled=True, token="fake-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    # do_api_request as an AsyncMock makes inspect.iscoroutinefunction(...) True,
    # so _bot_supports_rich() is satisfied (real Bot.do_api_request is async too).
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()  # keeps the post-send typing re-trigger quiet
    bot.send_message_draft = AsyncMock(return_value=True)  # legacy draft fallback
    adapter._bot = bot
    return adapter


def _rich_api_kwargs(adapter):
    """Return the api_kwargs dict from the last sendRichMessage call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    return call.kwargs["api_kwargs"]


@pytest.fixture()
def fast_sleep(monkeypatch):
    """Neutralize retry backoff sleeps inside the adapter's send loops."""
    sleeper = AsyncMock()
    monkeypatch.setattr("gateway.platforms.telegram.asyncio.sleep", sleeper)
    return sleeper


@pytest.mark.asyncio
async def test_rich_happy_path_sends_raw_markdown():
    adapter = _make_adapter()

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == "123"
    adapter._bot.do_api_request.assert_awaited_once()
    api_kwargs = _rich_api_kwargs(adapter)
    # Raw markdown — NOT MarkdownV2-escaped. Table pipes still present.
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    assert "| Case | Status |" in api_kwargs["rich_message"]["markdown"]
    assert "- [x] table renders" in api_kwargs["rich_message"]["markdown"]
    # Legacy path must not run on rich success.
    adapter._bot.send_message.assert_not_called()
    # Parity with the legacy success path: typing is re-triggered.
    adapter._bot.send_chat_action.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,expected_id",
    [
        (SimpleNamespace(message_id=123), "123"),
        ({"message_id": 123}, "123"),
        ({"result": {"message_id": 123}}, "123"),
        ({"result": None}, None),  # malformed envelope must not crash
    ],
)
async def test_rich_result_shapes_extract_message_id(raw, expected_id):
    """Without return_type, real PTB returns the raw dict — handle all shapes."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(return_value=raw)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == expected_id
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_rich_opt_out_uses_legacy():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_opt_out_accepts_string_false():
    adapter = _make_adapter(extra={"rich_messages": "false"})

    await adapter.send("12345", RICH_CONTENT)

    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_oversized_content_skips_rich_and_chunks():
    adapter = _make_adapter()
    # > 32,768 characters -> rich pre-check fails, legacy chunking takes over.
    oversized = "a" * 40000
    assert len(oversized) > TelegramAdapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", oversized)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    # Oversized content is split into multiple legacy chunks.
    assert adapter._bot.send_message.await_count > 1


@pytest.mark.asyncio
async def test_rich_limit_is_characters_not_bytes():
    """The Bot API limit is 32,768 UTF-8 *characters*; multi-byte text under
    the character cap must stay on the rich path even when its UTF-8 byte
    length exceeds 32,768 (CJK/emoji-heavy locales)."""
    adapter = _make_adapter()
    cjk = "测" * 20000  # 20k chars, 60k UTF-8 bytes
    assert len(cjk.encode("utf-8")) > 32768
    assert len(cjk) <= TelegramAdapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", cjk)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BadRequest("can't parse rich message"),
        BadRequest("Method not found"),
        RuntimeError("Method not found"),  # non-BadRequest capability signal
    ],
)
async def test_permanent_rich_error_falls_back_to_legacy(exc):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_awaited()  # legacy fallback ran


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [PTB_ENDPOINT_NOT_FOUND, PTB_INVALID_TOKEN_404])
async def test_real_ptb_endpoint_missing_falls_back_and_latches_off(exc):
    """PTB 22.6's actual unknown-endpoint errors (EndPointNotFound, and the
    request layer's InvalidToken wrap of HTTP 404) must fall back to legacy —
    a server without Bot API 10.1 would otherwise lose EVERY message — and
    latch rich sends off so later sends skip the doomed round-trip."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_awaited()
    assert adapter._rich_send_disabled is True

    # Subsequent sends go straight to legacy without re-trying rich.
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.send_message.reset_mock()
    result2 = await adapter.send("12345", RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_content_parse_badrequest_does_not_latch_sends_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("can't parse rich message"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert adapter._rich_send_disabled is False  # content error, not capability


@pytest.mark.asyncio
async def test_flood_control_retries_rich_in_place(fast_sleep):
    """RetryAfter (HTTP 429) means Telegram rejected the request — retrying
    the rich call after the mandated wait is duplicate-safe and must not
    drop the message or fall back to legacy."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=[FakeRetryAfter(5), SimpleNamespace(message_id=7)]
    )

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == "7"
    assert adapter._bot.do_api_request.await_count == 2
    adapter._bot.send_message.assert_not_called()
    # The server-mandated wait was honored.
    assert any(call.args[0] == 5.0 for call in fast_sleep.await_args_list)


@pytest.mark.asyncio
async def test_network_error_retries_then_fails_retryable(fast_sleep):
    """Pre-connect network errors retry with backoff (legacy parity); after
    exhaustion the failure is retryable and never legacy-resent."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=NetworkError("connection reset"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    assert result.retryable is True
    assert adapter._bot.do_api_request.await_count == 3
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_network_error_recovers_on_retry(fast_sleep):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=[NetworkError("connection reset"), SimpleNamespace(message_id=9)]
    )

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == "9"
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transient_timeout_does_not_retry_or_legacy_resend():
    """A plain (non-connect) timeout may have reached Telegram: exactly one
    attempt, no legacy resend, and non-retryable so upstream doesn't re-send."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    assert result.retryable is False
    assert adapter._bot.do_api_request.await_count == 1
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_connect_timeout_retries_and_is_retryable(fast_sleep):
    """A connect-timeout never reached Telegram — safe to retry in place and
    safe for upstream to retry the send."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("connect timed out"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    assert result.retryable is True
    assert adapter._bot.do_api_request.await_count == 3
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_routing_thread_id_maps_to_message_thread_id():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"thread_id": "5"})

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["message_thread_id"] == 5
    assert "direct_messages_topic_id" not in api_kwargs


@pytest.mark.asyncio
async def test_routing_direct_messages_topic_id_drops_message_thread_id():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"direct_messages_topic_id": "20189"})

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["direct_messages_topic_id"] == 20189
    # _thread_kwargs_for_send pairs the topic id with message_thread_id=None;
    # the rich payload must drop the None key, not send a stray field.
    assert "message_thread_id" not in api_kwargs


@pytest.mark.asyncio
async def test_reply_to_propagates_as_reply_parameters():
    """sendRichMessage takes reply_parameters; the pre-7.0 scalar
    reply_to_message_id is not part of the 10.1 contract."""
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, reply_to="999")

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["reply_parameters"] == {"message_id": 999}
    assert "reply_to_message_id" not in api_kwargs


@pytest.mark.asyncio
async def test_reply_to_combined_with_thread_id():
    """Replying inside a forum topic must carry BOTH routing fields."""
    adapter = _make_adapter()

    await adapter.send(
        "-100123", RICH_CONTENT, reply_to="999", metadata={"thread_id": "5"}
    )

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["message_thread_id"] == 5
    assert api_kwargs["reply_parameters"] == {"message_id": 999}


@pytest.mark.asyncio
async def test_dm_topic_fail_loud_skips_rich_and_refuses():
    """DM-topic sends without a reply anchor must refuse loudly via the
    legacy path — the rich path must not send outside the requested topic."""
    adapter = _make_adapter()

    result = await adapter.send(
        "123",  # private chat id
        RICH_CONTENT,
        metadata={"thread_id": "20197", "telegram_dm_topic_reply_fallback": True},
    )

    assert result.success is False
    assert "reply anchor" in (result.error or "")
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_notification_silent_by_default():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT)

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["disable_notification"] is True


@pytest.mark.asyncio
async def test_notification_opt_in_drops_disable_flag():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"notify": True})

    api_kwargs = _rich_api_kwargs(adapter)
    assert "disable_notification" not in api_kwargs


@pytest.mark.asyncio
async def test_disable_link_previews_carries_to_rich_payload():
    adapter = _make_adapter(extra={"disable_link_previews": True})

    await adapter.send("-100123", RICH_CONTENT)

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["link_preview_options"] == {"is_disabled": True}


@pytest.mark.asyncio
async def test_link_previews_enabled_by_default_in_rich_payload():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT)

    api_kwargs = _rich_api_kwargs(adapter)
    assert "link_preview_options" not in api_kwargs


@pytest.mark.asyncio
async def test_expect_edits_metadata_skips_rich():
    """Sends that will be edited later (streaming previews, tool-progress and
    status bubbles) must stay on the editable MarkdownV2 path — editing a
    rich-born message would flip its rendering format."""
    adapter = _make_adapter()

    result = await adapter.send(
        "12345", RICH_CONTENT, metadata={"expect_edits": True}
    )

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_gate_tolerates_missing_enabled_attr():
    """Adapters missing _rich_messages_enabled (object.__new__ in some tests)
    must not raise — the gate reads it via getattr(default=True), and a bot
    without an async do_api_request falls through to the legacy path."""
    adapter = _make_adapter()
    del adapter._rich_messages_enabled  # simulate object.__new__ construction
    del adapter._rich_send_disabled
    # SimpleNamespace bot has no do_api_request -> _bot_supports_rich() False.
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        send_chat_action=AsyncMock(),
    )

    result = await adapter.send("12345", "hello world")

    assert result.success is True
    assert result.message_id == "42"


# ── Streaming drafts: sendRichMessageDraft ─────────────────────────────


@pytest.mark.asyncio
async def test_rich_draft_happy_path_sends_raw_markdown():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessageDraft"
    api_kwargs = call.kwargs["api_kwargs"]
    assert api_kwargs["draft_id"] == 7
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    # Legacy plain-text draft must not run when rich draft succeeds.
    adapter._bot.send_message_draft.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BadRequest("Method not found"),
        PTB_ENDPOINT_NOT_FOUND,
        PTB_INVALID_TOKEN_404,
    ],
)
async def test_rich_draft_capability_failure_falls_back_and_latches_off(exc):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy plain-text draft delivered the frame
    adapter._bot.send_message_draft.assert_awaited_once()
    assert adapter._rich_draft_disabled is True

    # A subsequent frame skips the rich attempt entirely (latched off).
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.send_message_draft.reset_mock()
    result2 = await adapter.send_draft("12345", draft_id=8, content=RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_parse_error_does_not_latch_off():
    """Draft frames carry PARTIAL markdown (open fences, unterminated tables)
    — a per-frame parse 400 must fall back for that frame only, NOT disable
    rich drafts for the adapter's lifetime."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("can't parse rich message"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy draft carried this frame
    adapter._bot.send_message_draft.assert_awaited_once()
    assert adapter._rich_draft_disabled is False

    # The next frame tries rich again.
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.do_api_request.side_effect = None
    adapter._bot.do_api_request.return_value = True
    result2 = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_transient_failure_does_not_latch_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy draft carried this frame
    adapter._bot.send_message_draft.assert_awaited_once()
    # Transient errors must NOT permanently disable rich drafts.
    assert adapter._rich_draft_disabled is False


@pytest.mark.asyncio
async def test_rich_draft_opt_out_uses_legacy():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_oversized_uses_legacy():
    adapter = _make_adapter()
    oversized = "a" * 40000

    result = await adapter.send_draft("12345", draft_id=7, content=oversized)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


# ----------------------------------------------------------------------
# prefers_fresh_final_streaming: the stream consumer asks the adapter whether
# to finalize a streamed reply by sending a fresh (rich) message + deleting the
# preview, instead of final-editing the preview through the non-rich edit path.
# Telegram opts in exactly when the content is rich-eligible.
# ----------------------------------------------------------------------
def test_prefers_fresh_final_streaming_when_rich_enabled():
    adapter = _make_adapter()
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT) is True


def test_prefers_fresh_final_streaming_false_when_rich_disabled():
    adapter = _make_adapter(extra={"rich_messages": False})
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT) is False


# ----------------------------------------------------------------------
# streaming_overflow_limit: with rich on, the stream consumer may accumulate up
# to the 32,768-char rich cap before splitting, so a reply that fits one
# sendRichMessage / sendRichMessageDraft isn't fragmented at the 4,096 limit.
# ----------------------------------------------------------------------
def test_streaming_overflow_limit_is_rich_cap_when_enabled():
    adapter = _make_adapter()
    assert adapter.streaming_overflow_limit() == TelegramAdapter.RICH_MESSAGE_MAX_CHARS


def test_streaming_overflow_limit_none_when_rich_disabled():
    adapter = _make_adapter(extra={"rich_messages": False})
    assert adapter.streaming_overflow_limit() is None


def test_streaming_overflow_limit_none_when_rich_latched_off():
    adapter = _make_adapter()
    adapter._rich_send_disabled = True
    assert adapter.streaming_overflow_limit() is None
