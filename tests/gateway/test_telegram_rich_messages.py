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
DANGEROUS_DETAILS_MATH = (
    "<details><summary>Complex proof</summary>\n\n"
    "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n\n"
    "And inline \\(\\alpha + \\beta\\)\n"
    "</details>"
)

# PTB 22.6's real unknown-endpoint errors: do_api_request can raise
# EndPointNotFound for Bot API 404s, and the request layer can wrap that same
# missing endpoint as InvalidToken. Use class names here so the tests don't
# depend on optional PTB internals.
EndPointNotFound = type("EndPointNotFound", (Exception,), {})
InvalidToken = type("InvalidToken", (Exception,), {})
PTB_ENDPOINT_NOT_FOUND = EndPointNotFound(
    "Endpoint 'sendRichMessage' not found in Bot API"
)
PTB_INVALID_TOKEN_404 = InvalidToken(
    "Either the bot token was rejected by Telegram or the endpoint "
    "'sendRichMessage' does not exist."
)


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for the rich path."""
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"rich_messages": True, **(extra or {})},
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    # do_api_request as an AsyncMock makes inspect.iscoroutinefunction(...) True,
    # so _bot_supports_rich() is satisfied (real Bot.do_api_request is async too).
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()  # keeps the post-send typing re-trigger quiet
    bot.send_message_draft = AsyncMock(return_value=True)  # legacy draft fallback
    bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=1))  # legacy edit path
    bot.delete_message = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


def _rich_api_kwargs(adapter):
    """Return the api_kwargs dict from the single sendRichMessage call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected_id"),
    [
        (SimpleNamespace(message_id=123), "123"),
        ({"message_id": 123}, "123"),
        ({"result": {"message_id": 123}}, "123"),
        ({"result": None}, None),
    ],
)
async def test_rich_result_shapes_extract_message_id(raw, expected_id):
    """The raw Bot API path may return either a PTB object or a raw dict."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(return_value=raw)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == expected_id
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


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


@pytest.mark.asyncio
async def test_details_with_math_skips_rich_send_to_avoid_tdesktop_crash():
    adapter = _make_adapter()

    result = await adapter.send("12345", DANGEROUS_DETAILS_MATH)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_details_without_math_still_uses_rich_send():
    adapter = _make_adapter()

    result = await adapter.send(
        "12345",
        "<details><summary>Notes</summary>\nNo equations here.\n</details>",
    )

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_math_outside_details_still_uses_rich_send():
    adapter = _make_adapter()

    result = await adapter.send("12345", "Outside details: $$x^2 + y^2$$")

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_rich_messages_opt_out_uses_legacy_send_path():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_messages_opt_out_accepts_string_false():
    adapter = _make_adapter(extra={"rich_messages": "false"})

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_messages_default_is_enabled():
    """Rich messages are on by default (Bot API 10.1); rich-eligible content
    (tables/task lists/details/math) goes through sendRichMessage without the
    user having to opt in."""
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_rich_messages_can_be_opted_out():
    """Setting platforms.telegram.extra.rich_messages: false keeps every reply
    on the legacy MarkdownV2 path even for rich-eligible content."""
    config = PlatformConfig(
        enabled=True, token="fake-token", extra={"rich_messages": False}
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_plain_markdown_stays_on_legacy_path():
    """Ordinary replies (no table/task-list/details/math) stay on the legacy
    MarkdownV2 path for consistent client rendering, even with rich enabled."""
    adapter = _make_adapter()

    result = await adapter.send("12345", "Hello **there**\n\nA normal reply.")

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_expect_edits_metadata_keeps_preview_on_legacy_path():
    adapter = _make_adapter()

    result = await adapter.send(
        "12345",
        RICH_CONTENT,
        metadata={"expect_edits": True},
    )

    assert result.success is True
    # Streaming preview sends will be edited later, so they must not be born as
    # rich messages until Hermes wires rich_message edits directly.
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


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
    """Telegram's rich limit is UTF-8 characters, not encoded bytes."""
    adapter = _make_adapter()
    # Rich-eligible (table) so the content takes the rich path; the CJK body
    # is 20k chars / 60k UTF-8 bytes — over the byte count, under the char cap.
    cjk = "| a | b |\n|---|---|\n" + "测" * 20000  # 20k chars, ~60k UTF-8 bytes
    assert len(cjk.encode("utf-8")) > TelegramAdapter.RICH_MESSAGE_MAX_BYTES
    assert len(cjk) <= TelegramAdapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", cjk)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BadRequest("can't parse rich message"),
        BadRequest("Method not found"),
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
async def test_unknown_endpoint_error_falls_back_to_legacy():
    """A non-BadRequest 'Method not found' (old PTB/endpoint) degrades gracefully."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method not found"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_capability_error_latches_rich_send_off():
    """Endpoint-missing errors latch rich off so later sends skip the
    doomed extra roundtrip entirely."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method not found"))

    result = await adapter.send("12345", RICH_CONTENT)
    assert result.success is True
    assert adapter._rich_send_disabled is True

    # Second send skips rich entirely (no second do_api_request call).
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.send_message.reset_mock()
    result2 = await adapter.send("12345", RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [PTB_ENDPOINT_NOT_FOUND, PTB_INVALID_TOKEN_404])
async def test_real_ptb_endpoint_missing_falls_back_and_latches_off(exc):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_awaited()
    assert adapter._rich_send_disabled is True


@pytest.mark.asyncio
async def test_rich_payload_preserves_link_preview_disable():
    adapter = _make_adapter(extra={"disable_link_previews": True})

    result = await adapter.send(
        "12345", "| Link | Note |\n|---|---|\n| See https://example.com | x |"
    )

    assert result.success is True
    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["link_preview_options"] == {"is_disabled": True}


@pytest.mark.asyncio
async def test_per_message_bad_request_does_not_latch_off():
    """A parser/limit BadRequest is per-message — rich must stay enabled
    for subsequent messages."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("can't parse rich message"))

    result = await adapter.send("12345", RICH_CONTENT)
    assert result.success is True
    assert adapter._rich_send_disabled is False

    # Next message re-attempts rich.
    adapter._bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=124))
    result2 = await adapter.send("12345", RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimedOut("timed out"), NetworkError("connection reset")])
async def test_transient_rich_error_does_not_legacy_resend(exc):
    """Transient transport errors must NOT trigger a legacy resend (duplicate risk)."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transient_timeout_is_not_retryable():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send("12345", RICH_CONTENT)

    # A plain timeout may have reached Telegram -> non-retryable (no auto-resend).
    assert result.success is False
    assert result.retryable is False


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
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, reply_to="999")

    api_kwargs = _rich_api_kwargs(adapter)
    # Spec: sendRichMessage documents reply_parameters (ReplyParameters), not
    # the legacy reply_to_message_id scalar — unknown params are silently
    # ignored, which would quietly drop the reply anchor.
    assert api_kwargs["reply_parameters"] == {"message_id": 999}
    assert "reply_to_message_id" not in api_kwargs


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
async def test_rich_gate_tolerates_minimal_bot_without_raw_endpoint():
    """A bot without an async do_api_request falls through to the legacy path."""
    adapter = _make_adapter()
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        send_chat_action=AsyncMock(),
    )

    result = await adapter.send("12345", "hello world")

    assert result.success is True
    assert result.message_id == "42"


# ── Streaming drafts: sendRichMessageDraft ─────────────────────────────


@pytest.mark.asyncio
async def test_details_with_math_skips_rich_draft_to_avoid_tdesktop_crash():
    adapter = _make_adapter()
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=DANGEROUS_DETAILS_MATH)

    assert result.success is True
    bot.do_api_request.assert_not_called()
    bot.send_message_draft.assert_awaited_once()


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
async def test_rich_draft_capability_failure_falls_back_and_latches_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("Method not found"))

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
async def test_rich_draft_transient_failure_does_not_latch_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy draft carried this frame
    adapter._bot.send_message_draft.assert_awaited_once()
    # Transient errors must NOT permanently disable rich drafts.
    assert adapter._rich_draft_disabled is False


@pytest.mark.asyncio
async def test_rich_draft_oversized_uses_legacy():
    adapter = _make_adapter()
    oversized = "a" * 40000

    result = await adapter.send_draft("12345", draft_id=7, content=oversized)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


# ----------------------------------------------------------------------
# prefers_fresh_final_streaming: Telegram keeps streamed finals on the edit
# path, even when rich messages are enabled, so users do not briefly see two
# copies of the answer while the preview cleanup delete races the fresh send.
# ----------------------------------------------------------------------
def test_prefers_fresh_final_streaming_stays_disabled_when_rich_enabled():
    adapter = _make_adapter()
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT) is False


def test_prefers_fresh_final_streaming_honors_rich_opt_out():
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


def test_streaming_overflow_limit_none_when_rich_opted_out():
    adapter = _make_adapter(extra={"rich_messages": False})
    assert adapter.streaming_overflow_limit() is None


def test_streaming_overflow_limit_none_when_rich_latched_off():
    adapter = _make_adapter()
    adapter._rich_send_disabled = True
    assert adapter.streaming_overflow_limit() is None


@pytest.mark.asyncio
async def test_rich_draft_opt_out_uses_legacy():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message_draft.assert_awaited_once()


# ----------------------------------------------------------------------------
# Rich finalize via editMessageText (Bot API 10.1 rich_message edit param).
# Streamed previews finalize by editing the existing message IN PLACE as rich,
# so tables/task lists survive without a fresh send + delete (no duplicate).
# ----------------------------------------------------------------------------


def _rich_edit_kwargs(adapter):
    """Return the api_kwargs dict from the single editMessageText rich call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "editMessageText"
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_finalize_edit_uses_rich_for_table_content():
    """Finalizing a streamed preview whose content is a table edits the
    existing message IN PLACE via editMessageText's rich_message param —
    no fresh send, no delete, no duplicate."""
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    assert result.message_id == "555"  # same message, edited in place
    api_kwargs = _rich_edit_kwargs(adapter)
    assert api_kwargs["message_id"] == 555
    # RAW markdown is passed through so table pipes survive.
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    # No fresh send / delete — the whole point of the in-place rich edit.
    adapter._bot.edit_message_text.assert_not_called()
    adapter._bot.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_edit_plain_content_stays_legacy():
    """Finalizing plain content (no table/task-list/details/math) uses the
    legacy MarkdownV2 edit_message_text path, not the rich edit endpoint."""
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "12345", "555", "Just a normal answer, no rich constructs.", finalize=True,
    )

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_finalize_edit_rich_capability_error_falls_back_to_legacy():
    """A capability error on the rich edit latches rich off and falls back to
    the legacy MarkdownV2 edit so the user still gets the final answer."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=PTB_ENDPOINT_NOT_FOUND)

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    assert adapter._rich_send_disabled is True
    adapter._bot.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_finalize_edit_rich_not_modified_is_success_noop():
    """'Message is not modified' on a rich edit is a no-op success — must NOT
    fall through to a redundant legacy edit."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=BadRequest("Message is not modified")
    )

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_non_finalize_edit_never_uses_rich():
    """Intermediate (non-finalize) stream edits stay on the plain edit path;
    rich is only applied on the final edit."""
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=False,
    )

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_finalize_edit_opt_out_uses_legacy():
    """With rich_messages: false, even a table finalizes via the legacy
    MarkdownV2 edit path."""
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.edit_message_text.assert_awaited()


@pytest.mark.asyncio
async def test_finalize_edit_rich_over_markdownv2_limit_not_split():
    """A rich table that exceeds the 4,096 MarkdownV2 limit but fits the 32,768
    rich cap is edited in place as one rich message, NOT split into legacy
    chunks."""
    adapter = _make_adapter()
    big_table = "| a | b |\n|---|---|\n" + "\n".join(
        f"| {'x' * 50} | {'y' * 50} |" for _ in range(40)
    )
    assert len(big_table) > TelegramAdapter.MAX_MESSAGE_LENGTH
    assert len(big_table) <= TelegramAdapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.edit_message(
        "12345", "555", big_table, finalize=True,
    )

    assert result.success is True
    api_kwargs = _rich_edit_kwargs(adapter)
    assert api_kwargs["rich_message"]["markdown"] == big_table
    adapter._bot.edit_message_text.assert_not_called()
