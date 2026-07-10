"""Tests for Bot API 10.1 Rich Messages (sendRichMessage) on Telegram.

Final / new-message replies opportunistically use ``sendRichMessage`` with the
RAW agent markdown so tables, task lists, etc. render natively. The legacy
MarkdownV2 ``send_message`` path stays as the fallback for unsupported /
oversized content and for transports that lack the endpoint.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``
(:func:`_ensure_telegram_mock`), so these tests construct a real
``TelegramAdapter`` and wire a mock bot.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest, NetworkError, TimedOut


# Content exercising rich-only constructs: a heading, a real Markdown table,
# and a task list. Pipes / brackets must survive untouched into the payload.
RICH_CONTENT = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ✅ |\n\n- [x] table renders"
CJK_RICH_CONTENT = "## 持仓\n\n| 项目 | 状态 |\n|---|---|\n| 早盘 | 正常 |"
ASTRAL_CJK_RICH_CONTENT = "## Rare Han\n\n| glyph | status |\n|---|---|\n| \U00030000 | ok |"
TABLE_ONLY_CONTENT = (
    "| Team | W | L | GB |\n"
    "|---|---|---|---|\n"
    "| Red Sox | 36 | 34 | 6.0 |\n"
    "| Dodgers | 40 | 30 | 2.0 |"
)
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
async def test_cjk_rich_content_skips_rich_send_to_avoid_tdesktop_garble():
    adapter = _make_adapter()

    result = await adapter.send("12345", CJK_RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_astral_cjk_rich_content_skips_rich_send_to_avoid_tdesktop_garble():
    adapter = _make_adapter()

    result = await adapter.send("12345", ASTRAL_CJK_RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited_once()


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
async def test_rich_messages_default_is_legacy_copyable_path():
    """Rich messages stay opt-in because current Telegram clients can make
    Bot API rich messages hard to copy as plain text. Rich-eligible content
    defaults to the legacy MarkdownV2 path unless the user opts in."""
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
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_messages_can_be_opted_in():
    """Setting platforms.telegram.extra.rich_messages: true enables native
    Bot API rich rendering for tables/task lists/details/math."""
    config = PlatformConfig(
        enabled=True, token="fake-token", extra={"rich_messages": True}
    )
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
    # Rich-eligible (table) so the content takes the rich path; the accented
    # body is 20k chars / 40k UTF-8 bytes — over the byte count, under the
    # character cap. CJK is intentionally avoided here because affected
    # Telegram Desktop clients render CJK rich drafts incorrectly.
    accented = "| a | b |\n|---|---|\n" + "é" * 20000
    assert len(accented.encode("utf-8")) > TelegramAdapter.RICH_MESSAGE_MAX_BYTES
    assert len(accented) <= TelegramAdapter.RICH_MESSAGE_MAX_CHARS

    result = await adapter.send("12345", accented)

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
async def test_rich_transport_error_redacts_bot_token_even_when_redaction_disabled(monkeypatch):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=NetworkError(
            f"Timed out requesting https://api.telegram.org/bot{token}/sendRichMessage"
        )
    )

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/sendRichMessage" in result.error
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_send_error_redacts_bot_token_without_traceback(monkeypatch, caplog):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter({"rich_messages": False})
    adapter._bot.send_message = AsyncMock(
        side_effect=BadRequest(
            f"Bad Request: https://api.telegram.org/bot{token}/sendMessage"
        )
    )

    with caplog.at_level(logging.ERROR):
        result = await adapter.send("12345", "Plain legacy content.")

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/sendMessage" in result.error
    assert token not in caplog.text
    assert "bot123456789:***/sendMessage" in caplog.text
    adapter._bot.do_api_request.assert_not_called()


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
async def test_table_only_uses_rich_when_rich_messages_opt_out():
    """Pipe tables auto-route to sendRichMessage even without the full opt-in."""
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send("12345", TABLE_ONLY_CONTENT)

    assert result.success is True
    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["rich_message"]["markdown"] == TABLE_ONLY_CONTENT
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_table_only_uses_rich_with_default_config():
    """Default config keeps task lists on legacy but upgrades bare tables."""
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    result = await adapter.send("12345", TABLE_ONLY_CONTENT)

    assert result.success is True
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_dm_topic_resumed_send_uses_rich_for_table_without_reply_anchor():
    """Resumed/synthetic DM-topic sends route tables via direct_messages_topic_id."""
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send(
        "123",
        TABLE_ONLY_CONTENT,
        metadata={
            "thread_id": "20189",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "20189",
        },
    )

    assert result.success is True
    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["direct_messages_topic_id"] == 20189
    assert "reply_parameters" not in api_kwargs
    assert api_kwargs["rich_message"]["markdown"] == TABLE_ONLY_CONTENT


@pytest.mark.asyncio
async def test_finalize_edit_rich_includes_forum_topic_routing():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.edit_message(
        "-100123",
        "555",
        TABLE_ONLY_CONTENT,
        finalize=True,
        metadata={"thread_id": "5"},
    )

    assert result.success is True
    api_kwargs = _rich_edit_kwargs(adapter)
    assert api_kwargs["message_thread_id"] == 5
    assert api_kwargs["rich_message"]["markdown"] == TABLE_ONLY_CONTENT


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
    adapter = _make_adapter(extra={"rich_drafts": True})
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=DANGEROUS_DETAILS_MATH)

    assert result.success is True
    bot.do_api_request.assert_not_called()
    bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_default_uses_legacy_to_avoid_tdesktop_reflow_glitches():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_opt_in_sends_raw_markdown():
    adapter = _make_adapter(extra={"rich_drafts": True})
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
async def test_cjk_rich_content_skips_rich_draft_to_avoid_tdesktop_garble():
    adapter = _make_adapter(extra={"rich_drafts": True})
    adapter._bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=CJK_RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_capability_failure_falls_back_and_latches_off():
    adapter = _make_adapter(extra={"rich_drafts": True})
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
    adapter = _make_adapter(extra={"rich_drafts": True})
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy draft carried this frame
    adapter._bot.send_message_draft.assert_awaited_once()
    # Transient errors must NOT permanently disable rich drafts.
    assert adapter._rich_draft_disabled is False


@pytest.mark.asyncio
async def test_rich_draft_oversized_uses_legacy():
    adapter = _make_adapter(extra={"rich_drafts": True})
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
async def test_legacy_edit_error_logs_redacted_bot_token_without_traceback(monkeypatch, caplog):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter()
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=BadRequest(
            f"Bad Request: https://api.telegram.org/bot{token}/editMessageText"
        )
    )

    with caplog.at_level(logging.WARNING):
        result = await adapter.edit_message(
            "12345", "555", "Just a normal answer.", finalize=True,
        )

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/editMessageText" in result.error
    assert token not in caplog.text
    assert "bot123456789:***/editMessageText" in caplog.text


@pytest.mark.asyncio
async def test_finalize_edit_cjk_rich_content_stays_legacy_to_avoid_tdesktop_garble():
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "12345", "555", CJK_RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.edit_message_text.assert_awaited_once()


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


# --------------------------------------------------------------------------
# Rich-reply recovery (#47375): Telegram does not echo a sendRichMessage's
# content in reply_to_message (.text/.caption empty, .api_kwargs None), so we
# record message_id -> text at send time and recover it on inbound reply.
# --------------------------------------------------------------------------


def _reply_message(reply_to_id, *, reply_text=None, reply_caption=None, quote_text=None):
    """Build a mock inbound reply Message for _build_message_event."""
    replied = SimpleNamespace(
        message_id=int(reply_to_id),
        text=reply_text,
        caption=reply_caption,
    )
    quote = SimpleNamespace(text=quote_text) if quote_text is not None else None
    return SimpleNamespace(
        message_id=999,
        chat=SimpleNamespace(id=12345, type="private", title=None, full_name="U"),
        from_user=SimpleNamespace(
            id=42, username="u", first_name="U", last_name=None,
            full_name="U", is_bot=False,
        ),
        text="what did this mean?",
        caption=None,
        reply_to_message=replied,
        quote=quote,
        message_thread_id=None,
        is_topic_message=False,
        entities=[],
        date=None,
    )


def _reply_message_with_rich_blocks(
    reply_to_id,
    *,
    blocks,
    quote_text=None,
    api_kwargs_factory=dict,
):
    """Build a reply whose echoed content lives only in api_kwargs.rich_message."""
    replied = SimpleNamespace(
        message_id=int(reply_to_id),
        text=None,
        caption=None,
        api_kwargs=api_kwargs_factory({"rich_message": {"blocks": blocks}}),
    )
    quote = SimpleNamespace(text=quote_text) if quote_text is not None else None
    return SimpleNamespace(
        message_id=999,
        chat=SimpleNamespace(id=12345, type="private", title=None, full_name="U"),
        from_user=SimpleNamespace(
            id=42, username="u", first_name="U", last_name=None,
            full_name="U", is_bot=False,
        ),
        text="what did this mean?",
        caption=None,
        reply_to_message=replied,
        quote=quote,
        message_thread_id=None,
        is_topic_message=False,
        entities=[],
        date=None,
    )


@pytest.mark.asyncio
async def test_rich_reply_records_and_recovers_text(monkeypatch, tmp_path):
    """A reply to a rich-sent message resolves the original text via the index."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType
    from gateway import rich_sent_store

    adapter = _make_adapter()

    # _try_send_rich records (chat_id, message_id) -> content on a successful
    # rich send. Drive that path directly so the test doesn't depend on send()
    # gating heuristics (length, content shape) choosing the rich path.
    adapter._bot.do_api_request = AsyncMock(
        return_value=SimpleNamespace(message_id=678)
    )
    send_result = await adapter._try_send_rich(
        "12345", "Your morning briefing: CI is green.", None, None,
    )
    assert send_result is not None and send_result.success is True
    assert send_result.message_id == "678"
    assert rich_sent_store.lookup("12345", "678") == "Your morning briefing: CI is green."

    # Inbound reply carries NO text/caption (the rich-message blind spot).
    event = adapter._build_message_event(
        _reply_message("678"), MessageType.TEXT,
    )
    assert event.reply_to_message_id == "678"
    assert event.reply_to_text == "Your morning briefing: CI is green."


@pytest.mark.asyncio
async def test_rich_reply_lookup_miss_leaves_text_none(monkeypatch, tmp_path):
    """No recorded entry -> reply_to_text stays None, no crash."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message("404"), MessageType.TEXT,
    )
    assert event.reply_to_message_id == "404"
    assert event.reply_to_text is None


@pytest.mark.asyncio
async def test_rich_reply_native_quote_wins_over_lookup(monkeypatch, tmp_path):
    """A native partial quote takes precedence over the send-time index."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType
    from gateway import rich_sent_store

    rich_sent_store.record("12345", "678", "full recorded body")
    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message("678", quote_text="just this part"), MessageType.TEXT,
    )
    assert event.reply_to_text == "just this part"


@pytest.mark.asyncio
async def test_rich_reply_caption_wins_over_lookup(monkeypatch, tmp_path):
    """When Telegram DOES echo a caption, it wins over the index fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType
    from gateway import rich_sent_store

    rich_sent_store.record("12345", "678", "recorded body")
    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message("678", reply_caption="echoed caption"), MessageType.TEXT,
    )
    assert event.reply_to_text == "echoed caption"


@pytest.mark.asyncio
async def test_rich_reply_native_blocks_fill_reply_text_without_index(monkeypatch, tmp_path):
    """Echoed rich_message blocks should recover reply text natively."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType

    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message_with_rich_blocks(
            "678",
            blocks=[
                {"type": "paragraph", "text": ["Hello ", {"type": "bold", "text": "world"}]},
                {"type": "pre", "text": "Line 2"},
            ],
        ),
        MessageType.TEXT,
    )
    assert event.reply_to_text == "Hello world\nLine 2"


@pytest.mark.asyncio
async def test_rich_reply_native_blocks_win_over_index(monkeypatch, tmp_path):
    """Native rich echo should beat the local send-time index fallback."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType
    from gateway import rich_sent_store

    rich_sent_store.record("12345", "678", "recorded body")
    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message_with_rich_blocks(
            "678",
            blocks=[{"type": "paragraph", "text": ["Echoed ", {"type": "italic", "text": "body"}]}],
        ),
        MessageType.TEXT,
    )
    assert event.reply_to_text == "Echoed body"


@pytest.mark.asyncio
async def test_rich_reply_native_blocks_support_mappingproxy_like_api_kwargs(monkeypatch, tmp_path):
    """Duck-type api_kwargs via .get() so mappingproxy-like objects also work."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType

    class MappingProxyLike(dict):
        pass

    adapter = _make_adapter()
    event = adapter._build_message_event(
        _reply_message_with_rich_blocks(
            "678",
            blocks=[
                {"type": "heading", "text": "Status", "size": 2},
                {"type": "list", "items": [{"label": "-", "blocks": [{"type": "paragraph", "text": ["done"]}]}]},
            ],
            api_kwargs_factory=MappingProxyLike,
        ),
        MessageType.TEXT,
    )
    assert event.reply_to_text == "Status\n- done"


@pytest.mark.asyncio
async def test_try_edit_rich_records_streamed_final_for_reply_recovery(monkeypatch, tmp_path):
    """A streamed final finalized via editMessageText must be indexed too.

    The native rich echo covers most replies, but messages that predate the
    bot's first rich send have no echo — so editMessageText must mirror the
    fresh-send index the same way _try_send_rich does.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway import rich_sent_store

    adapter = _make_adapter()
    result = await adapter._try_edit_rich("12345", "5724", "Готово. Основной бот живой.")
    assert result is not None and result.success
    assert rich_sent_store.lookup("12345", "5724") == "Готово. Основной бот живой."


class TestRichStoreCorruptedIndexTrim:
    """Regression: scalar entries in rich_sent_index.json must not crash trim.

    Before the fix, a scalar (non-dict) value persisted by a bad writer or
    manual edit would make the trim branch crash at ``kv[1].get("ts", 0)``
    (AttributeError on a string), and the broad except on line 67 silently
    drops the write — poisoning all future record() calls for the profile.
    The fix sanitizes non-dict entries before trimming.
    """

    def test_scalar_entry_survives_trim(self, monkeypatch, tmp_path):
        """Seed scalar + valid entries, trigger trim, verify records intact."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import json, time
        from gateway import rich_sent_store

        # Lower the cap so record() triggers a trim after adding the new entry.
        monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 3)

        store_dir = tmp_path / "state"
        store_dir.mkdir(parents=True, exist_ok=True)
        store_file = store_dir / "rich_sent_index.json"
        now = int(time.time())
        # 3 valid entries + 1 scalar = 4 total. After sanitize: 3 valid.
        # Add new -> 4 valid > _MAX_ENTRIES(3) -> trim removes oldest.
        store_file.write_text(json.dumps({
            "100:1": {"t": "first valid", "ts": now - 300},
            "100:2": {"t": "second valid", "ts": now - 200},
            "100:3": {"t": "third valid", "ts": now - 100},
            "100:4": "scalar bad entry",
        }))

        rich_sent_store.record("100", "99", "new record")

        # New entry must be retrievable.
        assert rich_sent_store.lookup("100", "99") == "new record"
        # Scalar must have been dropped.
        with open(store_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "100:4" not in data
        # At least the two newest seeded entries survive trim.
        assert rich_sent_store.lookup("100", "2") == "second valid"
        assert rich_sent_store.lookup("100", "3") == "third valid"

    def test_all_scalar_entries_record_succeeds(self, monkeypatch, tmp_path):
        """Index with only scalar entries — record() must not crash."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import json
        from gateway import rich_sent_store

        monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 2)

        store_dir = tmp_path / "state"
        store_dir.mkdir(parents=True, exist_ok=True)
        store_file = store_dir / "rich_sent_index.json"
        store_file.write_text(json.dumps({
            "200:1": "scalar one",
            "200:2": 42,
        }))

        rich_sent_store.record("200", "3", "valid new record")
        assert rich_sent_store.lookup("200", "3") == "valid new record"

    def test_mixed_corruption_preserves_valid_and_new(self, monkeypatch, tmp_path):
        """None, list, and string scalars mixed with valid entries."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import json, time
        from gateway import rich_sent_store

        monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 4)

        store_dir = tmp_path / "state"
        store_dir.mkdir(parents=True, exist_ok=True)
        store_file = store_dir / "rich_sent_index.json"
        now = int(time.time())
        store_file.write_text(json.dumps({
            "300:1": {"t": "old valid", "ts": now - 500},
            "300:2": None,
            "300:3": {"t": "mid valid", "ts": now - 300},
            "300:4": [1, 2, 3],
            "300:5": {"t": "newest valid", "ts": now - 100},
        }))

        rich_sent_store.record("300", "6", "appended")

        assert rich_sent_store.lookup("300", "3") == "mid valid"
        assert rich_sent_store.lookup("300", "5") == "newest valid"
        assert rich_sent_store.lookup("300", "6") == "appended"
        with open(store_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for k in ("300:2", "300:4"):
            assert k not in data
